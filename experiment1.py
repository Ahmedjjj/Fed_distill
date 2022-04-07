import logging
import os
import random
from pathlib import Path
from collections import Counter

import torch
import torchvision.transforms as T
from resnet_cifar import ResNet18
from torch import nn

from fed_distill.cifar10 import load_cifar10_test, load_cifar10_train
from fed_distill.common import (
    AccuracyTester,
    AccuracyTrainer,
    BasicTrainer,
    setup_logger_stdout,
)
from fed_distill.deep_inv import (
    StudentTrainer,
    WeightedSampler,
    DeepInversionGrowingDataset,
)
from fed_distill.fed import HeterogenousDistribution
from fed_distill.resnet import get_resnet_cifar_adi, get_resnet_cifar_di

logger = logging.getLogger("fed_distill")
setup_logger_stdout()


# DeepInversion params
STUDENT_BATCH_SIZE = 256
ADAM_LR = 0.1
L2_SCALE = 0.0
VAR_SCALE = 1e-3
BN_SCALE = 10
COMP_SCALE = 10
BATCH_GRADIENT_UPDATES = 1000

# State and image saving
SAVE_DIR_IMAGES = "/mlodata1/jellouli/federated_learning/experiment1/images"
SAVE_DIR_STATES = "/mlodata1/jellouli/federated_learning/experiment1/states"
MOMENTUM = 0.9

# Common Training params
TRAIN_EPOCHS = 300
TESTING_BATCH_SIZE = 2048

TEACHER_TRAINING_BATCH_SIZE = 2048
EPOCH_GRADIENT_UPDATES = 195
INITIAL_BATCHES = 50

# Cifar10 dataset
DATA_ROOT = "/mlodata1/jellouli"

# data splitting
UNIF_PERCENT = 0.1

LOAD = False

torch.random.manual_seed(42)
random.seed(42)


def main():
    logger.info("Experiment starting")

    logger.info("Creating missing folders")
    Path(SAVE_DIR_IMAGES).mkdir(exist_ok=True, parents=True)
    Path(SAVE_DIR_STATES).mkdir(exist_ok=True, parents=True)

    num_nodes = 2

    logger.info("Splitting dataset using a heterogeneous distribution")
    train_dataset = load_cifar10_train(DATA_ROOT)
    train_dataset_node1, train_dataset_node2 = HeterogenousDistribution(
        train_dataset, unif_percentage=UNIF_PERCENT
    ).split(num_nodes)
    label_counter_node1 = Counter(train_dataset_node1.targets)
    label_counter_node2 = Counter(train_dataset_node2.targets)
    logger.info("node 1 dataset length %i", len(train_dataset_node1))
    logger.info("node 2 dataset length %i", len(train_dataset_node2))
    logger.info("Node 1 labels: %s", str(label_counter_node1))
    logger.info("Node 2 labels: %s", str(label_counter_node2))

    test_dataset = load_cifar10_test(DATA_ROOT)
    test_dataset_node1, test_dataset_node2 = HeterogenousDistribution(
        test_dataset, unif_percentage=UNIF_PERCENT
    ).split(num_nodes)
    logger.info("node 1 test dataset length %i", len(test_dataset_node1))
    logger.info("node 2 test dataset length %i", len(test_dataset_node2))

    logger.info("Creating accuracy testers")
    tester_all = AccuracyTester(test_dataset, batch_size=TESTING_BATCH_SIZE)
    tester_node1 = AccuracyTester(test_dataset_node1, batch_size=TESTING_BATCH_SIZE)
    tester_node2 = AccuracyTester(test_dataset_node2, batch_size=TESTING_BATCH_SIZE)

    resnet18_factory = lambda: ResNet18().to("cuda")
    logger.info(f"Creating Teacher node")

    teacher_model = resnet18_factory()
    teacher_optimizer = torch.optim.SGD(
        teacher_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    teacher_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        teacher_optimizer, milestones=[150, 250], gamma=0.1
    )
    teacher_trainer = AccuracyTrainer(
        BasicTrainer(
            teacher_model,
            teacher_optimizer,
            teacher_scheduler,
            nn.CrossEntropyLoss(),
            train_dataset_node1,
            batch_size=TEACHER_TRAINING_BATCH_SIZE,
        ),
        tester_all,
        result_model=resnet18_factory(),
    )

    if LOAD:
        saved_state = torch.load(os.path.join(SAVE_DIR_STATES, "teacher.pt"))
        teacher_model.load_state_dict(saved_state["best_model"])
    else:
        logger.info("Training teacher node locally")
        training_metrics = teacher_trainer.train_for(TRAIN_EPOCHS)
        training_metrics["test_acc_split_1"] = tester_node1(teacher_model)
        training_metrics["test_acc_split_2"] = tester_node2(teacher_model)
        logger.info("Teacher best acc: %f", max(training_metrics["test_accs"]))
        torch.save(training_metrics, os.path.join(SAVE_DIR_STATES, "teacher.pt"))

    logger.info("Creating Student Node")
    student_model = resnet18_factory()
    target_sampler = WeightedSampler(
        batch_size=STUDENT_BATCH_SIZE, labels=train_dataset_node1.targets
    )
    student_non_adaptive_di = get_resnet_cifar_di(
        class_sampler=target_sampler,
        adam_lr=ADAM_LR,
        l2_scale=L2_SCALE,
        bn_scale=BN_SCALE,
        var_scale=VAR_SCALE,
        batch_size=STUDENT_BATCH_SIZE,
        grad_updates_batch=BATCH_GRADIENT_UPDATES,
    )
    student_dataset = DeepInversionGrowingDataset(
        base_dataset=train_dataset_node2,
        teacher_net=teacher_model,
        deep_inversion=student_non_adaptive_di,
        num_initial_batches=INITIAL_BATCHES,
        transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),]),
    )
    student_adaptive_di = get_resnet_cifar_adi(
        class_sampler=target_sampler,
        adam_lr=ADAM_LR,
        l2_scale=L2_SCALE,
        var_scale=VAR_SCALE,
        bn_scale=BN_SCALE,
        comp_scale=COMP_SCALE,
        batch_size=STUDENT_BATCH_SIZE,
        grad_updates_batch=BATCH_GRADIENT_UPDATES,
    )

    student_optimizer = torch.optim.SGD(
        student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    student_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        student_optimizer, milestones=[150, 250], gamma=0.1
    )

    student_trainer = AccuracyTrainer(
        StudentTrainer(
            student_net=student_model,
            teacher_net=teacher_model,
            deep_inversion=student_adaptive_di,
            criterion=nn.CrossEntropyLoss(),
            optimizer=student_optimizer,
            scheduler=student_scheduler,
            dataset=student_dataset,
            epoch_gradient_updates=EPOCH_GRADIENT_UPDATES,
            new_batches_per_epoch=2,
        ),
        tester_all,
        result_model=resnet18_factory(),
    )

    training_metrics = student_trainer.train_for(TRAIN_EPOCHS)
    training_metrics["test_acc_split_1"] = tester_node1(student_model)
    training_metrics["test_acc_split_2"] = tester_node2(student_model)
    torch.save(training_metrics, os.path.join(SAVE_DIR_STATES, "student.pt"))

    student_dataset.save(os.path.join(SAVE_DIR_IMAGES, "generated_batches.pt"))


if __name__ == "__main__":
    main()
