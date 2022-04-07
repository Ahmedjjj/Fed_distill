import os
from pathlib import Path
import torch
from torch import nn 
import torchvision.transforms as T
import random

from fed_distill.cifar10.helpers import load_cifar10_test
from fed_distill.common.tester import AccuracyTester
from fed_distill.common.trainer import AccuracyTrainer
from fed_distill.deep_inv.growing_dataset import DeepInversionGrowingDataset
from fed_distill.deep_inv.student import StudentTrainer
from fed_distill.resnet import get_resnet_cifar_di
from fed_distill.resnet.deep_inv import get_resnet_cifar_adi
from fed_distill.deep_inv import RandomSampler
from fed_distill.common import setup_logger_stdout

from resnet_cifar import ResNet18

setup_logger_stdout()

TEACHER_WEIGHTS = (
    "/home/jellouli/dataset-distillation/model_weights/final_state_resnet.tar"
)

BATCH_SIZE = 256
ADAM_LR = 0.1
L2_SCALE = 0.0
VAR_SCALE = 1e-3
BN_SCALE = 10
COMP_SCALE = 10
BATCH_GRADIENT_UPDATES = 1000
TRAIN_EPOCHS = 300
EPOCH_GRADIENT_UPDATES = 195
INITIAL_BATCHES = 50

SAVE_DIR_IMAGES = "/mlodata1/jellouli/student_training/adaptive5/images"
SAVE_DIR_STATES = "/mlodata1/jellouli/student_training/adaptive5/states"
DATA_ROOT = "/mlodata1/jellouli"

torch.random.manual_seed(42)
random.seed(42)

def main():

    Path(SAVE_DIR_IMAGES).mkdir(exist_ok=True, parents=True)
    Path(SAVE_DIR_STATES).mkdir(exist_ok=True, parents=True)


    sampler = RandomSampler(batch_size=BATCH_SIZE)
    teacher = ResNet18().to("cuda")

    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS)["best_model"])

    student_non_adaptive_di = get_resnet_cifar_di(
        class_sampler=sampler,
        adam_lr=ADAM_LR,
        l2_scale=L2_SCALE,
        bn_scale=BN_SCALE,
        var_scale=VAR_SCALE,
        batch_size=BATCH_SIZE,
        grad_updates_batch=BATCH_GRADIENT_UPDATES,
    )
    student_dataset = DeepInversionGrowingDataset(
        base_dataset=dict(),
        teacher_net=teacher,
        deep_inversion=student_non_adaptive_di,
        num_initial_batches=INITIAL_BATCHES,
        transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),]),
    )
    student_adaptive_di = get_resnet_cifar_adi(
        class_sampler=sampler,
        adam_lr=ADAM_LR,
        l2_scale=L2_SCALE,
        var_scale=VAR_SCALE,
        bn_scale=BN_SCALE,
        comp_scale=COMP_SCALE,
        batch_size=BATCH_SIZE,
        grad_updates_batch=BATCH_GRADIENT_UPDATES,
    )

    student = ResNet18().to("cuda")
    student_optimizer = torch.optim.SGD(
        student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    student_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        student_optimizer, milestones=[150, 250], gamma=0.1
    )

    student_trainer = AccuracyTrainer(
        StudentTrainer(
            student_net=student,
            teacher_net=teacher,
            deep_inversion=student_adaptive_di,
            criterion=nn.CrossEntropyLoss(),
            optimizer=student_optimizer,
            scheduler=student_scheduler,
            dataset=student_dataset,
            epoch_gradient_updates=EPOCH_GRADIENT_UPDATES,
            new_batches_per_epoch=2,
        ),
        AccuracyTester(load_cifar10_test(DATA_ROOT), batch_size=2048),
        result_model=ResNet18(),
    )
    training_metrics = student_trainer.train_for(TRAIN_EPOCHS)
    torch.save(training_metrics, os.path.join(SAVE_DIR_STATES, "student.pt"))
    student_dataset.save(os.path.join(SAVE_DIR_IMAGES, "generated_batches.pt"))


    


if __name__ == "__main__":
    main()
