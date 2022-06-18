import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from hydra.utils import instantiate
from omegaconf import DictConfig

from fed_distill.cifar10 import (
    CIFAR10_TEST_TRANSFORM,
    CIFAR10_TRAIN_TRANSFORM,
    CIFAR10_INVERSION_TRANSFORM,
)
from fed_distill.data import (
    GrowingDataset,
    GrowingDatasetLoader,
    RandomSampler,
    extract_subset,
    mix_iterators,
)
from fed_distill.train import AccuracyTester, Trainer

logger = logging.getLogger("fed_distill")


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    t_cfg = cfg.teacher
    s_cfg = cfg.student
    if "seed" in s_cfg:
        logger.info("Setting seed to %i", s_cfg.seed)
        torch.random.manual_seed(cfg.student.seed)
        random.seed(cfg.student.seed)
        np.random.seed(cfg.student.seed)

    train_dataset = instantiate(cfg.dataset.train, transform=CIFAR10_TRAIN_TRANSFORM)
    test_dataset = instantiate(cfg.dataset.test, transform=CIFAR10_TEST_TRANSFORM)

    split = None
    if "split" in cfg:
        with open(cfg.split.save_file) as buffer:
            split = json.load(buffer)

    if s_cfg.adaptive:
        logger.info("Student will be trained adaptively")
    
    student = instantiate(s_cfg.model).to("cuda")

    deep_invs_o = []
    deep_invs = []
    teacher_save_folder = Path(t_cfg.save_folder)

    for i in range(t_cfg.num_teachers):
        weights = torch.load(teacher_save_folder / f"model_teacher{i}.pt")
        teacher = instantiate(t_cfg.model).to("cuda")
        teacher.load_state_dict(weights)

        teacher_dataset = (
            extract_subset(train_dataset, split[f"teacher{i}"]["train"])
            if split
            else train_dataset
        )
        teacher_test_dataset = (
            extract_subset(test_dataset, split[f"teacher{i}"]["test"])
            if split
            else test_dataset
        )
        dataset_targets = tuple(np.unique(teacher_dataset.targets).tolist())
        logger.info("Teacher %i train dataset classes: %s", i, str(dataset_targets))
        logger.info(
            "Teacher %i test dataset classes: %s",
            i,
            str(tuple(np.unique(teacher_test_dataset.targets))),
        )
        logger.info(
            "Teacher %i test accuracy %f",
            i,
            AccuracyTester(DataLoader(teacher_test_dataset, batch_size=s_cfg.batch_size_train))(teacher),
        )

        sampler = RandomSampler(
            batch_size=cfg.deep_inv.batch_size, classes=dataset_targets
        )
        inputs = torch.randn(
            (cfg.deep_inv.batch_size, *cfg.dataset.input_size),
            requires_grad=True,
            device="cuda",
        )
        optimizer = instantiate(cfg.deep_inv.optimizer)([inputs])

        if s_cfg.adaptive:
            di = instantiate(cfg.deep_inv.adi, loss={"classes": dataset_targets})(
                optimizer=optimizer, teacher=teacher, student=student, use_amp=cfg.amp
            )
        else:
            di = instantiate(cfg.deep_inv.di)(
                optimizer=optimizer, teacher=teacher, use_amp=cfg.amp
            )
        deep_invs_o.append(di)
        deep_invs.append(di.iterator_from_sampler(sampler))

    initial_dataset = None
    if "split" in cfg and len(split.keys()) > t_cfg.num_teachers:
        logger.info("Loading subset of training data to student")
        initial_datasets = []
        for i in range(t_cfg.num_teachers, len(split.keys())):
            logger.info(f"Loading subset corresponding to teacher %i", i)
            subset = extract_subset(train_dataset, split[f"teacher{i}"]["train"])
            logger.info("Student targets %s", str(np.unique(subset.targets)))
            initial_datasets.append(
                subset
            )
        initial_dataset = ConcatDataset(initial_datasets)

    student_dataset = GrowingDataset(
        stream=mix_iterators(deep_invs),
        base_dataset=initial_dataset,
        new_batch_transform=CIFAR10_INVERSION_TRANSFORM,
    )
    student_dataset.load(cfg.initial.save_path)

    student_loader = GrowingDatasetLoader(
        student_dataset,
        epoch_samples=len(train_dataset),
        new_batches_per_epoch=s_cfg.new_batches_per_epoch,
        batch_size=cfg.deep_inv.batch_size,
        shuffle=True,
    )

    optimizer = instantiate(s_cfg.optimizer)(student.parameters())
    scheduler = instantiate(s_cfg.scheduler)(optimizer)
    tester = AccuracyTester(DataLoader(test_dataset, s_cfg.batch_size_test))
    criterion = instantiate(s_cfg.criterion)
    trainer = Trainer(
        model=student,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        loader=student_loader,
        accuracy_criterion=tester,
    )
    trainer.train(s_cfg.train_epochs)

    save_folder = Path(s_cfg.save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    train_metrics = trainer.metrics

    model = train_metrics.pop("best_model")
    torch.save(model, save_folder / "student.pt")
    torch.save(train_metrics, save_folder / "metrics.pt")

    for i, di in enumerate(deep_invs_o):
        torch.save(di.metrics, save_folder / f"di_metrics_teacher{i}.pt")
    
    student_dataset.save(save_folder / "dataset.pt")


if __name__ == "__main__":
    main()
