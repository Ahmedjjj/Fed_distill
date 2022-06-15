import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from apex import amp
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fed_distill.data import (
    RandomSampler,
    extract_subset,
    GrowingDataset,
    mix_iterators,
)
from fed_distill.cifar10 import CIFAR10_TEST_TRANSFORM
from fed_distill.train import AccuracyTester

logger = logging.getLogger("fed_distill")


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    t_cfg = cfg.teacher
    if "seed" in cfg:
        logger.info("Setting seed to %i", cfg.seed)
        torch.random.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    train_dataset = instantiate(cfg.dataset.train)
    test_dataset = instantiate(cfg.dataset.test, transform=CIFAR10_TEST_TRANSFORM)

    split = None
    if "split" in cfg:
        with open(cfg.split.save_file) as buffer:
            split = json.load(buffer)

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
            AccuracyTester(DataLoader(teacher_test_dataset, batch_size=4098))(teacher),
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

        if cfg.amp:
            teacher, optimizer = amp.initialize(teacher, optimizer, opt_level="O1")

        di = instantiate(cfg.deep_inv.di)(
            optimizer=optimizer, teacher=teacher, use_amp=cfg.amp
        )

        deep_invs.append(di.iterator_from_sampler(sampler))

    dataset = GrowingDataset(stream=mix_iterators(deep_invs))

    save_file = Path(cfg.initial.save_path)

    if not save_file.parent.exists():
        save_file.parent.mkdir(parents=True)

    for i in range(cfg.initial.num_batches):
        dataset.grow()

    dataset.save(save_file)


if __name__ == "__main__":
    main()
