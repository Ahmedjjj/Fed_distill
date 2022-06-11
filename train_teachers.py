import json
import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import numpy as np

from fed_distill.data import extract_subset
from fed_distill.train import AccuracyTester, Trainer
from fed_distill.cifar10 import CIFAR10_TRAIN_TRANSFORM, CIFAR10_TEST_TRANSFORM
logger = logging.getLogger("fed_distill")


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    t_cfg = cfg.teacher
    if "seed" in cfg:
        logger.info("Setting seed to %i", cfg.seed)
        torch.random.manual_seed(cfg.seed)

    train_dataset = instantiate(cfg.dataset.train, transform=CIFAR10_TRAIN_TRANSFORM)
    test_dataset = instantiate(cfg.dataset.test, transform=CIFAR10_TEST_TRANSFORM)

    split = None
    if "split" in cfg:
        with open(cfg.split.save_file) as buffer:
            split = json.load(buffer)

    for t in range(t_cfg.num_teachers):
        model = instantiate(t_cfg.model).to("cuda")
        optimizer = instantiate(t_cfg.optimizer)(model.parameters())
        scheduler = instantiate(t_cfg.scheduler)(optimizer)
        criterion = instantiate(t_cfg.criterion)

        teacher_train_dataset = (
            extract_subset(train_dataset, split[f"teacher{t}"]["train"])
            if split
            else train_dataset
        )
        teacher_test_dataset = (
            extract_subset(test_dataset, split[f"teacher{t}"]["test"])
            if split
            else test_dataset
        )
        logger.info("Teacher %i train dataset classes: %s", t, str(np.unique(teacher_train_dataset.targets)))
        logger.info("Teacher %i test dataset classes: %s", t, str(np.unique(teacher_test_dataset.targets)))

        tester = AccuracyTester(DataLoader(teacher_test_dataset, t_cfg.batch_size_test))
        loader = DataLoader(teacher_train_dataset, t_cfg.batch_size_train, shuffle=True)

        trainer = Trainer(
            model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=loader,
            accuracy_criterion=tester,
        )

        trainer.train(t_cfg.train_epochs)

        save_folder = Path(t_cfg.save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        metrics_save_path = save_folder / f"metrics_teacher{t}"
        model_save_path = save_folder / f"model_teacher{t}"
        metrics = trainer.metrics
        torch.save(metrics["best_model"], model_save_path)
        metrics.pop("best_model")
        torch.save(metrics, metrics_save_path)


if __name__ == "__main__":
    main()
