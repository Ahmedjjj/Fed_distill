import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from fed_distill.train.tester import get_batch_accuracy

logger = logging.getLogger(__name__)


@dataclass
class Trainer:
    model: nn.Module
    criterion: nn.Module
    optimizer: Optimizer
    scheduler: _LRScheduler
    loader: DataLoader
    accuracy_criterion: Optional[Callable[[nn.Module], float]] = None
    device: Union[str, torch.device] = "cuda"

    def __post_init__(self):
        self.criterion.to(self.device)
        self.model.to(self.device)
        self._metrics = {"training_loss": []}
        self._metrics = {"training_acc": []}

        if self.accuracy_criterion:
            self._metrics["test_acc"] = []
            self._metrics["best_model"] = []
            self.best_acc = 0

    def _train_epoch(self) -> None:
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        for images, labels in self.loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            prediction = self.model(images)
            loss = self.criterion(prediction, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += float(loss) * len(labels)
            epoch_acc += get_batch_accuracy(self.model, images, labels) * len(labels)

        epoch_loss /= len(self.loader)
        epoch_acc /= len(self.loader)
        self._metrics["training_loss"].append(epoch_loss)
        self._metrics["training_acc"].append(epoch_acc)
        logger.info("Training loss %f", epoch_loss)
        logger.info("Training acc %f", epoch_acc)

        if self.accuracy_criterion:
            model_acc = self.accuracy_criterion(self.model)
            logger.info("Test acc: %f", model_acc)
            self._metrics["test_acc"].append(model_acc)
            if model_acc > self.best_acc:
                logger.info("Accuracy improved")
                self.best_acc = model_acc
                self._metrics["best_model"] = self.model.state_dict()

        self.scheduler.step()

    @property
    def metrics(self):
        return deepcopy(self._metrics)

    def train(self, num_epochs: int) -> None:
        for _ in tqdm(range(num_epochs)):
            self._train_epoch()
