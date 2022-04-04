from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict

from fed_distill.common.tester import AccuracyTester
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class Trainer(ABC):
    @abstractmethod
    def train_for(self, num_epochs: int = 1) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        raise NotImplementedError()

    def train_epoch(self) ->  Dict[str, Any]:
        return self.train_for(1)


@dataclass
class BasicTrainer(Trainer):
    net: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    criterion: nn.Module
    dataset: Dataset
    batch_size: int
    device: str = "cuda"

    def __post_init__(
        self,
    ) -> None:
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.train_metrics = {"train_losses": []}
        self.epoch = 1

    def _train_batch(self, images, labels) -> float:
        self.optimizer.zero_grad()
        prediction = self.net(images.to(self.device))
        loss = self.criterion(prediction, labels.to(self.device))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def train_for(self, num_epochs: int = 1) -> None:
        self.model.train()
        for _ in range(num_epochs):
            logger.info(f"Training epoch %i", self.epoch)
            epoch_loss = 0
            num_batches = 0
            for images, labels in self.loader:
                epoch_loss += self._train_batch(images, labels)
                num_batches += 1
            epoch_loss /= num_batches
            self.train_metrics["train_losses"].append(epoch_loss)
            self.scheduler.step()
            self.epoch += 1
        return self.train_metrics

    @property
    def model(self):
        return self.net


class AccuracyTrainer(Trainer):
    def __init__(
        self,
        base_trainer: Trainer,
        accuracy_tester: AccuracyTester,
        result_model: nn.Module,
    ) -> None:
        self.base = base_trainer
        self.tester = accuracy_tester
        self.result_model = result_model
        self.test_accs = []
        self.best_model = None
        self.best_acc = 0.0

    @property
    def model(self):
        self.result_model.load_state_dict(self.best_model)
        return self.result_model

    def train_for(self, num_epochs: int = 1) -> Dict[str, Any]:
        base_training_dict = {}
        for _ in range(num_epochs):
            base_training_dict = self.base.train_epoch()
            model = self.base.model
            accuracy = self.tester(model)
            self.test_accs.append(accuracy)
            logger.info("Testing accuracy: %f", accuracy)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_model = deepcopy(model.state_dict())

        final_dict_copy = deepcopy(base_training_dict)
        final_dict_copy["test_accs"] = self.test_accs
        return final_dict_copy
