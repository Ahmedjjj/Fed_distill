from abc import ABC, abstractmethod
from fileinput import filename

from fed_distill.common.state import StateSaver

from fed_distill.common.tester import AccuracyTester
import logging
from torch import nn
import copy

from resnet_cifar import ResNet18

logger = logging.getLogger(__name__)


class Trainer(ABC):
    @abstractmethod
    def train_for(self, num_epochs: int = 1) -> None:
        raise NotImplementedError()

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        raise NotImplementedError()

    def train_epoch(self) -> None:
        self.train_for(1)


class AccuracySelectionTrainer(Trainer):
    def __init__(
        self,
        base_trainer: Trainer,
        accuracy_tester: AccuracyTester,
        state_saver: StateSaver,
    ) -> None:
        self.base = base_trainer
        self.tester = accuracy_tester
        self.state_saver = state_saver
        self.state_saver.add("best_acc", 0.0)
        self.state_saver.add("best_model", self.base.model.state_dict())

    @property
    def model(self):
        model = ResNet18() #self.base.model.__class__(block=self.base.model.block, num_blocks=self.base.model.num_blocks)
        model.load_state_dict(self.state_saver["best_model"])
        return model

    def train_for(self, num_epochs: int = 1) -> None:
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}")
            self.base.train_epoch()
            model = self.base.model
            accuracy = self.tester(model)
            logger.info(f"Testing accuracy at epoch {epoch + 1}: {accuracy}")
            if accuracy > self.state_saver["best_acc"]:
                self.state_saver["best_acc"] = accuracy
                self.state_saver["best_model"] = model.state_dict()
                self.state_saver.save(name="current_best")
            self.state_saver.checkpoint()
