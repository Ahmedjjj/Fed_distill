import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict

from fed_distill.common.tester import AccuracyTester
from torch import nn

logger = logging.getLogger(__name__)


class Trainer(ABC):
    @abstractmethod
    def train_for(self, num_epochs: int = 1) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        raise NotImplementedError()

    def train_epoch(self) -> None:
        self.train_for(1)


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

    def train_for(self, num_epochs: int = 1) -> None:
        for _ in range(num_epochs):
            self.base.train_epoch()
            model = self.base.model
            accuracy = self.tester(model)
            logger.info("Accuracy: %d", accuracy)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_model = deepcopy(model.state_dict())
