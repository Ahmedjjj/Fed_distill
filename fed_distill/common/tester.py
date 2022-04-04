from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch


class Tester(ABC):
    @abstractmethod
    def __call__(self, model: nn.Module, **kwargs) -> Any:
        raise NotImplementedError()


class AccuracyTester(Tester):
    def __init__(
        self, dataset: Dataset, batch_size: int = 2048, device: str = "cuda"
    ) -> None:
        self.test_loader = DataLoader(dataset, batch_size=batch_size)
        self.len_test = len(dataset)
        self.device = device

    def __call__(self, model: nn.Module) -> float:
        model.eval()
        with torch.no_grad():
            num_correct = 0
            for images, labels in self.test_loader:
                result = model(images.to(self.device))
                pred = result.argmax(1)
                num_correct += (pred == labels.to(self.device)).sum()
            
            return float(num_correct / self.len_test)
