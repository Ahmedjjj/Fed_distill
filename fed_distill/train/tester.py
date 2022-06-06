from typing import Callable, Union

import torch.nn as nn
from torch.utils.data import DataLoader
import torch

class AccuracyTester:
    def __init__(self, loader: DataLoader, device: Union[torch.device, str]="cuda") -> None:
        self.loader = loader
        self.device = device
    
    def __call__(self, model: nn.Module) -> float:
        restore = model.training
        model.eval()
        correct = 0
        num_samples = 0
        with torch.no_grad():
            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                correct += (model(images).argmax(1) == labels).sum()
                num_samples += len(labels)

        if restore:
            model.train()

        return float(correct / num_samples)      


def get_batch_accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    restore = model.training
    model.eval()
    with torch.no_grad():
        acc = float(torch.sum(model(images).argmax(dim=1) == labels) / len(labels))
        if restore:
            model.train()
        return acc