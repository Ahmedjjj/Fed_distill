import logging
from dataclasses import dataclass
from sched import scheduler

import torch
import torch.nn as nn
from fed_distill.common.state import StateSaver
from fed_distill.common.tester import AccuracyTester
from fed_distill.common.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from resnet_cifar import ResNet18

logger = logging.getLogger(__name__)


class DatasetTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        dataset: Dataset,
        batch_size: int,
        state_saver: StateSaver,
        device: str = "cuda",
    ) -> None:
        self.model_ = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
        self.state_saver = state_saver
        self.state_saver.add("train_losses", [])
        self.device = device
        self.model_.to(device)

    def _train_epoch(self, images, labels) -> float:
        self.optimizer.zero_grad()
        prediction = self.model_(images.to(self.device))
        loss = self.criterion(prediction, labels.to(self.device))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def train_for(self, num_epochs: int = 1) -> None:
        self.model.train()
        for _ in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for images, labels in self.loader:
                epoch_loss += self._train_epoch(images, labels)
                num_batches += 1
            epoch_loss /= num_batches
            self.state_saver["train_losses"].append(epoch_loss)
            self.scheduler.step()
    
    @property
    def model(self):
        return self.model_


# @dataclass
# class ResnetTrainer(DatasetTrainer):
#     def __init__(
#         self,
#         train_dataset: Dataset,
#         test_dataset: Dataset,
#         train_batch_size: int,
#         test_batch_size:int,
#         state_saver: StateSaver,
#         training_epochs:
#         device: str = "cuda",
        
#     ) -> None:
#         self.tester = AccuracyTester(dataset=test_dataset, batch_size=test_batch_size)
#         state_saver.add("test_accs", [])

#         model = ResNet18()
#         optimizer = torch.optim.SGD(
#             model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
#         )
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(
#             optimizer, milestones=[150, 250], gamma=0.1
#         )
#         self.epochs = 
#         super().__init__()

    # def __post_init__(self):
    #     self.train_loader = 
    #     self.test_loader = DataLoader(
    #         self.test_dataset, batch_size=self.test_batch_size, num_workers=1
    #     )
    #     self.criterion = nn.CrossEntropyLoss()
    #     
    #     self.resnet.to(self.device)
    #     self.criterion.to(self.device)
    #     self.epoch = 1
    #     self.epoch_train_losses = []
    #     self.epoch_test_losses = []
    #     self.test_accs = []
    #     self.best_test_acc = -1
    #     self.best_state_dict = None


