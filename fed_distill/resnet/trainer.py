import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class ResnetTrainer:
    resnet: nn.Module
    train_dataset: Dataset
    test_dataset: Dataset
    train_batch_size: int
    test_batch_size: int
    num_workers: int = 8
    device: str = "cuda"
    save_path: str = "./final_state_resnet.tar"

    def __post_init__(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.test_batch_size, num_workers=1
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        self.resnet.to(self.device)
        self.criterion.to(self.device)
        self.epoch = 1
        self.epoch_train_losses = []
        self.epoch_test_losses = []
        self.test_accs = []
        self.best_test_acc = -1
        self.best_state_dict = None

    def train_epoch(self):
        lr = 0.1
        if 80 <= self.epoch < 120:
            lr = 0.01
        elif self.epoch >= 120:
            lr = 0.001

        for g in self.optimizer.param_groups:
            g["lr"] = lr

        epoch_loss = 0
        self.resnet.train()
        logger.info(f"Starting train epoch {self.epoch}")
        for i, (images, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            prediction = self.resnet(images.to(self.device))
            loss = self.criterion(prediction, labels.to(self.device))
            epoch_loss += float(loss)
            loss.backward()
            self.optimizer.step()

        self.epoch_train_losses.append(epoch_loss / len(self.train_dataset))
        logger.info(
            f"Training loss at epoch {self.epoch}: {epoch_loss / len(self.train_dataset):.3f}"
        )

    def test_epoch(self):
        self.resnet.eval()
        with torch.no_grad():
            epoch_loss = 0
            num_correct = 0
            logger.info(f"Starting test epoch {self.epoch}")
            for i, (images, labels) in enumerate(self.test_loader):
                prediction = self.resnet(images.to(self.device))
                loss = self.criterion(prediction, labels.to(self.device))
                epoch_loss += float(loss)
                pred_class = prediction.argmax(1)
                num_correct += (pred_class == labels.to(self.device)).sum()

            self.epoch_test_losses.append(epoch_loss / len(self.test_dataset))
            logger.info(
                f"Test loss at epoch {self.epoch}: {epoch_loss / len(self.test_dataset):.3f}"
            )
            acc = num_correct / len(self.test_dataset)
            logger.info(f"Test acc at epoch {self.epoch}: {acc:.3f}")
            self.test_accs.append(acc)
            if acc > self.best_test_acc:
                self.best_test_acc = acc
                self.best_state_dict = self.resnet.state_dict()

    def increment_epoch(self):
        self.epoch += 1

    def train(self, max_epoch):
        for _ in range(1, max_epoch + 1):
            self.train_epoch()
            self.test_epoch()
            self.increment_epoch()

        torch.save(
            {
                "train_losses": self.epoch_train_losses,
                "test_losses": self.epoch_test_losses,
                "test_accs": self.test_accs,
                "best_acc": self.best_test_acc,
                "best_model": self.best_state_dict,
            },
            self.save_path,
        )
