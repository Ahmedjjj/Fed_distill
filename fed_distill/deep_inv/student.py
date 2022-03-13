import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
from fed_distill.deep_inv.deep_inv import DeepInversion
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class StudentTrainer:
    deep_inversion: DeepInversion
    test_dataset: Dataset
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    test_batch_size: int = 4098
    training_epochs: int = 250
    epoch_gradient_updates: int = 195
    generation_steps: List[int] = field(default=list)
    initial_batches: List[Tuple(torch.Tensor, torch.Tensor)] = field(default=list)
    device: str = "cuda"
    save_prefix: str = ""
    save_images: bool = False
    imgs_save_folder: str = "."
    save_state: bool = False
    save_folder: str = "."
    save_every: int = None

    def __post_init__(self) -> None:
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.test_batch_size
        )
        self.student_test_accs = []
        self.student_train_accs = []
        self.teacher_train_accs = []
        self.best_model = self.student_net.state_dict()
        self.best_acc = 0.0

        self.all_batches = self.initial_batches.copy()
        del self.initial_batches

        self.student_net, self.teacher_net = (
            self.deep_inversion.student_net,
            self.deep_inversion.teacher_net,
        )
        self.epoch = 1
        self.current_batch = 0

    def _get_test_acc(self) -> float:
        self.student_net.eval()
        with torch.no_grad():
            num_correct = 0
            for input, targets in self.test_loader:
                num_correct += torch.sum(
                    self.student_net(input).argmax(dim=1) == targets
                )
            return float(num_correct / len(self.test_dataset))

    def _get_batch_acc(
        self, inputs: torch.Tensor, targets: torch.Tensor, teacher=False
    ) -> float:
        net = self.teacher_net if teacher else self.student_net
        with torch.no_grad():
            return float(torch.sum(net(inputs).argmax(dim=1) == targets) / len(targets))

    def _train_on_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        output = self.student_net(inputs.to(self.device))
        loss = self.criterion(output.to(self.device), targets.to(self.device))
        loss.backwards()
        self.optimizer.step()

    def _save_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        torch.save(
            {"images": inputs.cpu(), "labels": targets.cpu()},
            os.path.join(
                self.imgs_save_folder,
                self.save_prefix + f"_batch{self.current_batch}.tar",
            ),
        )

    def _incr_epoch(self):
        self.epoch += 1
        self.scheduler.step()

    def _save_state(self):
        state_dict = {
            "model": self.student_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "batch": self.current_batch,
            "student_test_accs": self.student_test_accs,
            "student_train_accs": self.student_train_accs,
            "teacher_test_accs": self.teacher_train_accs,
            "best_acc": self.best_acc,
            "best_model": self.best_model,
        }
        torch.save(
            state_dict,
            os.path.join(self.save_folder, self.save_prefix + "_epoch{self.epoch}.tar"),
        )

    def train_epoch(self):
        logger.info(f"Starting training of epoch {self.epoch}")
        self.student_net.train()
        for step in range(self.epoch_gradient_updates):
            logger.info(f"Starting batch {step + 1}")
            if step in self.generation_steps:
                logger.info(f"Generating data batch {self.current_batch}")
                new_inputs, new_targets = self.deep_inversion.compute_batch()
                if self.save_images:
                    self._save_batch(new_inputs, new_targets)

                student_acc = self._get_batch_acc(new_inputs, new_targets)
                teacher_acc = self._get_batch_acc(new_inputs, new_targets, teacher=True)
                logger.info(f"Student accuracy on new batch: {student_acc}")
                logger.info(f"Teacher accuracy on new batch: {teacher_acc}")
                self.student_train_accs.append(student_acc)
                self.teacher_train_accs.append(teacher_acc)

                new_inputs = new_inputs.detach()
                new_targets = new_targets.detach()
                self._train_on_batch(new_inputs, new_targets)

                self.all_batches.append((new_inputs.cpu(), new_targets.cpu()))

            else:
                training_batch_index = random.randint(0, len(self.all_batches) - 1)
                inputs, targets = self.all_batches[training_batch_index]
                self._train_on_batch(inputs, targets)

    def train(self, num_epochs):
        for _ in range(num_epochs):
            self.train_epoch()
            test_acc = self._get_test_acc()
            logger.info(f"Student test accuracy after epoch {self.epoch} : {test_acc}")
            self.student_test_accs.append(test_acc)
            if self.save_state and self.epoch % self.save_every == 0:
                self._save_state()
            self._incr_epoch()
