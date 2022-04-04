import logging
from dataclasses import dataclass
import math
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from fed_distill.common.trainer import Trainer
from fed_distill.deep_inv.deep_inv import DeepInversion
from fed_distill.deep_inv.growing_dataset import GrowingDataset


logger = logging.getLogger(__name__)


def get_batch_accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor):
    model.eval()
    with torch.no_grad():
        return float(torch.sum(model(images).argmax(dim=1) == labels) / len(labels))


def cycle_loader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            continue


@dataclass
class StudentTrainer(Trainer):
    student_net: nn.Module
    teacher_net: nn.Module
    deep_inversion: DeepInversion
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    dataset: GrowingDataset
    epoch_gradient_updates: int = 195
    new_batches_per_epoch: int = 2
    device: int = "cuda"

    def __post_init__(self) -> None:
        self.epoch = 1
        self.current_batch = 1
        self.train_metrics = {
            "student": {"train_accs": [], "instant_train_accs": []},
            "teacher": {"instant_train_accs": []},
        }
        self.teacher_net.eval()
        self.generation_steps = range(
            0,
            self.epoch_gradient_updates,
            math.ceil(self.epoch_gradient_updates / self.new_batches_per_epoch),
        )
        logger.info(
            "Generating new batches at steps %s", str(list(self.generation_steps))
        )

        self.train_loader = cycle_loader(
            DataLoader(
                self.dataset,
                batch_size=self.deep_inversion.batch_size,
                shuffle=True,
                num_workers=8,
            )
        )

    def _get_batch(self):
        logger.info("Generating data batch %i", self.current_batch)
        self.student_net.eval()
        new_inputs, new_targets = self.deep_inversion.compute_batch(
            teacher_net=self.teacher_net, student_net=self.student_net
        )
        self.current_batch += 1
        return new_inputs.to(self.device), new_targets.to(self.device)

    def _train_on_batch(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> None:
        self.student_net.train()
        self.optimizer.zero_grad()
        output = self.student_net(images.to(self.device))
        loss = self.criterion(output, targets.to(self.device))
        loss.backward()
        self.optimizer.step()

    def _incr_epoch(self):
        self.epoch += 1
        self.scheduler.step()

    def _train_epoch(self):
        logger.info("Starting training of epoch %i", self.epoch)
        self.student_net.train()
        epoch_acc = 0
        for step in range(self.epoch_gradient_updates):
            logger.debug("step %i", step)
            if step in self.generation_steps:
                # Grow dataset
                new_inputs, new_targets = self._get_batch()
                self.dataset.add_batch(new_inputs, new_targets)

                student_acc = get_batch_accuracy(
                    self.student_net, new_inputs, new_targets
                )
                teacher_acc = get_batch_accuracy(
                    self.teacher_net, new_inputs, new_targets
                )
                self.train_metrics["student"]["instant_train_accs"].append(student_acc)
                self.train_metrics["teacher"]["instant_train_accs"].append(teacher_acc)
                logger.info("Student accuracy on new batch: %f", student_acc)
                logger.info("Teacher accuracy on new batch: %f", teacher_acc)

                images, labels = self.dataset.get_last_batch()
            else:
                images, labels = next(self.train_loader)

            self._train_on_batch(images, labels)
            epoch_acc += get_batch_accuracy(
                self.student_net, images.to(self.device), labels.to(self.device)
            )

        epoch_acc /= self.epoch_gradient_updates
        logger.info("Student training acc: %f", epoch_acc)
        self.train_metrics["student"]["train_accs"].append(epoch_acc)
        self._incr_epoch()

    def train_for(self, num_epochs: int = 1) -> Dict[str, Any]:
        for _ in range(num_epochs):
            self._train_epoch()
        return self.train_metrics

    @property
    def model(self):
        return self.student_net
