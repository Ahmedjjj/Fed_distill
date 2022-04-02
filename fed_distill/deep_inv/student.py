import logging
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from fed_distill.common.state import (
    ImageSaver,
    NoOpImageSaver,
    NoOpStateSaver,
    StateSaver,
)
from fed_distill.common.trainer import Trainer
from fed_distill.deep_inv.deep_inv import DeepInversion

logger = logging.getLogger(__name__)


def get_batch_accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor):
    model.eval()
    with torch.no_grad():
        return float(torch.sum(model(images).argmax(dim=1) == labels) / len(labels))


@dataclass
class StudentTrainer(Trainer):
    deep_inversion: DeepInversion
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    student_net: nn.Module
    train_transform: Callable
    epoch_gradient_updates: int = 195
    generation_steps: List[int] = field(default_factory=list)
    initial_batches: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    num_initial_batches : int = 50,
    state_saver: StateSaver = field(default=NoOpStateSaver)
    image_saver: ImageSaver = field(default=NoOpImageSaver)
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.state_saver.add("student_train_accs", [])
        self.state_saver.add("student_train_accs_instant", [])
        self.state_saver.add("teacher_train_accs_instant", [])
        self.teacher_net = self.deep_inversion.teacher_net.to(self.device)
        self.student_net.to(self.device)
        self.all_batches = self.initial_batches
        self.epoch = 1
        self.current_batch = len(self.initial_batches) + 1
        num_batches_gen = max(0, self.num_initial_batches - len(self.initial_batches))
        if num_batches_gen > 0:
            logger.info("Generating %i initial batches using deep inversion" % num_batches_gen)
            for _ in range(num_batches_gen):
                self.all_batches.append(self._get_batch(adaptive=False))

    def _get_batch(self, adaptive=True):
        logger.info(f"Generating data batch {self.current_batch}")

        self.student_net.eval()
        new_inputs, new_targets = self.deep_inversion.compute_batch(adaptive=adaptive)
        new_inputs = new_inputs.detach().cpu()
        new_targets = new_targets.detach().cpu()

        self.image_saver.save_batch(
            new_inputs, new_targets, f"batch_{self.current_batch}"
        )
        self.current_batch += 1

        return new_inputs.to(self.device), new_targets.to(self.device)

    def _train_on_batch(self, batch: torch.Tensor, targets: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        self.student_net.train()
        output = self.student_net(self.train_transform(batch).to(self.device))
        loss = self.criterion(output.to(self.device), targets.to(self.device))
        loss.backward()
        self.optimizer.step()

    def _incr_epoch(self):
        self.epoch += 1
        self.scheduler.step()

    def _train_epoch(self):
        logger.info(f"Starting training of epoch {self.epoch}")
        self.student_net.train()
        epoch_acc = 0
        for step in range(self.epoch_gradient_updates):
            logger.debug(f"Starting batch {step + 1}")

            if step in self.generation_steps:
                new_inputs, new_targets = self._get_batch()
                student_acc = get_batch_accuracy(
                    self.student_net, new_inputs, new_targets
                )
                teacher_acc = get_batch_accuracy(
                    self.teacher_net, new_inputs, new_targets
                )

                logger.info(f"Student accuracy on new batch: {student_acc}")
                logger.info(f"Teacher accuracy on new batch: {teacher_acc}")

                self.state_saver["student_train_accs_instant"].append(student_acc)
                self.state_saver["teacher_train_accs_instant"].append(teacher_acc)

                self._train_on_batch(new_inputs, new_targets)
                epoch_acc += get_batch_accuracy(self.student_net, new_inputs, new_targets)
                self.all_batches.append((new_inputs, new_targets))

            else:
                training_batch_index = random.randint(0, len(self.all_batches) - 1)
                inputs, targets = self.all_batches[training_batch_index]
                self._train_on_batch(inputs, targets)
                epoch_acc += get_batch_accuracy(self.student_net, inputs, targets)

        epoch_acc /= self.epoch_gradient_updates
        logger.info(f"Student training acc: {epoch_acc}")
        self.state_saver["student_train_accs"].append(epoch_acc)
        self._incr_epoch()

    def train_for(self, num_epochs: int = 1) -> None:
        for _ in range(num_epochs):
            self._train_epoch()
    
    @property
    def model(self):
        return self.student_net

    def set_teacher(self, teacher:nn.Module) -> None:
        if self.epoch != 1:
            logger.warning(f"set_teacher called after training was done!") 
        self.deep_inversion.set_teacher(teacher)