import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
from fed_distill.cifar10 import load_cifar10_test
from fed_distill.deep_inv.deep_inv import (
    DeepInversion,
    ResnetCifarAdaptiveDeepInversion,
    ResnetCifarDeepInversion,
)
from fed_distill.deep_inv.sampler import TargetSampler
from resnet_cifar import ResNet18
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class StudentTrainer:
    deep_inversion: DeepInversion
    test_dataset: Dataset
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    student_net: nn.Module = None
    test_batch_size: int = 4098
    training_epochs: int = 250
    epoch_gradient_updates: int = 195
    generation_steps: List[int] = field(default=list)
    initial_batches: List[Tuple[torch.Tensor, torch.Tensor]] = field(default=list)
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
        self.student_train_accs_instanteneous = []
        self.teacher_train_accs_instanteneous = []
        self.student_train_accs = []

        self.teacher_net = self.deep_inversion.teacher_net

        if self.student_net == None:
            self.student_net = self.deep_inversion.student_net
        self.student_net.to(self.device)

        self.best_model = self.student_net.state_dict()
        self.best_acc = 0.0

        self.all_batches = self.initial_batches.copy()

        self.epoch = 1
        self.current_batch = 1

    def _get_test_acc(self) -> float:
        self.student_net.eval()
        with torch.no_grad():
            num_correct = 0
            for input, targets in self.test_loader:
                num_correct += torch.sum(
                    self.student_net(input.to(self.device)).argmax(dim=1)
                    == targets.to(self.device)
                )
            return float(num_correct / len(self.test_dataset))

    def _get_batch_acc(
        self, inputs: torch.Tensor, targets: torch.Tensor, teacher=False
    ) -> float:
        net = self.teacher_net if teacher else self.student_net
        net.eval()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        with torch.no_grad():
            return float(torch.sum(net(inputs).argmax(dim=1) == targets) / len(targets))

    def _train_on_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        self.student_net.train()
        output = self.student_net(inputs.to(self.device))
        loss = self.criterion(output.to(self.device), targets.to(self.device))
        loss.backward()
        self.optimizer.step()

    def _save_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        torch.save(
            {"images": inputs.cpu().detach(), "labels": targets.cpu().detach()},
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
            "student_train_accs_instanteneous": self.student_train_accs_instanteneous,
            "teacher_train_accs_instanteneous": self.teacher_train_accs_instanteneous,
            "student_train_accs": self.student_train_accs,
            "best_acc": self.best_acc,
            "best_model": self.best_model,
        }
        torch.save(
            state_dict,
            os.path.join(
                self.save_folder, self.save_prefix + f"_epoch{self.epoch}.tar"
            ),
        )

    def _get_batch(self):
        logger.info(f"Generating data batch {self.current_batch + 1}")
        self.student_net.eval()
        new_inputs, new_targets = self.deep_inversion.compute_batch()
        new_inputs = new_inputs.detach().cpu()
        new_targets = new_targets.detach().cpu()
        if self.save_images:
            self._save_batch(new_inputs, new_targets)
        self.current_batch += 1
        return new_inputs, new_targets

    def train_epoch(self):
        logger.info(f"Starting training of epoch {self.epoch}")
        self.student_net.train()
        epoch_acc = 0
        for step in range(self.epoch_gradient_updates):
            logger.debug(f"Starting batch {step + 1}")
            if step in self.generation_steps:
                new_inputs, new_targets = self._get_batch()
                student_acc = self._get_batch_acc(new_inputs, new_targets)
                teacher_acc = self._get_batch_acc(new_inputs, new_targets, teacher=True)
                logger.info(f"Student accuracy on new batch: {student_acc}")
                logger.info(f"Teacher accuracy on new batch: {teacher_acc}")

                self.student_train_accs_instanteneous.append(student_acc)
                self.teacher_train_accs_instanteneous.append(teacher_acc)

                self._train_on_batch(new_inputs, new_targets)
                epoch_acc += self._get_batch_acc(new_inputs, new_targets)
                self.all_batches.append((new_inputs, new_targets))

            else:
                training_batch_index = random.randint(0, len(self.all_batches) - 1)
                inputs, targets = self.all_batches[training_batch_index]
                self._train_on_batch(inputs, targets)
                epoch_acc += self._get_batch_acc(inputs, targets)

        epoch_acc /= self.epoch_gradient_updates
        logger.info(f"Student training acc: {epoch_acc}")
        self.student_train_accs.append(epoch_acc)

    def train(self):
        for _ in range(self.epoch, self.training_epochs + 1):
            self.train_epoch()
            test_acc = self._get_test_acc()
            if test_acc > self.best_acc:
                logger.info("Accuracy improved")
                self.best_acc = test_acc
                self.best_model = self.student_net.state_dict()

            logger.info(f"Student test accuracy after epoch {self.epoch} : {test_acc}")
            self.student_test_accs.append(test_acc)
            if self.save_state and self.epoch % self.save_every == 0:
                self._save_state()
            self._incr_epoch()


class ResnetCifarStudentTrainer(StudentTrainer):
    def __init__(
        self,
        teacher_net: nn.Module,
        class_sampler: TargetSampler,
        initial_batches=[],
        num_initial_batches: int = 50,
        data_root: str = ".",
        adam_lr: float = 0.1,
        l2_scale: float = 0.0,
        var_scale: float = 1e-3,
        bn_scale: float = 10,
        comp_scale: float = 10,
        batch_size: int = 256,
        grad_updates_batch: int = 1000,
        test_batch_size: int = 4098,
        training_epochs: int = 300,
        lr_decay_milestones: List[int] = [150, 250],
        epoch_gradient_updates: int = 195,
        device: str = "cuda",
        save_prefix: str = "",
        save_images: bool = False,
        imgs_save_folder: str = ".",
        save_state: bool = False,
        save_folder: str = ".",
        save_every: int = None,
    ) -> None:

        initial_deep_inv = ResnetCifarDeepInversion(
            teacher_net=teacher_net,
            class_sampler=class_sampler,
            adam_lr=adam_lr,
            l2_scale=l2_scale,
            var_scale=var_scale,
            bn_scale=bn_scale,
            batch_size=batch_size,
            grad_updates_batch=grad_updates_batch,
            device=device,
        )
        initial_batches = initial_batches.copy()
        num_batches_gen = max(0, num_initial_batches - len(initial_batches))

        for i in range(num_batches_gen):
            logger.info(f"Generating initial batch {i}")
            images, targets = initial_deep_inv.compute_batch()
            images = images.cpu().detach()
            targets = targets.cpu().detach()
            initial_batches.append((images, targets))
            if save_images:
                torch.save(
                    {"images": images, "labels": targets},
                    os.path.join(
                        imgs_save_folder, save_prefix + f"_initial_batch{i + 1}.tar",
                    ),
                )

        student_net = ResNet18()
        optimizer = torch.optim.SGD(
            student_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_decay_milestones, gamma=0.1
        )

        if comp_scale > 0.0:
            deep_inv = ResnetCifarAdaptiveDeepInversion(
                teacher_net=teacher_net,
                student_net=student_net,
                class_sampler=class_sampler,
                adam_lr=adam_lr,
                l2_scale=l2_scale,
                var_scale=var_scale,
                bn_scale=bn_scale,
                comp_scale=comp_scale,
                batch_size=batch_size,
                grad_updates_batch=grad_updates_batch,
                device=device,
            )
        else:
            deep_inv = initial_deep_inv

        super().__init__(
            deep_inversion=deep_inv,
            test_dataset=load_cifar10_test(data_root),
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=nn.CrossEntropyLoss(),
            student_net=student_net,
            test_batch_size=test_batch_size,
            training_epochs=training_epochs,
            epoch_gradient_updates=epoch_gradient_updates,
            generation_steps=[0, epoch_gradient_updates // 2],
            initial_batches=initial_batches,
            device=device,
            save_prefix=save_prefix,
            save_images=save_images,
            imgs_save_folder=imgs_save_folder,
            save_state=save_state,
            save_folder=save_folder,
            save_every=save_every,
        )
