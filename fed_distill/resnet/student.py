import logging
from typing import Callable, List

import torch
import torchvision.transforms as T
from fed_distill.cifar10.cifar10_helpers import load_cifar10_test
from fed_distill.common.state import (
    ImageSaver,
    NoOpImageSaver,
    NoOpStateSaver,
    StateSaver,
)
from fed_distill.common.tester import AccuracyTester
from fed_distill.deep_inv.sampler import TargetSampler
from fed_distill.deep_inv.student import StudentTrainer
from fed_distill.resnet.deep_inv import (
    ResnetCifarAdaptiveDeepInversion,
    ResnetCifarDeepInversion,
)
from torch import nn

from resnet_cifar import ResNet18

logger = logging.getLogger(__name__)


class ResnetCifarStudentTrainer(StudentTrainer):
    """
    Train a Resnet18 on a synthetic cifar10 dataset generated from a teacher model, using accuracy on the test set
    as a selection criterion.
    """

    def __init__(
        self,
        teacher_net: nn.Module,
        cifar10_test_root: str,
        initial_batches=[],
        class_sampler: TargetSampler = None,
        train_transform: Callable = None,
        num_initial_batches: int = 50,
        adam_lr: float = 0.1,
        l2_scale: float = 0.0,
        var_scale: float = 1e-3,
        bn_scale: float = 10,
        comp_scale: float = 10,
        batch_size: int = 256,
        grad_updates_batch: int = 1000,
        lr_decay_milestones: List[int] = [150, 250],
        epoch_gradient_updates: int = 195,
        training_epochs: int = 300,
        state_saver: StateSaver = NoOpStateSaver(),
        image_saver: ImageSaver = NoOpImageSaver(),
        device: str = "cuda",
    ) -> None:

        if train_transform is None:
            train_transform = T.Compose(
                [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
            )

        # Generate initial batches
        num_batches_gen = max(0, num_initial_batches - len(initial_batches))
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
        for i in range(num_batches_gen):
            logger.info(f"Generating initial batch {i}")
            images, targets = initial_deep_inv.compute_batch()
            images = images.cpu().detach()
            targets = targets.cpu().detach()
            initial_batches.append((images, targets))
            image_saver.save_batch(images, targets, f"initial_batch{i}")

        self.training_epochs = training_epochs

        student_net = ResNet18()
        optimizer = torch.optim.SGD(
            student_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_decay_milestones, gamma=0.1
        )
        state_saver.model = student_net
        state_saver.optimizer = optimizer
        state_saver.scheduler = scheduler

        self.tester = AccuracyTester(load_cifar10_test(cifar10_test_root))

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
            optimizer=optimizer,
            scheduler=scheduler,
            train_transform=train_transform,
            criterion=nn.CrossEntropyLoss(),
            student_net=student_net,
            epoch_gradient_updates=epoch_gradient_updates,
            generation_steps=[0, epoch_gradient_updates // 2],
            initial_batches=initial_batches,
            state_saver=state_saver,
            image_saver=image_saver,
            device=device,
        )

    def train(self):
        for _ in range(self.epoch, self.training_epochs + 1):
            logger.info(f"Starting epoch {self.epoch}")
            self._train_epoch()
            accuracy = self.tester(self.student_net)
            logger.info(f"Accuracy at epoch {self.epoch}: {accuracy}")
            if accuracy > self.state_saver["best_acc"]:
                logger.info("Accuracy improved")
                self.state_saver["best_acc"] = accuracy
                self.state_saver["best_model"] = self.student_net.state_dict()

            self.state_saver.checkpoint()

    @property
    def model(self) -> nn.Module:
        model = ResNet18()
        model.load_state_dict(self.state_saver['best_model'])
        return model