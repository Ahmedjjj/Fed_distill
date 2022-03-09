import random
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import deepinversion_cifar10
import torch
import torch.nn as nn
import tqdm
import cifar10_helpers
from trainer import ResnetTrainer
from torch.utils.data import IterableDataset, Dataset, DataLoader


class TargetSampler(ABC):
    def __iter__():
        raise NotImplementedError()


class BalancedSampler(TargetSampler):
    def __init__(self, batch_size: int, num_classes: int) -> None:
        if not batch_size % num_classes == 0:
            raise ValueError("Batch size has to be a multiple of the number of classes")
        self.targets = cifar10_helpers.get_balanced_targets(batch_size, num_classes)

    def __iter__(self):
        while True:
            yield self.targets


class RandomSampler(TargetSampler):
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield torch.LongTensor(
                [random.randint(0, 9) for _ in range(self.batch_size)]
            )


@dataclass
class DeepInversion(IterableDataset):
    teacher_net: nn.Module
    input_shape: Tuple[int, int]
    class_sampler: TargetSampler
    batch_size: int = 256
    num_batches: int = 1000
    epochs: int = 1000
    input_jitter: bool = False
    l2_scale: float = 0.0
    var_scale: float = 5e-5
    bn_scale: float = 10
    optimizer_class: torch.optim.Optimizer = torch.optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: str = "cuda"

    def __post_init__(self):
        self.inputs = torch.randn(
            (self.batch_size, 3, *self.input_shape),
            requires_grad=True,
            device=self.device,
        )
        self.class_sampler = iter(self.class_sampler)
        self.optimizer = self.optimizer_class([self.inputs], **self.optimizer_kwargs)
        self.teacher_net.eval().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.bn_losses = []
        for module in self.teacher_net.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn_losses.append(
                    deepinversion_cifar10.DeepInversionFeatureHook(module)
                )

    def _compute_batch(self):
        best_cost = 1e6
        best_inputs = self.inputs.data

        for group in self.optimizer.param_groups:
            group.update(self.optimizer.defaults)

        for _ in tqdm.tqdm(range(self.epochs)):
            inputs = self.inputs
            if self.input_jitter:
                lim_0, lim_1 = 2, 2
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs = torch.roll(self.inputs, shifts=(off1, off2), dims=(2, 3))

            self.optimizer.zero_grad()
            outputs = self.teacher_net(inputs)
            targets = next(self.class_sampler).to(self.device)
            loss = self.criterion(outputs, targets)

            # apply total variation regularization
            diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
            diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
            diff3 = inputs[:, :, 1:, :-1] - inputs[:, :, :-1, 1:]
            diff4 = inputs[:, :, :-1, :-1] - inputs[:, :, 1:, 1:]

            loss_var = (
                torch.norm(diff1)
                + torch.norm(diff2)
                + torch.norm(diff3)
                + torch.norm(diff4)
            )
            loss = loss + self.var_scale * loss_var
            loss = loss + self.bn_scale * sum([mod.r_feature for mod in self.bn_losses])
            loss = loss + self.l2_scale * torch.norm(inputs, 2)
            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

            loss.backward()
            self.optimizer.step()
        return best_inputs, targets

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self._compute_batch()


@dataclass
class StudentTrainer(ResnetTrainer):
    resnet: nn.Module
    train_dataset: DeepInversion
    test_dataset: Dataset
    test_batch_size: int
    device: str = "cuda"
    save_path: str = "."

    def __post_init__(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=1)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.test_batch_size
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


class TensorDatasetWrapper(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.images[index]), self.labels[index]

