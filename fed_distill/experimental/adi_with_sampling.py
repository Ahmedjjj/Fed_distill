import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch import nn
from torch.cuda import amp

from fed_distill.data.label_sampler import TargetSampler
from fed_distill.deep_inv.deep_inv import DeepInversionFeatureHook
from fed_distill.deep_inv.loss import ADILoss
from fed_distill.train.tester import get_batch_accuracy

@dataclass
class AdaptiveDeepInversionWithSampling:
    loss: ADILoss
    optimizer: torch.optim.Optimizer
    teachers: Iterable[nn.Module]
    student: Optional[nn.Module] = None
    grad_updates_batch: int = 1000
    input_jitter: bool = True
    use_amp: bool = True
    num_teachers: int = 3

    def __post_init__(self) -> None:
        self.teachers = tuple(self.teachers)
        self.inputs = self.optimizer.param_groups[0]["params"][
            0
        ]  # extract initial matrix
        self._metrics = {"instant_acc_teacher": []}
        if self.student:
            self._metrics["instant_acc_student"] = []
        self.bn_losses = []

    def _prepare_teacher(self, teacher_net: nn.Module) -> None:
        losses = []
        for module in teacher_net.modules():
            if isinstance(module, nn.BatchNorm2d):
                losses.append(DeepInversionFeatureHook(module))
        self.bn_losses.append(losses)

    def _cleanup_hooks(self) -> None:
        for losses in self.bn_losses:
            for feature_hook in losses:
                feature_hook.close()
        self.bn_losses = []

    def _reset_optimizer(self) -> None:
        for group in self.optimizer.param_groups:
            group.update(self.optimizer.defaults)

    @staticmethod
    def _input_jitter(inputs: torch.Tensor, lim_x: int, lim_y: int) -> torch.Tensor:
        off1 = random.randint(-lim_x, lim_x)
        off2 = random.randint(-lim_y, lim_y)
        return torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

    def __call__(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Initialize input randomly
        self.inputs.data = torch.randn(
            self.inputs.shape,
            requires_grad=True,
            device=self.inputs.device,
            dtype=self.inputs.dtype,
        )
        targets = targets.to(self.inputs.device)

        # Register teacher feature hooks
        for teacher in self.teachers:
            self._prepare_teacher(teacher)

        # prepare optimizer and targets
        best_cost = 1e6
        best_inputs = self.inputs.data
        self._reset_optimizer()

        # Both teachers and student need to be in eval mode
        restore_teachers = []
        for teacher in self.teachers:
            restore_teachers.append(teacher.training)
            teacher.eval()
        if self.student:
            restore_student = self.student.training
            self.student.eval()
        
        if self.use_amp:
            scaler = amp.GradScaler()

        for _ in tqdm.tqdm(range(self.grad_updates_batch)):
            inputs = self.inputs
            if self.input_jitter:
                inputs = self._input_jitter(inputs, 2, 2)
            
            teacher_indxs = np.random.choice(range(len(self.teachers)), size=self.num_teachers, replace=False)
            with amp.autocast(enabled=self.use_amp):
                loss = 0
                self.optimizer.zero_grad()
                student_output = None
                if self.student:
                    student_output = self.student(inputs)
                for i in teacher_indxs:
                    teacher = self.teachers[i]
                    teacher_output = teacher(inputs)
                    teacher_bns = [mod.r_feature for mod in self.bn_losses[i]]
                    loss += self.loss(
                        inputs, targets, teacher_output, teacher_bns, student_output
                    )
                loss /= self.num_teachers

                if best_cost > loss.item():
                    best_cost = loss.item()
                    best_inputs = inputs.data
            
            if self.use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        self._cleanup_hooks()

        self._metrics["instant_acc_teacher"].extend(
            get_batch_accuracy(teacher, self.inputs, targets) for teacher in self.teachers
        )
        if self.student:
            self._metrics["instant_acc_student"].append(
                get_batch_accuracy(self.student, self.inputs, targets)
            )

        for r, t in zip(restore_teachers, self.teachers):
            if r:
                t.train()
        if self.student and restore_student:
            self.student.train()

        return best_inputs, targets

    @property
    def metrics(self):
        return deepcopy(self._metrics)

    def iterator_from_sampler(
        self, sampler: TargetSampler
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for targets in sampler:
            yield self.__call__(targets)


    
    