import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import tqdm

from fed_distill.data.label_sampler import TargetSampler
from fed_distill.deep_inv.loss import ADILoss, DILoss
from fed_distill.train.tester import get_batch_accuracy


# This class taken from: https://github.com/NVlabs/DeepInversion/blob/master/cifar10/deepinversion_cifar10.py
class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module: nn.Module) -> None:
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = (
            input[0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view([nch, -1])
            .var(1, unbiased=False)
        )

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(
            module.running_var.data.type(var.type()) - var, 2
        ) + torch.norm(module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self) -> None:
        self.hook.remove()


@dataclass
class AdaptiveDeepInversion:
    loss: ADILoss # loss to use for batch generation
    optimizer: torch.optim.Optimizer # optimizer to user
    teacher: nn.Module # teacher module
    student: Optional[nn.Module] = None # student module
    grad_updates_batch: int = 1000 # number of gradient updates
    input_jitter: bool = True # if True, each iteration will apply input jittering
    use_amp: bool = True # if True, use Automatic Mixed Precision (very good speedup)

    def __post_init__(self) -> None:
        self.inputs = self.optimizer.param_groups[0]["params"][
            0
        ]  # extract initial matrix
        self._metrics = {"instant_acc_teacher": []}
        if self.student:
            self._metrics["instant_acc_student"] = []

    def _prepare_teacher(self, teacher_net: nn.Module) -> None:
        self.bn_losses = []
        for module in teacher_net.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn_losses.append(DeepInversionFeatureHook(module))

    def _cleanup_hooks(self) -> None:
        for feature_hook in self.bn_losses:
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
        self._prepare_teacher(self.teacher)

        # prepare optimizer and targets
        best_cost = 1e6
        best_inputs = self.inputs.data
        self._reset_optimizer()

        # Both teacher and student need to be in eval mode
        restore_teacher = self.teacher.training
        if self.student:
            restore_student = self.student.training

        self.teacher.eval()
        if self.student:
            self.student.eval()
        
        if self.use_amp:
            scaler = amp.GradScaler()

        for _ in tqdm.tqdm(range(self.grad_updates_batch)):
            inputs = self.inputs
            if self.input_jitter:
                inputs = self._input_jitter(inputs, 2, 2)
            
            with amp.autocast(enabled=self.use_amp):
                self.optimizer.zero_grad()
                teacher_output = self.teacher(inputs)
                student_output = None
                if self.student:
                    student_output = self.student(inputs)
                teacher_bns = [mod.r_feature for mod in self.bn_losses]

                loss = self.loss(
                    inputs, targets, teacher_output, teacher_bns, student_output
                )

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

        self._metrics["instant_acc_teacher"].append(
            get_batch_accuracy(self.teacher, self.inputs, targets)
        )
        if self.student:
            self._metrics["instant_acc_student"].append(
                get_batch_accuracy(self.student, self.inputs, targets)
            )

        if restore_teacher:
            self.teacher.train()
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


class DeepInversion(AdaptiveDeepInversion):
    def __init__(
        self,
        loss: DILoss,
        optimizer: torch.optim.Optimizer,
        teacher: nn.Module,
        grad_updates_batch: int = 1000,
        input_jitter: bool = True,
        use_amp: bool = True,
    ):
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            teacher=teacher,
            student=None,
            grad_updates_batch=grad_updates_batch,
            input_jitter=input_jitter,
            use_amp=use_amp,
        )

