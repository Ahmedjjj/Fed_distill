import random
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import tqdm
from fed_distill.deep_inv.loss import DeepInversionLoss
from fed_distill.deep_inv.sampler import TargetSampler


# This class taken from: https://github.com/NVlabs/DeepInversion/blob/master/cifar10/deepinversion_cifar10.py
class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
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

    def close(self):
        self.hook.remove()


def input_jitter(inputs: torch.Tensor, lim_x, lim_y) -> torch.Tensor:
    off1 = random.randint(-lim_x, lim_x)
    off2 = random.randint(-lim_y, lim_y)
    return torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))


@dataclass
class DeepInversion:
    teacher_net: nn.Module
    input_shape: Tuple[int, int]
    class_sampler: TargetSampler
    loss: DeepInversionLoss
    optimizer: torch.optim.Optimizer
    student_net: nn.Module = None
    batch_size: int = 256
    grad_updates_batch: int = 1000
    input_jitter: bool = True
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.inputs = self.optimizer.param_groups[0]["params"][
            0
        ]  # extract initial matrix
        assert self.inputs.shape == (self.batch_size, 3, *self.input_shape)
        self.class_sampler = iter(self.class_sampler)
        self.teacher_net.eval().to(self.device)

        if self.student_net:
            self.student_net.to(self.device)

        self.bn_losses = []
        for module in self.teacher_net.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn_losses.append(DeepInversionFeatureHook(module))

    def _reset_optimizer(self) -> None:
        for group in self.optimizer.param_groups:
            group.update(self.optimizer.defaults)

    def compute_batch(
        self, get_losses: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, List[float]],
    ]:
        self.inputs.data = torch.randn(
            (self.batch_size, 3, *self.input_shape),
            requires_grad=True,
            device=self.device,
        )

        best_cost = 1e6
        best_inputs = self.inputs.data
        self._reset_optimizer()
        targets = next(self.class_sampler).to(self.device)

        if self.student_net:
            self.student_net.eval()

        losses = []

        for _ in tqdm.tqdm(range(self.grad_updates_batch)):
            inputs = self.inputs
            if self.input_jitter:
                inputs = input_jitter(inputs, 2, 2)

            self.optimizer.zero_grad()
            teacher_output = self.teacher_net(inputs)
            student_output = None
            if self.student_net:
                student_output = self.student_net(inputs)
            teacher_bns = [mod.r_feature for mod in self.bn_losses]

            loss = self.loss(
                inputs, targets, teacher_output, teacher_bns, student_output
            )
            losses.append(float(loss))

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

            loss.backward()
            self.optimizer.step()
        if get_losses:
            return best_inputs, targets, losses

        return best_inputs, targets


class ResnetCifarDeepInversion(DeepInversion):
    def __init__(
        self,
        teacher_net: nn.Module,
        class_sampler: TargetSampler,
        adam_lr: float,
        l2_scale: float,
        var_scale: float,
        bn_scale: float,
        batch_size: int = 256,
        grad_updates_batch: int = 1000,
        device="cuda",
    ):
        inputs = torch.randn((batch_size, 3, 32, 32), requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([inputs], lr=adam_lr)
        loss = DeepInversionLoss(
            l2_scale=l2_scale, var_scale=var_scale, bn_scale=bn_scale, comp_scale=0.0
        )
        super().__init__(
            teacher_net=teacher_net,
            input_shape=(32, 32),
            class_sampler=class_sampler,
            loss=loss,
            optimizer=optimizer,
            student_net=None,
            batch_size=batch_size,
            grad_updates_batch=grad_updates_batch,
            input_jitter=True,
            device=device,
        )


class ResnetCifarAdaptiveDeepInversion(DeepInversion):
    def __init__(
        self,
        teacher_net: nn.Module,
        student_net: nn.Module,
        class_sampler: TargetSampler,
        adam_lr: float,
        l2_scale: float,
        var_scale: float,
        bn_scale: float,
        comp_scale: float,
        batch_size: int = 256,
        grad_updates_batch: int = 1000,
        device="cuda",
    ):
        inputs = torch.randn((batch_size, 3, 32, 32), requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([inputs], lr=adam_lr)
        loss = DeepInversionLoss(
            l2_scale=l2_scale,
            var_scale=var_scale,
            bn_scale=bn_scale,
            comp_scale=comp_scale,
        )
        super().__init__(
            teacher_net=teacher_net,
            input_shape=(32, 32),
            class_sampler=class_sampler,
            loss=loss,
            optimizer=optimizer,
            student_net=student_net,
            batch_size=batch_size,
            grad_updates_batch=grad_updates_batch,
            input_jitter=True,
            device=device,
        )
