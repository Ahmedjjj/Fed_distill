from torch import nn
import torch
from fed_distill.deep_inv.deep_inv import DeepInversion
from fed_distill.deep_inv.loss import DeepInversionLoss
from fed_distill.deep_inv.sampler import TargetSampler


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
