import torch
from fed_distill.deep_inv.deep_inv import DeepInversion
from fed_distill.deep_inv.loss import DeepInversionLoss
from fed_distill.deep_inv.sampler import TargetSampler

def get_resnet_cifar_adi(
                        class_sampler: TargetSampler,
                        adam_lr: float,
                        l2_scale: float,
                        var_scale: float,
                        bn_scale:float,
                        comp_scale:float,
                        batch_size: int=256,
                        grad_updates_batch:int=1000,
                        device="cuda") -> DeepInversion:

    inputs = torch.randn((batch_size, 3, 32, 32), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([inputs], lr=adam_lr)
    loss = DeepInversionLoss(
        l2_scale=l2_scale, var_scale=var_scale, bn_scale=bn_scale, comp_scale=comp_scale
    )
    return DeepInversion(loss=loss, optimizer=optimizer, class_sampler=class_sampler, batch_size=batch_size, grad_updates_batch=grad_updates_batch, input_jitter=True)

def get_resnet_cifar_di(class_sampler: TargetSampler,
                        adam_lr: float,
                        l2_scale: float,
                        var_scale: float,
                        bn_scale:float,
                        batch_size: int=256,
                        grad_updates_batch:int=1000,
                        device="cuda") -> DeepInversion:

    return get_resnet_cifar_adi(class_sampler, adam_lr, l2_scale, var_scale, bn_scale, 0.0, batch_size,grad_updates_batch, device)
