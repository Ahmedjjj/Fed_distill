from typing import Iterable, List

import torch
import torch.nn.functional as F
from fed_distill.deep_inv import ADILoss


class ADIEntropyLoss(ADILoss):
    def __init__(
        self,
        classes: Iterable[int],
        classes_other: Iterable[int],
        l2_scale: float = 0,
        entropy_scale: float = 1,
        var_scale: float = 0.00005,
        bn_scale: float = 10,
        comp_scale: float = 0,
        softmax_temp: float = 3,
    ) -> None:
        super().__init__(
            l2_scale, var_scale, bn_scale, comp_scale, softmax_temp, classes
        )
        self.classes_other = tuple(sorted(set(classes_other)))
        self.entropy_scale = entropy_scale

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        teacher_output: torch.Tensor,
        other_teacher_output: torch.Tensor,
        teacher_bns: List[torch.Tensor] = None,
        student_output: torch.Tensor = None,
    ) -> torch.Tensor:
        base_loss = super().forward(
            inputs, targets, teacher_output, teacher_bns, student_output
        )
        
        other_teacher_output = other_teacher_output[:, self.classes]
        entropy = F.softmax(other_teacher_output, dim=1) * F.log_softmax(other_teacher_output, dim=1)
        entropy = entropy.sum(dim=1).mean()
        base_loss -= self.entropy_scale * entropy

        return base_loss
