from turtle import forward
from typing import List
import torch
from torch import nn

from fed_distill.deep_inv.loss import JensonShannonDiv, ADILoss, compute_var


class MultiTeacherADILoss(ADILoss):
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        teacher_outputs: List[torch.Tensor],
        teacher_bns: List[torch.Tensor] = None,
        student_output: torch.Tensor = None,
    ) -> torch.Tensor:
        num_teachers = len(teacher_outputs)

        loss = (
            sum(
                self.criterion(teacher_output, targets)
                for teacher_output in teacher_outputs
            )
            / num_teachers
        )

        if self.l2_scale > 0.0:
            loss += self.l2_scale * torch.norm(inputs, 2)

        if self.var_scale > 0.0:
            loss += self.var_scale * compute_var(inputs)

        if self.bn_scale > 0.0:
            assert teacher_bns is not None
            loss += self.bn_scale * sum(teacher_bns) / num_teachers

        if self.comp_scale > 0.0:
            loss += self.comp_scale * sum(
                (
                    1 - self.js_div(teacher_output, student_output)
                    for teacher_output in teacher_outputs
                )
            ) / num_teachers

        return loss

