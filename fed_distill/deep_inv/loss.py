from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_var(inputs: torch.Tensor) -> torch.Tensor:
    diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
    diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
    diff3 = inputs[:, :, 1:, :-1] - inputs[:, :, :-1, 1:]
    diff4 = inputs[:, :, :-1, :-1] - inputs[:, :, 1:, 1:]
    return torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)


class JensonShannonDiv(nn.Module):
    def __init__(self, softmax_temp: float = 3) -> None:
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.softmax_temp = softmax_temp

    def forward(
        self, output_first: torch.Tensor, output_second: torch.Tensor
    ) -> torch.Tensor:
        first_probs = F.softmax(output_first / self.softmax_temp, dim=1)
        second_probs = F.softmax(output_second / self.softmax_temp, dim=1)
        mean_probs = 0.5 * (first_probs + second_probs)

        first_probs = torch.clamp(first_probs, 0.01, 0.99)
        second_probs = torch.clamp(second_probs, 0.01, 0.99)
        second_probs = torch.clamp(second_probs, 0.01, 0.99)

        loss = 0.5 * (
            self.kl_div(torch.log(first_probs), mean_probs)
            + self.kl_div(torch.log(second_probs), mean_probs)
        )

        return torch.clamp(loss, 0.0, 1.0)


class DeepInversionLoss(nn.Module):
    def __init__(
        self,
        l2_scale: float = 0.0,
        var_scale: float = 5e-5,
        bn_scale: float = 10,
        comp_scale: float = 0,
        softmax_temp: float = 3,
    ):
        super().__init__()
        self.l2_scale = l2_scale
        self.var_scale = var_scale
        self.bn_scale = bn_scale
        self.comp_scale = comp_scale
        self.criterion = nn.CrossEntropyLoss()
        self.js_div = JensonShannonDiv(softmax_temp)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        teacher_output: torch.Tensor,
        teacher_bns: List[torch.Tensor] = None,
        student_output: torch.Tensor = None,
    ) -> torch.Tensor:
        loss = self.criterion(targets, teacher_output)
        if self.l2_scale > 0.0:
            loss += self.l2_scale * torch.norm(inputs, 2)

        if self.var_scale > 0.0:
            loss += self.var_scale * compute_var(inputs)

        if self.bn_scale > 0.0:
            assert teacher_bns
            loss += self.bn_scale * sum(teacher_bns)

        if self.comp_scale > 0.0:
            assert student_output
            loss += self.comp_scale * (1 - self.js_div(teacher_output, student_output))

        return loss
