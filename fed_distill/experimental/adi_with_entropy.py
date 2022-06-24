from typing import Optional, Tuple
from fed_distill.deep_inv import AdaptiveDeepInversion
from fed_distill.experimental.loss_with_entropy import ADIEntropyLoss
from dataclasses import dataclass
from torch import nn 
from fed_distill.train.tester import get_batch_accuracy
from torch.cuda import amp
import tqdm
import torch

@dataclass
class AdaptiveDeepInversionWithEntropy(AdaptiveDeepInversion):
    loss: ADIEntropyLoss
    optimizer: torch.optim.Optimizer
    teacher: nn.Module
    other_teacher: Optional[nn.Module] = None
    student: Optional[nn.Module] = None
    grad_updates_batch: int = 1000
    input_jitter: bool = True
    use_amp: bool = True

    
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
        restore_other_teacher = self.other_teacher.training
        if self.student:
            restore_student = self.student.training

        self.teacher.eval()
        self.other_teacher.eval
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
                other_teacher_output = self.other_teacher(inputs)
                student_output = None
                if self.student:
                    student_output = self.student(inputs)
                teacher_bns = [mod.r_feature for mod in self.bn_losses]

                loss = self.loss(
                    inputs, targets, teacher_output, other_teacher_output, teacher_bns, student_output
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
        if restore_other_teacher:
            self.other_teacher.train()

        return best_inputs, targets
