from fed_distill.common import Trainer
from fed_distill.deep_inv.student import StudentTrainer
from fed_distill.resnet.trainer import DatasetTrainer


class AlternatingTrainer(Trainer):
    def __init__(
        self, dataset_trainer: DatasetTrainer, student_trainer: StudentTrainer
    ):
        self.dataset_trainer = dataset_trainer
        self.student_trainer = student_trainer
        self.model_ = None

    def _train_epoch(self, epoch: int) -> None:
        if epoch % 2 == 0:
            self.dataset_trainer.train_epoch()
            self.model_ = self.dataset_trainer.model
        else:
            self.student_trainer.train_epoch()
            self.model_ = self.student_trainer.model

    def train_for(self, num_epochs: int = 1) -> None:
        for epoch in range(num_epochs):
            self._train_epoch(epoch)

    @property
    def model(self):
        return self.model_