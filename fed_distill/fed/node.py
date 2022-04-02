from fed_distill.deep_inv.student import StudentTrainer
from fed_distill.fed.alternating_trainer import AlternatingTrainer

# from fed_distill.fed.topology import Topology
from fed_distill.resnet.trainer import DatasetTrainer


class Node:
    def __init__(
        self, trainer: DatasetTrainer, student_trainer: StudentTrainer = None,
    ) -> None:
        self.trainer = trainer
        self.student_trainer = student_trainer
        self.topology = None

    def register_topology(self, t) -> None:
        self.topology = t

    @property
    def local_model(self):
        return self.trainer.model

    def train(self, num_epochs: int) -> None:
        self.trainer.train_for(num_epochs)

    def train_with_other(self, node_id:int, num_epochs: int) -> None:
        if self.topology is None:
            raise ValueError(
                f"No topology registered, Please call register_topology first!"
            )

        teacher_net = self.topology[node_id].local_model
        self.student_trainer.set_teacher(teacher_net)
        alternating_trainer = AlternatingTrainer(
            self.trainer, self.student_trainer
        )
        alternating_trainer.train_for(num_epochs)
