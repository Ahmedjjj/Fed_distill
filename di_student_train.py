import logging
import sys
import os

import torch
import torchvision.transforms as T

from trainer import ResnetTrainer
from resnet_cifar import ResNet18
from deep_inversion import TensorDatasetWrapper
from cifar10_helpers import load_cifar10_test

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Reproducibility
torch.manual_seed(42)
logger = logging.getLogger()

DATASET_PATH = "data/batch.tar"
TARGETS_PATH = "data/targets.tar"


def main(data_dir, output_dir, epochs):
    train_transform = T.Compose(
        [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),]
    )
    targets = torch.load(TARGETS_PATH).cpu()
    train_dataset = TensorDatasetWrapper(
        torch.load(DATASET_PATH).cpu(), targets, transform=train_transform
    )
    test_dataset = load_cifar10_test(data_dir)

    resnet = ResNet18()

    trainer = ResnetTrainer(
        resnet=resnet,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_batch_size=len(targets),
        test_batch_size=2048,
        num_workers=1,
        save_path=os.path.join(output_dir, "final_state_student_resnet18.tar"),
    )

    trainer.train(epochs)


if __name__ == "__main__":
    main("/mlodata1/jellouli", ".", 2000)
