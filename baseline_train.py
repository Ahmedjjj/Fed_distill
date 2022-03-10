from distutils.util import subst_vars
import logging
import sys
import os

import torch
from torch.utils.data import Subset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from trainer import ResnetTrainer
from resnet_cifar import ResNet18
from deep_inversion import TensorDatasetWrapper
from cifar10_helpers import load_cifar10_test
import itertools
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Reproducibility
torch.manual_seed(42)
logger = logging.getLogger()

TARGETS_PATH = "data/targets.tar"


def main(data_dir, output_dir, epochs):
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(data_dir, transform=train_transform, download=True)
    targets = torch.load(TARGETS_PATH)

    indices = []
    for label, samples in itertools.groupby(sorted(list(targets))):
        num_samples = len(list(samples))
        targets = np.argwhere(np.array(train_dataset.targets) == int(label))
        indices.extend(targets[:num_samples].reshape(-1).tolist())

    logger.info("Preparing subset")
    train_dataset = Subset(train_dataset, indices)

    logger.info(f"Subset of size {len(train_dataset)}")
    test_transform = T.Compose(
        [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]
    )

    test_dataset = CIFAR10(
        data_dir, train=False, transform=test_transform, download=True
    )

    resnet = ResNet18()

    trainer = ResnetTrainer(
        resnet=resnet,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_batch_size=len(train_dataset),
        test_batch_size=2048,
        save_path=os.path.join(output_dir, "final_state_baseline.tar"),
    )

    trainer.train(epochs)


if __name__ == "__main__":
    main("/mlodata1/jellouli", ".", 1000)
