# This code is based on: https://github.com/huawei-noah/Efficient-Computing/blob/master/Data-Efficient-Model-Compression/DAFL/teacher-train.py

import logging
import sys
import os

import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from trainer import ResnetTrainer
from resnet_cifar import ResNet18

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Reproducibility
torch.manual_seed(42)


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
        train_batch_size=2048,
        test_batch_size=2048,
        save_path=os.path.join(output_dir, "final_state_resnet18.tar"),
    )

    trainer.train(epochs)


if __name__ == "__main__":
    main("/mlodata1/jellouli", ".", 200)

