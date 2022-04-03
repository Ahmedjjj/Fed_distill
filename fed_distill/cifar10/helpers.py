import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def get_class_name(label: int) -> str:
    class_to_name = {
        0: "aiplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    return class_to_name[label]


def load_cifar10_test(root: str) -> torch.utils.data.Dataset:
    transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return CIFAR10(root=root, train=False, transform=transform, download=True)


def load_cifar10_train(root: str) -> torch.utils.data.Dataset:
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(root, transform=train_transform, download=True)
    return train_dataset


def prepare_to_visualize(img: torch.Tensor) -> np.array:
    mean, std = np.array(CIFAR10_MEAN), np.array(CIFAR10_STD)
    denormalize = T.Normalize(tuple(-mean / std), tuple(1.0 / std))
    transform = T.Compose(
        [
            denormalize,
            lambda x: (x - x.min()) / (x.max() - x.min()),
            lambda x: x.cpu().numpy().transpose(1, 2, 0),
        ]
    )
    return transform(img)

