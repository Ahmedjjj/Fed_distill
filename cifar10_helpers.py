import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

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


def load_cifar10_test(root: str) -> torch.utils.data.Dataset:
    transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return CIFAR10(root=root, train=False, transform=transform)


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


def test_accuracy(
    model: nn.Module, data_root: str, device="cuda", batch_size=8192
) -> float:
    model.eval().to(device)
    eval_set = load_cifar10_test(data_root)
    with torch.no_grad():
        num_correct = 0
        for images, labels in DataLoader(eval_set, batch_size=batch_size):
            result = model(images.to(device))
            pred = result.argmax(1)
            num_correct += (pred == labels.to(device)).sum()
        return float(num_correct / len(eval_set))


def get_balanced_targets(batch_size: int, num_classes=10) -> torch.Tensor:
    assert batch_size % num_classes == 0
    return torch.tensor([[i] * (batch_size // 10) for i in range(num_classes)]).reshape(
        -1
    )

