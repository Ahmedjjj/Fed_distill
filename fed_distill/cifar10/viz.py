import numpy as np
import torch
import torchvision.transforms as T

from fed_distill.cifar10.constants import CIFAR10_MEAN, CIFAR10_STD


def prepare_to_visualize(img: torch.Tensor) -> np.ndarray:
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
