import numpy as np
import torch
import torchvision.transforms as T

from fed_distill.cifar10.constants import CIFAR10_MEAN, CIFAR10_STD


def prepare_to_visualize(img: torch.Tensor) -> np.ndarray:
    """
    Given an image generated from deep inversion (cifar10),
    perform the necessary operations in order to visualize it.
    In particular:
        . The image need to be denormalized (inverse of whitened) according the cifar10 mean and std
        . It is brought back into the [0..1] range
        . Axis 0 and 2 are transposed 
        . The image is transformed into a numpy array

    Args:
        img (torch.Tensor): tensor representing an image (shape: (3, 32, 32))

    Returns:
        np.ndarray: prepared image (shape: (32, 32, 3))
    """
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
