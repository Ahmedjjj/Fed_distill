import logging
import os
from abc import ABC, abstractmethod
from glob import glob

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StateSaver(ABC):
    def __init__(
        self,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = dict()

    @abstractmethod
    def save(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def checkpoint(self, **kwargs):
        raise NotImplementedError()

    def add(self, metric_name, initial_value):
        if not metric_name in self.metrics:
            self.metrics[metric_name] = initial_value
        else:
            raise ValueError("Metric already present")

    def __getitem__(self, k):
        return self.metrics[k]

    def __setitem__(self, k, v):
        self.metrics[k] = v


class NoOpStateSaver(StateSaver):
    def save(self, **kwargs) -> None:
        pass
    
    def checkpoint(self, **kwargs):
        pass


class ImageSaver(ABC):
    @abstractmethod
    def save_batch(
        self, images: torch.Tensor, targets: torch.Tensor = None, filename=None
    ) -> None:
        raise NotImplementedError()


class NoOpImageSaver(ImageSaver):
    def save_batch(
        self, images: torch.Tensor, targets: torch.Tensor = None, filename=None
    ) -> None:
        pass


def load_batches_from_dir(location: str):
    assert os.path.isdir(location)
    initial_batches = []
    for i, batch_p in enumerate(glob(os.path.join(location, "*initial*.tar"))):
        logger.info(f"Loading initial batch {i+1}")
        batch = torch.load(batch_p)
        initial_batches.append((batch["images"], batch["labels"]))

    return initial_batches


class DefaultStateSaver(StateSaver):
    def __init__(
        self,
        save_location: str,
        checkpoint_every: int = 50,
        save_prefix: str = "",
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        assert os.path.isdir(save_location)
        self.save_location = save_location
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_prefix = save_prefix
        self.checkpoint_every = checkpoint_every
        self.checkpoint_counter = 0
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler)

    def save(self, name=None, **kwargs) -> None:
        if name is None:
            name = f"{self.save_prefix}_state_{self.checkpoint_counter}.tar"
        else:
            name = f"{self.save_prefix}_{name}.tar"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "metrics": self.metrics,
            },
            os.path.join(self.save_location, name),
        )

    def checkpoint(self, **kwargs):
        self.checkpoint_counter += 1
        if self.checkpoint_counter % self.checkpoint_every == 0:
            self.save()


class DefaultImageSaver(ImageSaver):
    def __init__(self, save_location: str) -> None:
        assert os.path.isdir(save_location)
        self.save_location = save_location

    def save_batch(
        self, images: torch.Tensor, targets: torch.Tensor = None, filename=None
    ) -> None:
        torch.save(
            {"images": images, "labels": targets},
            os.path.join(self.save_location, filename),
        )

