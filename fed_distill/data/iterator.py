import logging
from typing import Iterator, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

def mix_iterators(
    iterators: Sequence[Iterator[Tuple[torch.Tensor, torch.Tensor]]]
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    i = 0
    while True:
        logger.info("Generating batch from teacher %i", i)
        yield next(iterators[i])
        i = (i + 1) % len(iterators)

