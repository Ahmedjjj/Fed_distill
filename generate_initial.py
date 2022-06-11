import json
import logging
from pathlib import Path
from typing import Iterator, Sequence

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np

from fed_distill.data import extract_subset
from fed_distill.data import RandomSampler
from fed_distill.data.growing_dataset import GrowingDataset
logger = logging.getLogger("fed_distill")

def mix_iterators(iterators: Sequence[Iterator[torch.Tensor, torch.Tensor]]) -> Iterator[torch.Tensor, torch.Tensor]:
    i = 0
    while True:
        logger.info("Generating batch from teacher %i", i)
        yield next(iterators[i])
        i = (i + 1) % len(iterators)

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    t_cfg = cfg.teacher
    if "seed" in cfg:
        logger.info("Setting seed to %i", cfg.seed)
        torch.random.manual_seed(cfg.seed)
    
    train_dataset = instantiate(cfg.dataset.train)

    split = None
    if "split" in cfg:
        with open(cfg.split.save_file) as buffer:
            split = json.load(buffer)

    deep_invs = []
    teacher_save_folder = Path(t_cfg.save_folder)

    for i in range(t_cfg.num_teachers):
        weights = torch.load(teacher_save_folder / f"model_teacher{i}.pt")
        teacher = instantiate(t_cfg.model).to("cuda")
        teacher.load_state_dict(weights)
        
        teacher_dataset = (
            extract_subset(train_dataset, split[f"teacher{i}"]["train"])
            if split
            else train_dataset
        )
        dataset_targets = tuple(np.unique(teacher_dataset.targets).tolist())
        sampler = RandomSampler(batch_size=cfg.di.batch_size, classes=dataset_targets)
        
        inputs = torch.randn((cfg.di.batch_size, *cfg.dataset.input_size), requires_grad=True, device="cuda")
        optimizer = instantiate(cfg.deep_inversion.optimizer)([inputs])
        di = instantiate(cfg.deep_inversion.di)(optimizer=optimizer, teacher=teacher)
        
        deep_invs.append(di.iterator_from_sampler(sampler))
    
    dataset = GrowingDataset(stream=mix_iterators(deep_invs))

    save_file = Path(cfg.initial.save_path)

    if not save_file.parent.exists():
        save_file.parent.mkdir(parents=True)

    for i in range(cfg.initial.num_batches):
        dataset.grow()
    
    dataset.save(save_file)
    
    