import json
import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np

from fed_distill.data import index_split_heter

logger = logging.getLogger("fed_distill")


@hydra.main(config_path="config")
def main(cfg: DictConfig) -> None:
    if "seed" in cfg.split:
        logger.info("Setting seed to %i", cfg.split.seed)
        np.random.seed(cfg.split.seed)

    train_dataset = instantiate(cfg.dataset.train)
    test_dataset = instantiate(cfg.dataset.test)

    train_splits = index_split_heter(
        train_dataset, cfg.split.num_nodes, cfg.split.unif_percentage
    )
    test_splits = index_split_heter(
        test_dataset, cfg.split.num_nodes, cfg.split.unif_percentage
    )
    final_output = {}
    for t in range(cfg.split.num_nodes):
        final_output[f"teacher{t}"] = {"train": train_splits[t], "test": test_splits[t]}

    save_file = Path(cfg.split.save_file)
    if not save_file.parent.exists():
        save_file.parent.mkdir(parents=True)
    
    with open(cfg.split.save_file, "w") as buffer:
        json.dump(final_output, buffer)


if __name__ == "__main__":
    main()
