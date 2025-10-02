from typing import Optional

import os
import logging
from glob import glob

def create_logger(logging_dir: Optional[str]=None, verbose: int=1):
    verbose_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    handlers = [logging.StreamHandler()]
    if logging_dir is not None:
        handlers.append(logging.FileHandler(os.path.join(logging_dir, "log.txt")))

    logging.basicConfig(
        level=verbose_map.get(verbose, logging.INFO),
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)
    return logger


def setup_experiment(results_dir: os.PathLike):
    """Create an experiment directory for the current run."""

    # Make results directory
    os.makedirs(results_dir, exist_ok=True)
    experiment_index = len(glob(os.path.join(results_dir, "*")))
    experiment_dir = os.path.join(results_dir, f"{experiment_index:03d}")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # Make experiment directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir