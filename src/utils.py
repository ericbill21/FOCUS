from typing import Optional

import torch
from torch import Tensor
from contextlib import contextmanager

def append_dims(x: Tensor, target_ndim: int) -> Tensor:
    """
    Unsqueeze tensor `x` along its last dimensions until it has `target_ndim` dimensions.
    """
    nd = x.ndim
    if nd > target_ndim:
        raise ValueError(f"x.ndim ({nd}) > target_ndim ({target_ndim})")
    return x.reshape(*x.shape, *(1,) * (target_ndim - nd))


def sample_time_indices(
    K,
    start_inter=5,
    end_inter=18,
    n_rand=10,
    device="cpu",
    generator: Optional[torch.Generator] = None,
) -> Tensor:

    fixed_idx = torch.arange(start_inter, end_inter, device=device)

    leftout = torch.cat(
        [
            torch.arange(0, start_inter, device=device),
            torch.arange(end_inter, K, device=device),
        ]
    )

    rand_idx = torch.randperm(leftout.shape[0], device=device)[:n_rand]
    return torch.cat([fixed_idx, leftout[rand_idx]])