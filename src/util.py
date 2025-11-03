import random

import numpy as np
import torch
from contextlib import nullcontext


# from https://github.com/3145tttt/GAS/blob/main/src/gas/utils/random.py
def set_global_seed(seed: int) -> None:
    """
    Set global seed for reproducibility.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # We need speedup
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_torch_context(device: str):
    is_cuda = (device == 'cuda')
    is_bf16_supported = (is_cuda and torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    if is_cuda and is_bf16_supported:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return nullcontext()