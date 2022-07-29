import os
import random

import torch
import numpy as np


def set_random_seed(seed: int = 1):
    torch.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = '1'


def set_torch_determinism():
    # Note that this feature is experimental. Be aware of using it.
    torch._set_deterministic(d=True)
