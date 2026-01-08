import os
import random

import numpy as np
import torch


def fix_seed(seed=42):
    # cuBLAS deterministic settings (must be set before CUDA operations)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Use deterministic algorithms
    torch.use_deterministic_algorithms(True)
