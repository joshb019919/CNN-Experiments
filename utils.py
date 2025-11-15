#!/usr/bin/env python3
import random
import torch
import torch.nn as nn
import numpy as np

def set_seed(seed=42):
    """Guarantees nonrandom behavior across executions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def he_init(module):
    """Apply He initialization to Conv layers."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
            