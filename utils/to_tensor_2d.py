import torch
import numpy as np

def to_tensor_2d(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a numpy.ndarray or torch.Tensor")
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {x.shape}")
    return x.double().to(device)