import numpy as np
import torch
import torch.nn as nn


def fixed_interp_linear(min_val, max_val, X, knots):
    """
    X: (B, C, D, H, W); knots: (N_knots, C), requires_grad = True
    """
    B, C, D, H, W = X.shape
    N_knots, _ = knots.shape
    X = torch.clip(X, min_val, max_val)  # (B, C, D, H, W)
    X = (X - min_val) / (max_val - min_val) * (N_knots - 1)  # scaling to be in [0, N_knots - 1]; (B, C, D, H, W)
    X = X.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
    X = X.reshape(-1, C)  # (N, C)
    X_floor = torch.floor(X).long().detach()
    X_ceil = torch.ceil(X).long().detach()
    vals_floor = torch.gather(knots, dim=0, index=X_floor)  # (N, C)
    vals_ceil = torch.gather(knots, dim=0, index=X_ceil)  # (N, C)
    vals_out = vals_floor * (X_ceil - X) + vals_ceil * (X - X_floor)  # (N, C)
    vals_out = vals_out.reshape((B, D, H, W, C)).permute(0, 4, 1, 2, 3)  # (N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)

    return vals_out
