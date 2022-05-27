import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_interp_linear(min_val, max_val, X, knots, mode="3d"):
    """
    X: (B, C, D, H, W) or (B, C, H, W); knots: (N_knots, C), requires_grad = True
    """
    assert mode in ("3d", "2d")
    if mode == "3d":
        B, C, D, H, W = X.shape
    else:
        B, C, H, W = X.shape
    N_knots, _ = knots.shape
    X = torch.clip(X, min_val, max_val)  # (B, C, D, H, W) or (B, C, H, W)
    X = (X - min_val) / (max_val - min_val) * (N_knots - 1)  # scaling to be in [0, N_knots - 1]; (B, C, D, H, W) or (B, C, H, W)
    if mode == "3d":
        X = X.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
    else:
        X = X.permute(0, 2, 3, 1)  # (B, H, W, C)
    X = X.reshape(-1, C)  # (N, C)
    X_floor = torch.floor(X).long().detach()
    X_ceil = torch.ceil(X).long().detach()
    vals_floor = torch.gather(knots, dim=0, index=X_floor)  # (N, C)
    vals_ceil = torch.gather(knots, dim=0, index=X_ceil)  # (N, C)
    vals_out = vals_floor * (X_ceil - X) + vals_ceil * (X - X_floor)  # (N, C)
    if mode == "3d":
        vals_out = vals_out.reshape((B, D, H, W, C)).permute(0, 4, 1, 2, 3)  # (N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
    else:
        vals_out = vals_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)  # (N, C) -> (B, H, W, C) -> (B, C, H, W)

    return vals_out


def fixed_interp_linear_complex(min_val, max_val, X, knots, mode="3d"):
    X = X.to(torch.complex64)
    X_real, X_imag = torch.real(X), torch.imag(X)
    vals_out =  fixed_interp_linear(min_val, max_val, X_real, knots, mode) + \
                1j * fixed_interp_linear(min_val, max_val, X_imag, knots, mode)
    return vals_out


def conv_complex(X, D_real, D_imag, mode="3d"):
    """
    3d:
    X: (B, C_in, D, H, W); D_real, D_img: (C_out, C_in, kD, kH, kW)
    2d:
    X: (B, C_in, H, W); D_real, D_img: (C_out, C_in, kH, kW)
    """
    assert mode in ("3d", "2d"), "invalid mode"
    assert D_real.shape == D_imag.shape
    padding = tuple(np.array(D_real.shape[2:]) // 2)
    X = X.to(torch.complex64)
    X_real, X_imag = torch.real(X), torch.imag(X)
    if mode == "3d":
        X_real_out = F.conv3d(X_real, D_real, padding=padding) - F.conv3d(X_imag, D_imag,
                                                                          padding=padding)  # (B, C_out, D, H, W)
        X_imag_out = F.conv3d(X_imag, D_real, padding=padding) + F.conv3d(X_real, D_imag, padding=padding)
    else:
        X_real_out = F.conv2d(X_real, D_real, padding=padding) - F.conv2d(X_imag, D_imag,
                                                                          padding=padding)  # (B, C_out, D, H, W)
        X_imag_out = F.conv2d(X_imag, D_real, padding=padding) + F.conv2d(X_real, D_imag, padding=padding)
    X_out = X_real_out + 1j * X_imag_out

    # (B, C_out, D, H, W) or (B, C_out, H, W)
    return X_out


def conv_adj_complex(X, D_real, D_imag, mode="3d"):
    """
    3d:
    X: (B, C_out, D, H, W); D_real, D_img: (C_out, C_in, kD, kH, kW)
    2d:
    X: (B, C_out, H, W); D_real, D_img: (C_out, C_in, kH, kW)
    """
    assert mode in ("3d", "2d"), "invalid mode"
    assert D_real.shape == D_imag.shape
    X = X.to(torch.complex64)
    padding = tuple(np.array(D_real.shape[2:]) // 2)
    X_real, X_imag = torch.real(X), torch.imag(X)
    if mode == "3d":
        X_real_out = F.conv_transpose3d(X_real, D_real, padding=padding) - F.conv_transpose3d(X_imag, D_imag,
                                                                          padding=padding)  # (B, C_out, D, H, W)
        X_imag_out = F.conv_transpose3d(X_imag, D_real, padding=padding) + F.conv_transpose3d(X_real, D_imag,
                                                                                              padding=padding)
    else:
        X_real_out = F.conv_transpose2d(X_real, D_real, padding=padding) - F.conv_transpose2d(X_imag, D_imag,
                                                                          padding=padding)  # (B, C_out, D, H, W)
        X_imag_out = F.conv_transpose2d(X_imag, D_real, padding=padding) + F.conv_transpose2d(X_real, D_imag, padding=padding)
    X_out = X_real_out + 1j * X_imag_out

    #(B, C_in, D, H, W) or (B, C_in, H, W)
    return X_out


def i2k_complex(X):
    """
    X: (B, C, D, H, W) or (B, C, H, W)
    """
    X = X.to(torch.complex64)
    X_k_space = torch.fft.fftn(X, dim=[-1, -2])
    X_k_space_shifted = torch.fft.fftshift(X_k_space, dim=[-1, -2])

    return X_k_space_shifted


def k2i_complex(X):
    """
    X: (B, C, D, H, W) or (B, C, H, W)
    """
    X = X.to(torch.complex64)
    X_i_shifted = torch.fft.ifftshift(X, dim=[-1, -2])
    X_img = torch.fft.ifftn(X_i_shifted, dim=[-1, -2])

    return X_img


def normalize_tensor(X, axes: tuple, eps=1e-6):
    X_mean = torch.mean(X, dim=axes, keepdim=True)
    X_std = torch.std(X, dim=axes, keepdim=True)
    X_out = (X - X_mean) / (X_std + eps)

    return X_out
