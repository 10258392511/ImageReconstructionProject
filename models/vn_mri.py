import torch
import torch.nn as nn

from ImageReconstructionProject.helpers.recon_utils import *


class VNMRI(nn.Module):
    def __init__(self, params):
        """
        params: configs.vn_mri_params
        bash:
            mode: "vtv", "3d" or "2d"
        """
        super(VNMRI, self).__init__()
        self.params = params
        assert self.params["mode"] in ("vtv", "3d", "2d")
        if self.params["mode"] == "3d":
            kernel_size = [self.params["kernel_size"] for _ in range(3)]
        else:
            kernel_size = [self.params["kernel_size"] for _ in range(2)]
        self.kernels_real = nn.Parameter(torch.empty((self.params["num_layers"], self.params["num_filters"],
                                                      self.params["in_channels"], *kernel_size)))
        self.kernels_imag = nn.Parameter(torch.empty((self.params["num_layers"], self.params["num_filters"],
                                                      self.params["in_channels"], *kernel_size)))
        nn.init.trunc_normal_(self.kernels_real, std=self.params["init_kernel_std"])
        nn.init.trunc_normal_(self.kernels_imag, std=self.params["init_kernel_std"])
        self.knots = nn.Parameter(torch.empty(self.params["num_layers"], self.params["num_knots"], self.params["num_filters"]))
        nn.init.trunc_normal_(self.knots, std=self.params["init_knots_std"])
        self.alphas = nn.Parameter(torch.empty((self.params["num_layers"],)))
        nn.init.uniform_(self.alphas, *self.params["init_alpha_range"])
        self.momentums = nn.Parameter(torch.empty((self.params["num_layers"],)))
        nn.init.uniform_(self.momentums, *self.params["init_momentum_range"])
        self.k0 = nn.Parameter(torch.tensor(1.))

    def forward(self, X_ks, X_mask):
        X_ks, norm = self.normalize_ks_(X_ks, X_mask)
        X_img = k2i_complex(X_ks) * self.k0  # init X_img
        m = torch.zeros_like(X_img)  # running avg of gradients
        for layer_idx in range(self.params["num_layers"]):
            D_real = self.kernels_real[layer_idx, ...]
            D_imag = self.kernels_imag[layer_idx, ...]
            knots = self.knots[layer_idx, ...]
            alpha = self.alphas[layer_idx]
            momentum = self.momentums[layer_idx, ...]
            reg_grad = self.compute_reg_grad_(X_img, D_real, D_imag, knots)
            data_grad = self.compute_data_grad_(X_img, X_ks, X_mask)
            grad = alpha * data_grad + reg_grad
            m = momentum * m + grad
            X_img = X_img - m

        X_img = self.denormalize_img_(X_img, norm)

        return X_img

    def normalize_ks_(self, X_ks, X_mask):
        """
        X_ks: (B, C, T, H, W) or (B, C, H, W)
        X_mask: (B, C, T, 1, W) or (B, C, 1, W)
        """
        sum_dims = tuple(range(2, len(X_ks.shape)))
        norm_ks = (torch.abs(X_ks) ** 2).sum(dim=sum_dims, keepdims=True)
        sum_mask = X_mask.sum(dim=sum_dims, keepdims=True)
        norm = torch.sqrt(norm_ks / sum_mask)

        return X_ks / norm, norm

    def denormalize_img_(self, X_img, norm):
        """
        X_img: (B, C, T, H, W) or (B, C, H, W)
        norm: (B, C, 1, 1, 1) or (B, C, 1, 1)
        """
        return X_img * norm

    def compute_reg_grad_(self, X_img, D_real, D_imag, knots):
        if self.params["mode"] == "vtv":
            # X_img: (B, C_in, T, H, W); D_real, D_imag: (C_out, C_in, K, K); knots: (C_out, N_knots)
            B, C_in, T, H, W = X_img.shape
            C_out = D_real.shape[0]
            DX = torch.zeros((B, C_out, T, H, W), dtype=X_img.dtype, device=X_img.device)  # (B, C_out, T, H, W)
            for t in range(T):
                DX[:, :, t, :, :] = conv_complex(X_img[:, :, t, :, :], D_real, D_imag, mode="2d")
            activation_in = (torch.abs(DX) ** 2).sum(dim=2)  # (B, C_out, H, W)
            activation_in = torch.sqrt(activation_in) / T  # (B, C_out, 1, H, W)
            # (B, C_out, 1, H, W)
            activation_out = fixed_interp_linear_complex(*self.params["activation_range"], activation_in, knots, "2d")
            VTV_out = DX * activation_out.unsqueeze(2)  # (B, C_out, T, H, W)

            X_img_out = torch.zeros_like(X_img)  # (B, C_in, T, H, W)
            for t in range(T):
                # (B, C_in, H, W)
                X_img_out[:, :, t, :, :] = conv_adj_complex(VTV_out[:, :, t, :, :], D_real, D_imag, "2d")

            # (B, C_in, T, H, W)
            return X_img_out

        # (B, C_out, T, H, W) or (B, C_out, H, W)
        activation_in = conv_complex(X_img, D_real, D_imag, mode=self.params["mode"])
        # (B, C_out, T, H, W) or (B, C_out, H, W)
        activation_out = fixed_interp_linear_complex(*self.params["activation_range"], activation_in, knots,
                                                     mode=self.params["mode"])
        # (B, C_in, T, H, W) or (B, C_in, H, W)
        X_img_out = conv_adj_complex(activation_out, D_real, D_imag, mode=self.params["mode"])

        return X_img_out

    def compute_data_grad_(self, X_img, X_ks, X_mask):
        X_img_ks = i2k_complex(X_img)  # (B, C, T, H, W) or (B, C, H, W)
        X_img_ks = X_img_ks * X_mask
        X_img_out = k2i_complex(X_img_ks - X_ks)

        # (B, C, T, H, W) or (B, C, H, W)
        return X_img_out
