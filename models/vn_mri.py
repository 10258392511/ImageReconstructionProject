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
        pass

    def normalize_ks_(self, X_ks, X_mask):
        pass

    def denormalize_img_(self, X_img, norm):
        pass

    def compute_reg_grad_(self, X_img, D_real, D_imag, knots):
        pass

    def compute_data_grad_(self, X_imag, X_ks, X_mask):
        pass
