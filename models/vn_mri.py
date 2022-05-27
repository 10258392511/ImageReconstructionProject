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
        # TODO: initialize all parameters with appropriate distr.
        self.params = self.params
        self.kernels_real = None
        self.kernels_imag = None
        self.knots = None
        self.alphas = None
        self.momentums = None
        self.k0 = None

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
