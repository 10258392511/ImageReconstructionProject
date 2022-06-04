from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

dataset_args = {
    "path": "data/2dt_heart.mat",
    "split_ratio": 0.9,
    "seed": 0,
    "half_window_size": 5,  # same as hands-on 8
    "sampling_ratio": 0.2,  # hands-on 8: 0.3
    "noise_std": 0.01,
    "eps_log": 1e-6
}

# in consistency with hands-on 8
vn_mri_params = {
    "num_layers": 7,
    "num_knots": 31,
    "kernel_size": 5,
    "activation_range": (-1, 1),
    "num_filters": 15,
    "in_channels": 1,
    "init_kernel_std": 1e-3,
    "init_knots_std": 1e-3,
    "init_alpha_range": (0, 0.01),
    "init_momentum_range": (0.05, 0.2),
}

vn_mri_opt_params = {
    "class": AdamW,
    "args": {
        "lr": 1e-3
    },
    "scheduler": LambdaLR,
    "scheduler_args": {
        "lr_lambda": lambda epoch: max(0.95 ** epoch, 1e-4 / vn_mri_opt_params["args"]["lr"])
    }
}
