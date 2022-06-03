import sys

# path = r"D:\testings\Python\TestingPython"
path = "/home/zhexwu/GraduateCourses"
if path not in sys.path:
    sys.path.append(path)

import argparse
import ImageReconstructionProject.scripts.configs as configs
import ImageReconstructionProject.helpers.pytorch_utils as ptu

from datetime import datetime
from torch.utils.data import DataLoader
from ImageReconstructionProject.models.vn_mri import VNMRI
from ImageReconstructionProject.helpers.datasets import MRIDataset
from ImageReconstructionProject.helpers.utils import create_log_dir
from ImageReconstructionProject.helpers.trainer import VNMRITrainer


if __name__ == '__main__':
    """
    python scripts/run_vnmri.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_ratio", type=float, default=configs.dataset_args["sampling_ratio"])
    parser.add_argument("--noise_std", type=float, default=configs.dataset_args["noise_std"])
    parser.add_argument("--mode", default="vtv")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--grad_clip_val", default=None)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    params = vars(args)
    params.update({
        "notebook": False
    })
    dataset_args = configs.dataset_args.copy()
    dataset_args.update({
        "sampling_ratio": params["sampling_ratio"],
        "noise_std": params["noise_std"]
    })
    model_args = configs.vn_mri_params.copy()
    model_args.update({
        "mode": params["mode"]
    })
    params.update({
        "opt_params": configs.vn_mri_opt_params.copy()
    })
    params["opt_params"]["args"]["lr"] = params["lr"]
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    # dataset
    if params["mode"] == "2d":
        dim = "2d"
    else:
        dim = "3d"
    mri_ds_train = MRIDataset(dataset_args, "train", dim)
    mri_ds_test = MRIDataset(dataset_args, "test", dim)
    train_loader = DataLoader(mri_ds_train, batch_size=params["batch_size"], shuffle=True,
                              num_workers=params["num_workers"])
    test_loader = DataLoader(mri_ds_test, batch_size=params["batch_size"], num_workers=params["num_workers"])

    # model
    model = VNMRI(model_args).to(ptu.device)

    # training
    log_params = {
        "sampling_ratio": dataset_args["sampling_ratio"],
        "noise_std": dataset_args["noise_std"],
        "mode": model_args["mode"],
        "lr": params["lr"]
    }
    log_dir_name = create_log_dir(time_stamp, log_params)
    params.update({
        "log_dir": f"run/{log_dir_name}",
        "param_save_dir": f"params/{log_dir_name}"
    })
    trainer = VNMRITrainer(model, train_loader, test_loader, params)

    ### VM only ###
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open(f"/home/zhexwu/GraduateCourses/submission/vnmri_log/{log_dir_name}.txt",
                    "w")
    sys.stdout = log_file
    sys.stderr = log_file
    ### end of VM only block ###
    trainer.train()
