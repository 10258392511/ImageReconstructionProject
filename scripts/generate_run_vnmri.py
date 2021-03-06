import sys

path = "/home/zhexwu/GraduateCourses/"
if path not in sys.path:
    sys.path.append(path)

import subprocess
import argparse
import ImageReconstructionProject.scripts.configs as configs

def make_bash_script(hyper_param_dict: dict):
    """
    This script and submitting to sbatch should run in submission/
    Layout:
        + SemesterProject2
        + submission
    cd cmd: start from where the .sh file locates
    """
    bash_script = f"""#!/bin/bash
#SBATCH --output=vnmri_log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
eval "$(conda shell.bash hook)"
conda activate deep_learning
cd ../ImageReconstructionProject

python ./scripts/run_vnmri.py --num_epochs 100 --num_workers 5 --batch_size 2 --mode {hyper_param_dict["mode"]} --sampling_ratio {hyper_param_dict["sampling_ratio"]} --noise_std {hyper_param_dict["noise_std"]} --lr {hyper_param_dict["lr"]}"""

    return bash_script


def create_filename(hyper_param_dict: dict):
    filename = ""
    for key, val in hyper_param_dict.items():
        filename += f"{key}_{val}_"

    filename = filename[:-1].replace(".", "_") + ".sh"

    return filename


if __name__ == '__main__':
    """
    python ./generate_run_vnmri.py --set_num 1
    """
    hyper_params = dict()
    # set 1
    hyper_params[1] = [
        {"mode": "vtv", "sampling_ratio": configs.dataset_args["sampling_ratio"],
         "noise_std": configs.dataset_args["noise_std"], "lr": 5e-3},
        {"mode": "3d", "sampling_ratio": configs.dataset_args["sampling_ratio"],
         "noise_std": configs.dataset_args["noise_std"], "lr": 1e-3},
        {"mode": "2d", "sampling_ratio": configs.dataset_args["sampling_ratio"],
         "noise_std": configs.dataset_args["noise_std"], "lr": 1e-3}
    ]

    # set 2
    hyper_params[2] = [
        {"mode": "3d", "sampling_ratio": configs.dataset_args["sampling_ratio"],
         "noise_std": configs.dataset_args["noise_std"], "lr": 5e-3},
        {"mode": "3d", "sampling_ratio": configs.dataset_args["sampling_ratio"],
         "noise_std": configs.dataset_args["noise_std"], "lr": 3e-3},
        {"mode": "3d", "sampling_ratio": configs.dataset_args["sampling_ratio"],
         "noise_std": configs.dataset_args["noise_std"], "lr": 1e-3},
        {"mode": "3d", "sampling_ratio": configs.dataset_args["sampling_ratio"],
         "noise_std": configs.dataset_args["noise_std"], "lr": 8e-4},
        {"mode": "3d", "sampling_ratio": configs.dataset_args["sampling_ratio"],
         "noise_std": configs.dataset_args["noise_std"], "lr": 5e-4}
    ]

    # set 3
    hyper_params[3] = [
        {"mode": "vtv", "sampling_ratio": 1 / 5,
         "noise_std": 0.01, "lr": 5e-3},
        {"mode": "vtv", "sampling_ratio": 1 / 10,
         "noise_std": 0.01, "lr": 5e-3},
        {"mode": "vtv", "sampling_ratio": 1 / 15,
         "noise_std": 0.01, "lr": 5e-3},
        {"mode": "vtv", "sampling_ratio": 1 / 20,
         "noise_std": 0.01, "lr": 5e-3},
        {"mode": "vtv", "sampling_ratio": 1 / 25,
         "noise_std": 0.01, "lr": 5e-3}
    ]

    # set 4
    hyper_params[4] = [
        {"mode": "vtv", "sampling_ratio": 1 / 5,
         "noise_std": 0.01, "lr": 5e-3},
        {"mode": "vtv", "sampling_ratio": 1 / 5,
         "noise_std": 0.02, "lr": 5e-3},
        {"mode": "vtv", "sampling_ratio": 1 / 5,
         "noise_std": 0.03, "lr": 5e-3},
        {"mode": "vtv", "sampling_ratio": 1 / 5,
         "noise_std": 0.04, "lr": 5e-3},
        {"mode": "vtv", "sampling_ratio": 1 / 5,
         "noise_std": 0.05, "lr": 5e-3}
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_num", type=int, choices=hyper_params.keys(), required=True)

    args = parser.parse_args()

    hyper_params_list = hyper_params[args.set_num]
    for hyper_param_dict_iter in hyper_params_list:
        filename = create_filename(hyper_param_dict_iter)
        bash_script = make_bash_script(hyper_param_dict_iter)
        subprocess.run(f"echo '{bash_script}' > {filename}", shell=True)
        # print(f"{filename}")
        # print(f"{bash_script}")
        # print("-" * 50)
