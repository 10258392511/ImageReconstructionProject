import sys

# change this to parent directory of the project, e.g.,
# mine is at "D:\testings\Python\TestingPython\ImageReconstructionProject"
path = r"D:\testings\Python\TestingPython"
if path not in sys.path:
    sys.path.append(path)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import ImageReconstructionProject.helpers.pytorch_utils as ptu

from ImageReconstructionProject.helpers.utils import download_file, unzip_file


if __name__ == '__main__':
    X_test = torch.randn((2, 3)).to(ptu.device)
    X_test_np = ptu.to_numpy(X_test)
    print(f"{X_test}\n{X_test_np}")

    url = "https://polybox.ethz.ch/index.php/s/SgPWhQEmKr0MtpO/download"
    save_dir = "./"
    filename = "data.zip"
    download_file(url, save_dir, filename)
    unzip_file(os.path.join(save_dir, filename), save_dir, if_remove_zip_file=True)
