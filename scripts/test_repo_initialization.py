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
import ImageReconstructionProject.helpers.pytorch_utils as ptu


if __name__ == '__main__':
    X_test = torch.randn((2, 3)).to(ptu.device)
    X_test_np = ptu.to_numpy(X_test)
    print(f"{X_test}\n{X_test_np}")
