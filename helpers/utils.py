import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile

from urllib.request import urlretrieve


def visualize_slice(data: np.ndarray, t, index):
    """
    I think only 3D visualization without additional info (e.g. segmentation) is needed, so I'll use vanilla
    visualization instead of SimpleITK. I use the following widget in Jupyter Notebook

    from ipywidgets import interact, fixed

    interact(visualize_slice, data=fixed(data_np), t=(0, data_np.shape[2] - 1), index=(0, data_np.shape[3] - 1))
    """
    assert 0 <= t < data.shape[2], 0 <= index < data.shape[3]
    fig, axis = plt.subplots()
    handle = axis.imshow(data[:, :, t, index], cmap="gray")
    plt.colorbar(handle, ax=axis)
    plt.show()


def download_file(url, save_dir, save_filename):
    print("downloading...")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename)
    urlretrieve(url, save_path)
    print("Done!")


def unzip_file(filename, save_dir, if_remove_zip_file=True):
    assert ".zip" in filename
    print("unzipping...")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with zipfile.ZipFile(filename) as zip_rf:
        zip_rf.extractall(save_dir)
    if if_remove_zip_file:
        os.remove(filename)
    print("Done!")
