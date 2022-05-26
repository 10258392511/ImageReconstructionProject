import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import ImageReconstructionProject.scripts.configs as configs

from torch.utils.data import Dataset


def load_mat_data(path: str = None):
    if path is None:
        path = configs.dataset_args["path"]
    assert ".mat" in path
    data = scipy.io.loadmat(path)

    return data["imgs"].astype(np.float32)


def train_test_split(data, split_ratio=None, seed=None):
    # data: (H, W, T, B)
    if split_ratio is None:
        split_ratio = configs.dataset_args["split_ratio"]
    if seed is None:
        seed = configs.dataset_args["seed"]
    indices = np.arange(data.shape[-1])
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_ind = int(data.shape[-1] * split_ratio)
    train_ind, test_ind = indices[:split_ind], indices[split_ind:]
    data_train, data_test = data[..., train_ind], data[..., test_ind]

    return data_train, data_test


class MRIDataset(Dataset):
    def __init__(self, params, mode="train", dim="3d"):
        """
        params: configs.dataset_args
        """
        assert mode in ("train", "test")
        assert dim in ("3d", "2d")
        super(MRIDataset, self).__init__()
        self.params = params
        self.mode = mode
        self.dim = dim
        data_np = load_mat_data(self.params["path"])
        data_train, data_test = train_test_split(data_np, self.params["split_ratio"], self.params["seed"])
        if self.mode == "train":
            self.data = data_train
        else:
            self.data = data_test
        if self.dim == "2d":
            # 2d: (H, W, N_all)
            self.data = self.data.reshape((*self.data.shape[:2], -1))
        else:
            # 3d: (T, H, W, N)
            self.data = self.data.transpose(2, 0, 1, 3)

    def __len__(self):
        return self.data.shape[-1]

    def __getitem__(self, ind, return_orig_mask=False):
        img = self.data[..., ind][None, ...]  # (1, T, H, W) or (1, H, W)
        mask = self.generate_mask_()  # (1, T, 1, W) or (1, 1, W)
        img_k_space = np.fft.fftn(img, axes=(-1, -2))  # (1, T, H, W) or (1, H, W)
        img_k_space += (np.random.randn(*img_k_space.shape) + 1j * np.random.randn(*img_k_space.shape)) * \
                       self.params["noise_std"]
        img_k_space = np.fft.fftshift(img_k_space, axes=(-1, -2))
        img_k_space_masked = img_k_space * mask

        if not return_orig_mask:
            return img, img_k_space_masked, mask
        else:
            # for visualization
            return img, img_k_space_masked, mask, img_k_space

    def generate_mask_(self):
        # masking on the last dim: 3d: (1, T, 1, W); 2d: (1, 1, W)
        if self.dim == "3d":
            T, _, W, _ = self.data.shape
            mask = np.random.uniform(0., 1., (1, T, 1, W))
        else:
            _, W, _ = self.data.shape
            mask = np.random.uniform(0., 1., (1, 1, W))
        mask = (mask < self.params["sampling_ratio"])
        last_dim_center = mask.shape[-1] // 2
        start_ind, end_ind = last_dim_center - self.params["half_window_size"], \
                             last_dim_center + self.params["half_window_size"] + 1
        mask[..., start_ind:end_ind] = 1.

        return mask

    def visualize_data(self, img, img_k_space_masked, mask, img_k_space, **kwargs):
        """
        3d: (1, T, H, W), (1, T, H, W), (1, T, 1, W), (1, T, H, W)
        2d: (1, H, W), (1, H, W), (1, 1, W), (1, H, W)
        """
        if self.dim == "3d":
            # only visualize the first slice
            img, img_k_space_masked, mask, img_k_space = img[0, 0, ...], img_k_space_masked[0, 0, ...], \
                                                         mask[0, 0, 0, ...], img_k_space[0, 0, ...]
        else:
            img, img_k_space_masked, mask, img_k_space = img[0, ...], img_k_space_masked[0, ...], \
                                                         mask[0, 0, ...], img_k_space[0, ...]
        # (W,) -> (1, W) -> (H, W)
        mask = np.tile(mask[None, :], (img.shape[0], 1))
        zero_padded_recons = np.fft.ifftn(np.fft.ifftshift(img_k_space_masked, axes=(-1, -2)), axes=(-1, -2))
        zero_padded_recons = np.abs(zero_padded_recons)
        all_imgs = [img, mask, None, img_k_space, img_k_space_masked, zero_padded_recons]
        titles = ["img", "mask", None, "k-space", "masked k-space", "zero-padded reconstruction"]
        figsize = kwargs.get("figsize", (10.8, 7.2))
        fraction = kwargs.get("fraction", 0.2)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes_flatten = axes.flatten()

        for i in range(len(all_imgs)):
            img_iter = all_imgs[i]
            if img_iter is None:
                continue
            title = titles[i]
            axis = axes_flatten[i]
            if "k-space" in title:
                img_iter = np.log(np.abs(img_iter) + self.params["eps_log"])
            handle = axis.imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axis, fraction=fraction)
            axis.set_title(title)
        fig.delaxes(axes[0, -1])
        fig.tight_layout()
        plt.show()
