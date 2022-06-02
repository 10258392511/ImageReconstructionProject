import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import ImageReconstructionProject.scripts.configs as configs
import ImageReconstructionProject.helpers.pytorch_utils as ptu

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from ImageReconstructionProject.models.vn_mri import VNMRI
from ImageReconstructionProject.helpers.utils import create_param_save_path
from ImageReconstructionProject.helpers.recon_utils import *


class VNMRITrainer(object):
    def __init__(self, model: VNMRI, train_loader, test_loader, params):
        """
        params:
        bash:
            batch_size, num_epochs, log_dir, param_save_dir, notebook, grad_clip_val
        """
        self.params = params
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.opt = configs.vn_mri_opt_params["class"](self.model.parameters(), **configs.vn_mri_opt_params["args"])
        self.scheduler = configs.vn_mri_opt_params["scheduler"](self.opt, **configs.vn_mri_opt_params["scheduler_args"])
        self.writer = SummaryWriter(self.params["log_dir"])
        self.global_steps = {"train": 0, "epoch": 0}
        self.loss = nn.L1Loss(reduction="sum")

    def train_(self):
        self.model.train()
        if self.params["notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.train_loader, total=len(self.train_loader), desc="training", leave=False)

        loss_avg = 0
        for i, (X_img, X_img_ks_mask, X_mask) in enumerate(pbar):
            # # TODO: comment out
            # if i == 2:
            #     break
            X_img = X_img.to(ptu.device)
            X_img_ks_mask = X_img_ks_mask.to(ptu.device)
            X_mask = X_mask.to(ptu.device)
            X_img_recons = self.model(X_img_ks_mask, X_mask)
            loss = self.loss(torch.abs(X_img_recons), torch.abs(X_img))
            self.opt.zero_grad()
            loss.backward()
            if self.params.get("grad_clip_val", None) is not None:
                nn.utils.clip_grad_value_(self.model.parameters(), self.params["grad_clip_val"])
            self.opt.step()

            # logging
            loss_avg += loss.item() * X_img.shape[0]
            self.writer.add_scalar("train_loss", loss.item(), self.global_steps["train"])
            self.global_steps["train"] += 1
            pbar.set_description(f"train loss: {loss.item(): .3f}")

        return loss_avg / len(self.train_loader.dataset)

    @torch.no_grad()
    def eval_(self):
        self.model.eval()
        if self.params["notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.test_loader, total=len(self.test_loader), desc="testing", leave=False)

        loss_avg = 0
        for i, (X_img, X_img_ks_mask, X_mask) in enumerate(pbar):
            # # TODO: comment out
            # if i == 2:
            #     break
            X_img = X_img.to(ptu.device)
            X_img_ks_mask = X_img_ks_mask.to(ptu.device)
            X_mask = X_mask.to(ptu.device)
            X_img_recons = self.model(X_img_ks_mask, X_mask)
            loss = self.loss(torch.abs(X_img_recons), torch.abs(X_img))

            # logging
            loss_avg += loss.item() * X_img.shape[0]
            pbar.set_description(f"eval loss: {loss.item(): .3f}")

        return loss_avg / len(self.test_loader.dataset)

    def train(self):
        if self.params["notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.params["num_epochs"], desc="epoch")

        best_eval_loss = float("inf")
        for epoch in pbar:
            train_loss = self.train_()
            eval_loss = self.eval_()
            self.scheduler.step()

            # save the best model
            if best_eval_loss > eval_loss:
                best_eval_loss = eval_loss
                self.save_model_()

            # logging
            self.writer.add_scalar("epoch_train_loss", train_loss, self.global_steps["epoch"])
            self.writer.add_scalar("epoch_eval_loss", eval_loss, self.global_steps["epoch"])
            self.end_of_epoch_eval_()
            self.global_steps["epoch"] += 1
            for param in self.opt.param_groups:
                break
            pbar.set_description(f"loss: train: {train_loss: .3f}, eval: {eval_loss: .3f}, lr: {param['lr']}")

    @torch.no_grad()
    def end_of_epoch_eval_(self):
        ind = np.random.randint(len(self.test_loader.dataset))
        X_img_gt, X_ks, X_mask = self.test_loader.dataset[ind]
        X_img_gt = torch.from_numpy(X_img_gt).to(ptu.device).unsqueeze(0)  # (1, 1, T, H, W) or (1, 1, H, W)
        X_ks = torch.from_numpy(X_ks).to(ptu.device).unsqueeze(0)
        X_mask = torch.from_numpy(X_mask).to(ptu.device).unsqueeze(0)

        X_ks, norm = self.model.normalize_ks_(X_ks, X_mask)
        X_img = k2i_complex(X_ks) * self.model.k0  # init X_img
        m = torch.zeros_like(X_img)  # running avg of gradients
        if self.model.params["mode"] == "2d":
            X_imgs_out = [self.model.denormalize_img_(X_img, norm)[0, 0]]
        else:
            X_imgs_out = [self.model.denormalize_img_(X_img, norm)[0, 0, 0]]  # store the first slice
        for layer_idx in range(self.model.params["num_layers"]):
            D_real = self.model.kernels_real[layer_idx, ...]
            D_imag = self.model.kernels_imag[layer_idx, ...]
            knots = self.model.knots[layer_idx, ...]
            alpha = self.model.alphas[layer_idx]
            momentum = self.model.momentums[layer_idx, ...]
            reg_grad = self.model.compute_reg_grad_(X_img, D_real, D_imag, knots)
            data_grad = self.model.compute_data_grad_(X_img, X_ks, X_mask)
            grad = alpha * data_grad + reg_grad
            m = momentum * m + grad
            X_img = X_img - m
            if self.model.params["mode"] == "2d":
                X_imgs_out.append(self.model.denormalize_img_(X_img, norm)[0, 0])
            else:
                X_imgs_out.append(self.model.denormalize_img_(X_img, norm)[0, 0, 0])

        if self.model.params["mode"] == "2d":
            X_imgs_out.append(X_img_gt[0, 0])
        else:
            X_imgs_out.append(X_img_gt[0, 0, 0])
        # make the plot
        # list[(H, W)] -> (N, 1, H, W)
        X_imgs_out = torch.stack(X_imgs_out, dim=0).unsqueeze(1).detach().cpu()
        imgs = torch.abs(make_grid(X_imgs_out, nrow=X_imgs_out.shape[0]))  # (1, H', W')
        if self.params["notebook"]:
            fig, axis = plt.subplots(figsize=(7.2, 7.2 * X_imgs_out.shape[0]))
            axis.imshow(ptu.to_numpy(imgs.permute(1, 2, 0)))
            plt.show()

        self.writer.add_image("epoch_eval_imgs", imgs, self.global_steps["epoch"])

    def save_model_(self):
        model_path = create_param_save_path(self.params["param_save_dir"], "model.pt")
        torch.save(self.model.state_dict(), model_path)
