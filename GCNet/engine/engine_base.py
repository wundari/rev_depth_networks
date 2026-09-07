# %% load necessary modules
import torch
import torch._dynamo
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from natsort import natsorted
from pathlib import Path
import glob
import sys
import inspect
import math

from dataset.scene_flow import SceneFlowFlyingThingsDataset
from dataset.scene_flow import SceneFlowMonkaaDataset
from modules.gcnet import build_gcnet
from utilities.misc import NestedTensor

from dataclasses import dataclass, asdict
import json
from jaxtyping import Float
from config.config import GCNetconfig

# reproducibility
import random
from config.config import GCNetconfig

config = GCNetconfig

seed_number = config.seed  # 1618  # 12321 #3407
torch.manual_seed(seed_number)
torch.cuda.manual_seed(seed_number)
random.seed(seed_number)
np.random.seed(seed_number)
os.environ["PYTHONHASHSEED"] = str(seed_number)


# initialize random seed number for dataloader
def seed_worker(worker_id):
    worker_seed = seed_number  # torch.initial_seed()  % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed_number)

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# %%
class Engine:

    def __init__(self, config: GCNetconfig) -> None:

        self.config = config
        self.device = config.device
        self.h = config.img_height
        self.w = config.img_width

        # dataset directory
        # parent_folder = "/media/wundari/S990Pro2_4TB"
        parent_folder = os.path.abspath(f"{os.curdir}/..")
        if config.dataset == "sceneflow_flying":
            self.datadir = f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/"
        elif config.dataset == "sceneflow_monkaa":
            self.datadir = f"{parent_folder}/Dataset/SceneFlow_complete/Monkaa/"

        # saving directory
        self.save_dir = os.path.join(
            "run", config.dataset, f"bino_interaction_{config.binocular_interaction}"
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # experiment directory
        runs = natsorted(glob.glob(os.path.join(self.save_dir, "experiment_*")))
        if config.load_state:
            run_id = int(runs[-1].split("_")[-1]) if runs else 0
        else:
            run_id = int(runs[-1].split("_")[-1]) + 1 if runs else 0
        self.experiment_dir = os.path.join(
            self.save_dir,
            f"experiment_{run_id}",
        )
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # save config file
        self.save_config()

        # pred_images directory
        self.pred_images_dir = os.path.join(self.experiment_dir, "pred_images")
        if not os.path.exists(self.pred_images_dir):
            os.makedirs(self.pred_images_dir)

        # load model
        self.model = build_gcnet(config)
        # load pre-trained GCNet, if provided
        if config.load_state:
            print(
                "load pretrained GCNet:\n"
                + f"binocular interaction: {config.binocular_interaction}\n"
                + f"experiment id: {config.experiment_id}\n"
                + f"model: {config.resume}"
            )
            self.experiment_dir = os.path.join(
                self.save_dir,
                f"experiment_{config.experiment_id}",
            )
            checkpoint = torch.load(
                f"{self.experiment_dir}/{config.resume}", map_location="cuda"
            )
            pretrained_dict = checkpoint["state_dict"]

            # fix the keys of the state dictionary
            unwanted_prefix = "_orig_mod."
            for k, v in list(pretrained_dict.items()):
                if k.startswith(unwanted_prefix):
                    pretrained_dict[k[len(unwanted_prefix) :]] = pretrained_dict.pop(k)
            self.model.load_state_dict(pretrained_dict)
        # compile model
        if config.compile_mode is not None:
            self.model = torch.compile(
                self.model, mode=config.compile_mode
            )  # use compile_mode = "default" for layer analysis
        self.model.to(self.device)

        # if self.train_or_eval_mode == "train":
        # self.model.train()  # training mode
        print(
            f"GCNet was successfully loaded to {self.device}, "
            + f"binocular interaction: {config.binocular_interaction}\n"
            + f"compile mode: {config.compile_mode}\n"
            + f"experiment dir: {self.experiment_dir}\n"
        )
        print(
            f"GCNet will be trained on {config.dataset} dataset "
            + f"with batch size {config.batch_size} for {config.epochs} epochs"
        )

    def save_config(self):
        config_dict = self.config.to_dict()
        with open(f"{self.experiment_dir}/config.json", "w") as f:
            json.dump(config_dict, f)

    def prepare_dataset(self):
        # prepare dataset
        # train_list = ["driving", "flying", "monkaa"]
        # val_list = ["driving", "flying", "monkaa"]

        if self.config.dataset == "sceneflow_flying":
            dataset_train = SceneFlowFlyingThingsDataset(
                self.datadir, self.config, "train"
            )
            dataset_validation = SceneFlowFlyingThingsDataset(
                self.datadir, self.config, "validation"
            )
            dataset_test = SceneFlowFlyingThingsDataset(
                self.datadir, self.config, "test"
            )
        elif self.config.dataset == "sceneflow_monkaa":
            dataset_train = SceneFlowMonkaaDataset(self.datadir, self.config, "train")
            dataset_validation = SceneFlowMonkaaDataset(
                self.datadir, self.config, "validation"
            )
            dataset_test = SceneFlowMonkaaDataset(self.datadir, self.config, "test")

        data_loader_train = data.DataLoader(
            dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        data_loader_validation = data.DataLoader(
            dataset_validation,
            batch_size=self.config.batch_size_val,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        data_loader_test = data.DataLoader(
            dataset_test,
            batch_size=self.config.batch_size_val,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        return (data_loader_train, data_loader_validation, data_loader_test)

    def check_input(self, data_loader):
        # check input
        inputs = next(iter(data_loader))
        left = inputs["left"]
        right = inputs["right"]
        disp = inputs["disp"]
        #
        i = 0
        img_left = (left[i] / left[i].max() * 128 + 127).to(torch.uint8)
        img_right = (right[i] / right[i].max() * 128 + 127).to(torch.uint8)
        img_disp = disp[i]

        ## create patch
        # generate random patch location
        patch_size = 100
        b, c, h, w = left.size()
        row_start = random.randint(0, h - patch_size)
        row_end = row_start + patch_size
        col_start = random.randint(0, w - patch_size)
        col_end = col_start + patch_size

        # left patch
        patch_left = img_left[:, row_start:row_end, col_start:col_end]

        # right patch
        patch_right = img_right[:, row_start:row_end, col_start:col_end]

        # disparity patch, the patch location is the same as left patch
        patch_disp = img_disp[row_start:row_end, col_start:col_end]
        # patch_disp = img_disp_shifted[row_start:row_end, col_start:col_end]

        # shift right image
        flip_input = 0
        patch_right_shift = torch.zeros((3, patch_size, patch_size), dtype=torch.uint8)

        for i in range(patch_size):
            id_row = row_start + i
            for j in range(patch_size):
                if flip_input:
                    # shift with respect to right disparity image
                    id_col = (col_start + j + patch_disp[i, j]).to(torch.int)
                else:
                    # shift with respect to left disparity image
                    id_col = (col_start + j - patch_disp[i, j]).to(torch.int)

                patch_right_shift[:, i, j] = img_right[:, id_row, id_col]

        fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=3)
        v_min = -100
        v_max = 100
        ## draw left patch
        axes[0].imshow(img_left.permute(1, 2, 0))
        axes[0].set_title("Left patch")
        axes[0].axis("off")
        # draw box
        axes[0].plot([col_start, col_end], [row_start, row_start], "r-")
        axes[0].plot([col_start, col_start], [row_start, row_end], "r-")
        axes[0].plot([col_start, col_end], [row_end, row_end], "r-")
        axes[0].plot([col_end, col_end], [row_start, row_end], "r-")
        ## draw right patch
        axes[1].imshow(img_right.permute(1, 2, 0))
        axes[1].set_title("Right patch")
        axes[1].axis("off")
        # draw box
        axes[1].plot([col_start, col_end], [row_start, row_start], "r-")
        axes[1].plot([col_start, col_start], [row_start, row_end], "r-")
        axes[1].plot([col_start, col_end], [row_end, row_end], "r-")
        axes[1].plot([col_end, col_end], [row_start, row_end], "r-")
        ## draw disp map
        axes[2].imshow(img_disp, cmap="jet", vmin=v_min, vmax=v_max)
        axes[2].set_title("Disparity map (left)")
        axes[2].axis("off")

        fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=4)
        axes[0].imshow(patch_left.permute(1, 2, 0))
        axes[0].set_title("Left patch")
        axes[0].axis("off")
        axes[1].imshow(patch_right.permute(1, 2, 0))
        axes[1].set_title("Right patch")
        axes[1].axis("off")

        # shifted image
        axes[2].imshow(patch_right_shift.permute(1, 2, 0))
        axes[2].set_title("Shifted right image")
        axes[2].axis("off")

        # disparity map
        temp = axes[3].imshow(patch_disp, cmap="jet", vmin=v_min, vmax=v_max)
        axes[3].set_title("Disparity map (left)")
        axes[3].axis("off")
        # colorbar
        l_ax, b_ax, w_ax, h_ax = axes[3].get_position().bounds
        cax = plt.gcf().add_axes([l_ax + w_ax + 0.03, b_ax, 0.03, h_ax])
        cbar_ticks = np.arange(v_min, v_max + 1, 50)
        cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks)

        print(
            "mean-disp: {:.2f}, min-disp: {}, max-disp: {}".format(
                img_disp.float().mean(), img_disp.min(), img_disp.max()
            )
        )

    def save_checkpoint(self, epoch, iter, optimizer, best=False):
        """
        Save current state of training
        """

        # save model
        checkpoint = {
            "epoch": epoch,
            "iter": iter,
            "state_dict": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "lr_scheduler": lr_scheduler.state_dict(),
            # "best_pred": prev_best,
        }

        if best:
            filename = (
                "epoch_" + str(epoch) + "_iter_" + str(iter) + "_model_best.pth.tar"
            )
            filename = os.path.join(self.experiment_dir, filename)
            torch.save(checkpoint, filename)
        else:
            filename = "epoch_" + str(epoch) + "_model.pth.tar"
            filename = os.path.join(self.experiment_dir, filename)
            torch.save(checkpoint, filename)

    def configure_optimizers(self, weight_decay, learning_rate):

        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups.
        # Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embedding decay,
        # all biases and layernorms dont't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, "
            + f"with {num_decay_params} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            + f"with {num_nodecay_params} parameters"
        )

        # create adamw optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )

        return optimizer

    def get_lr(self, it, max_steps):

        if self.config.warmup_steps is not None:
            # 1 linear warmup for warmup_iters steps
            if it < self.config.warmup_steps:
                return self.config.max_lr * (it + 1) / self.config.warmup_steps

            # 2 if it > lr_decay_iters, return min learning rate
            if it > max_steps:
                return self.config.min_lr

            # 3 in between, use cosine decay down to min learning rate
            decay_ratio = (it - self.config.warmup_steps) / (
                max_steps - self.config.warmup_steps
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (
                1.0 + math.cos(math.pi * decay_ratio)
            )  # coeff starts at 1 and goes to 0

            return self.config.min_lr + coeff * (
                self.config.max_lr - self.config.min_lr
            )

        else:

            # 2 if it > lr_decay_iters, return min learning rate
            if it > max_steps:
                return self.config.min_lr

            # 3 in between, use cosine decay down to min learning rate
            decay_ratio = it / max_steps
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (
                1.0 + math.cos(math.pi * decay_ratio)
            )  # coeff starts at 1 and goes to 0

            return self.config.min_lr + coeff * (
                self.config.max_lr - self.config.min_lr
            )

    def train(self, train_loader, val_loader):

        # build loss criterion
        if self.config.loss == "smooth_l1":
            criterion = nn.SmoothL1Loss()
        elif self.config.loss == "l1":
            criterion = nn.L1Loss()

        # configure optimizer
        optimizer = self.configure_optimizers(self.config.weight_decay, self.config.lr)
        # optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)

        # scheduler
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=np.arange(1, self.config.epochs), gamma=0.2
        # )

        scaler = torch.cuda.amp.GradScaler()
        if self.config.epochs > 10:
            max_steps_lr = 10 * len(train_loader)
        else:
            max_steps_lr = self.config.epochs * len(train_loader)
        max_steps = self.config.epochs * len(train_loader)
        max_norm = 1.0
        losses_train = []
        accs_train = []  # 3-pix accuracy for training
        losses_val = []
        accs_val = []  # 3-pix accuracy for val
        loss_val_prev = np.inf

        for epoch in range(self.config.start_epoch, self.config.epochs):
            tepoch = tqdm(train_loader)

            # for i in range(iter_num, len(train_loader)):
            for idx, inputs in enumerate(tepoch):
                step = epoch * len(train_loader) + idx

                # once in a while evaluate validation loss
                if step % self.config.eval_interval == 0:
                    self.model.eval()

                    with torch.no_grad():
                        loss_val_accu = 0  # val loss accumulate
                        acc_val_accu = 0  # 3-pix accuracy accumulate
                        for j in range(self.config.eval_iter):

                            inputs = next(iter(val_loader))

                            # build NestedTensor
                            inputs = NestedTensor(
                                inputs["left"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                                inputs["right"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                                disp=inputs["disp"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                                ref=inputs["ref"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                            )

                            with torch.autocast(
                                device_type=self.device, dtype=torch.float16
                            ):
                                disp_pred = self.model(inputs)

                                # compute loss
                                loss_val = criterion(inputs.disp, disp_pred)

                                # accumulate val loss
                                loss_val_accu += loss_val

                                # compute accuracy (3 pixel error for val dataset)
                                diff = torch.abs(disp_pred - inputs.disp)
                                acc_val_accu += torch.sum(diff < 3)

                        # print(f'val loss: {loss_val["aggregated"].item():.4f}')
                        # gather validation loss
                        loss_val_mean = loss_val_accu / self.config.eval_iter
                        losses_val.append(loss_val_mean.item())

                        # gather 3-pix accuracy val
                        acc_val_mean = acc_val_accu / float(
                            self.h
                            * self.w
                            * self.config.batch_size_val
                            * self.config.eval_iter
                        )
                        accs_val.append(acc_val_mean.item())

                        # save best model where loss_val[i] < loss_val[i - 1]
                        if loss_val_mean < loss_val_prev:
                            print(
                                "found best model, "
                                + f"loss_val_curr: {loss_val_mean.item():.4f} "
                                + f"loss_val_prev: {loss_val_prev:.4f}, saving"
                            )
                            self.save_checkpoint(epoch, optimizer, best=True)
                            loss_val_prev = loss_val_mean.item()

                        ## once a while check predicted disparity on the val loader
                        left = inputs.left.to("cpu")
                        right = inputs.right.to("cpu")
                        disp = inputs.disp.to("cpu")
                        disp_pred = disp_pred.data.cpu().numpy()
                        # normalize to (0, 255), for visualization
                        img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
                        img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

                        # visualize
                        figsize = (16, 10)
                        vmin = -100
                        vmax = 100
                        sns.set_theme()
                        sns.set_theme(
                            context="paper", style="white", font_scale=2, palette="deep"
                        )

                        fig, axes = plt.subplots(
                            nrows=self.config.batch_size_val, ncols=4, figsize=figsize
                        )
                        if self.config.batch_size_val == 1:
                            # left image
                            axes[0].imshow(img_left[0].permute(1, 2, 0))
                            axes[0].set_title("Left")

                            # right image
                            axes[1].imshow(img_right[0].permute(1, 2, 0))
                            axes[1].set_title("Right")

                            # predicted disparity
                            axes[2].imshow(
                                disp_pred[0],
                                cmap="jet",
                                vmin=vmin,
                                vmax=vmax,
                            )
                            axes[2].set_title("Pred. disparity")

                            # disparity ground truth
                            temp = axes[3].imshow(
                                disp[0], cmap="jet", vmin=vmin, vmax=vmax
                            )
                            axes[3].set_title("Ground truth")

                            # colorbar
                            l_ax, b_ax, w_ax, h_ax = axes[3].get_position().bounds
                            cax = plt.gcf().add_axes(
                                [l_ax + w_ax + 0.03, b_ax, 0.03, h_ax]
                            )
                            cbar_ticks = np.arange(vmin, vmax + 1, 50)
                            cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
                            cbar.ax.set_yticklabels(cbar_ticks)
                        else:
                            for k in range(self.config.batch_size_val):
                                # left image
                                axes[k, 0].imshow(img_left[k].permute(1, 2, 0))
                                axes[k, 0].set_title("Left")

                                # right image
                                axes[k, 1].imshow(img_right[k].permute(1, 2, 0))
                                axes[k, 1].set_title("Right")

                                # predicted disparity
                                axes[k, 2].imshow(
                                    disp_pred[k],
                                    cmap="jet",
                                    vmin=vmin,
                                    vmax=vmax,
                                )
                                axes[k, 2].set_title("Pred. disparity")

                                # disparity ground truth
                                temp = axes[k, 3].imshow(
                                    disp[k], cmap="jet", vmin=vmin, vmax=vmax
                                )
                                axes[k, 3].set_title("Ground truth")

                                # colorbar
                                l_ax, b_ax, w_ax, h_ax = (
                                    axes[k, 3].get_position().bounds
                                )
                                cax = plt.gcf().add_axes(
                                    [l_ax + w_ax + 0.03, b_ax, 0.03, h_ax]
                                )
                                cbar_ticks = np.arange(vmin, vmax + 1, 50)
                                cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
                                cbar.ax.set_yticklabels(cbar_ticks)

                        # turn off axis for all subplots
                        for ax in axes.ravel():
                            ax.set_axis_off()

                        plt.savefig(
                            f"{self.pred_images_dir}/output_val.pdf",
                            dpi=600,
                            bbox_inches="tight",
                        )
                        plt.close()

                # training loop
                self.model.train()
                optimizer.zero_grad()

                inputs = next(iter(train_loader))
                # print(inputs["left"].size())

                # build nested tensor
                inputs = NestedTensor(
                    inputs["left"].pin_memory().to(self.device, non_blocking=True),
                    inputs["right"].pin_memory().to(self.device, non_blocking=True),
                    disp=inputs["disp"].pin_memory().to(self.device, non_blocking=True),
                    ref=inputs["ref"].pin_memory().to(self.device, non_blocking=True),
                )

                # forward pass
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    disp_pred = self.model(inputs)

                    # compute loss
                    loss_train = criterion(inputs.disp, disp_pred)

                    # compute 3-pix acc
                    diff = torch.abs(disp_pred - inputs.disp)
                    acc_train = (diff < 3).float().mean()

                # gather training L1 loss each iteration
                losses_train.append(loss_train.item())

                # gather 3-pix accuracies
                accs_train.append(acc_train.item())

                # terminate training if exploded
                if not math.isfinite(loss_train.item()):
                    print(f"Loss is {loss_train.item()}, stopping training")
                    sys.exit(1)

                # backprop
                scaler.scale(loss_train).backward()

                # clip norm
                # if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                # update learning rate
                lr = self.get_lr(step, max_steps_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # step optimizer
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                # sync cuda
                torch.cuda.synchronize()

                tepoch.set_description(
                    f"step {step}/{max_steps} |"
                    + f"train loss: {loss_train.item():.4f} |"
                    + f"train 3-pix acc: {acc_train.item():.4f} |"
                    + f"val loss: {loss_val_mean:.4f} |"
                    + f"val 3-pix acc: {acc_val_mean.item():.4f} |"
                    + f"lr: {optimizer.param_groups[0]['lr']:.4e}"
                )

                # clear cache
                torch.cuda.empty_cache()

            # update lr
            # scheduler.step()

            # save model each epoch
            self.save_checkpoint(epoch, optimizer)

        # save train and val losses
        np.save(f"{self.experiment_dir}/losses_train.npy", losses_train)
        np.save(f"{self.experiment_dir}/losses_val.npy", losses_val)
        np.save(f"{self.experiment_dir}/accs_train.npy", accs_train)
        np.save(f"{self.experiment_dir}/accs_val.npy", accs_val)

    def train_v2(self, train_loader, val_loader):

        # build loss criterion
        if self.config.loss == "smooth_l1":
            criterion = nn.SmoothL1Loss()
        elif self.config.loss == "l1":
            criterion = nn.L1Loss()

        # configure optimizer
        optimizer = self.configure_optimizers(self.config.weight_decay, self.config.lr)
        # optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)

        # scheduler
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=np.arange(1, self.config.epochs), gamma=0.2
        # )

        scaler = torch.cuda.amp.GradScaler()
        if self.config.epochs > 10:
            max_steps_lr = 10 * len(train_loader)
        else:
            max_steps_lr = self.config.epochs * len(train_loader)
        max_steps = self.config.epochs * len(train_loader)
        max_norm = 1.0
        losses_train = []
        accs_train = []  # 3-pix accuracy for training
        losses_val = []
        accs_val = []  # 3-pix accuracy for val
        loss_val_prev = np.inf

        for epoch in range(self.config.start_epoch, self.config.epochs):
            tepoch = tqdm(train_loader)

            self.model.train()
            # for i in range(iter_num, len(train_loader)):
            for idx, inputs in enumerate(tepoch):
                step = epoch * len(train_loader) + idx

                # training loop
                inputs = next(iter(train_loader))
                # print(inputs["left"].size())

                # build nested tensor
                inputs = NestedTensor(
                    inputs["left"].pin_memory().to(self.device, non_blocking=True),
                    inputs["right"].pin_memory().to(self.device, non_blocking=True),
                    disp=inputs["disp"].pin_memory().to(self.device, non_blocking=True),
                    ref=inputs["ref"].pin_memory().to(self.device, non_blocking=True),
                )

                # zero the gradients
                optimizer.zero_grad()

                # forward pass
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    disp_pred = self.model(inputs)

                    # compute loss
                    loss_train = criterion(inputs.disp, disp_pred)

                    # compute 3-pix acc
                    diff = torch.abs(disp_pred - inputs.disp)
                    acc_train = (diff < 3).float().mean()

                # terminate training if exploded
                if not math.isfinite(loss_train.item()):
                    print(f"Loss is {loss_train.item()}, stopping training")
                    sys.exit(1)

                # backprop
                scaler.scale(loss_train).backward()

                # clip norm
                # if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                # update learning rate
                lr = self.get_lr(step, max_steps_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # step optimizer
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                # sync cuda
                torch.cuda.synchronize()

                tepoch.set_description(
                    f"step {step}/{max_steps} |"
                    + f"train loss: {loss_train.item():.4f} |"
                    + f"train 3-pix acc: {acc_train.item():.4f} |"
                    + f"lr: {optimizer.param_groups[0]['lr']:.4e}"
                )

                # clear cache
                torch.cuda.empty_cache()

                # update lr
                # scheduler.step()

                # once in a while evaluate validation loss
                if (step >= 0) & (step % self.config.eval_interval == 0):
                    self.model.eval()

                    with torch.no_grad():
                        loss_train_accu = 0  # train loss accumulate
                        acc_train_accu = 0  # 3-pix acc train accumulate
                        loss_val_accu = 0  # val loss accumulate
                        acc_val_accu = 0  # 3-pix acc val accumulate
                        for j in range(self.config.eval_iter):

                            ################
                            ## train loss ##
                            ################
                            inputs = next(iter(train_loader))

                            # build NestedTensor
                            inputs = NestedTensor(
                                inputs["left"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                                inputs["right"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                                disp=inputs["disp"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                                ref=inputs["ref"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                            )

                            with torch.autocast(
                                device_type=self.device, dtype=torch.float16
                            ):
                                disp_pred = self.model(inputs)

                                # compute loss
                                loss_train = criterion(inputs.disp, disp_pred)

                                # accumulate val loss
                                loss_train_accu += loss_train

                                # compute accuracy (3 pixel error for train dataset)
                                diff = torch.abs(disp_pred - inputs.disp)
                                acc_train_accu += torch.sum(diff < 3)

                            #####################
                            ## validation loss ##
                            #####################
                            inputs = next(iter(val_loader))

                            # build NestedTensor
                            inputs = NestedTensor(
                                inputs["left"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                                inputs["right"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                                disp=inputs["disp"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                                ref=inputs["ref"]
                                .pin_memory()
                                .to(self.device, non_blocking=True),
                            )

                            with torch.autocast(
                                device_type=self.device, dtype=torch.float16
                            ):
                                disp_pred = self.model(inputs)

                                # compute loss
                                loss_val = criterion(inputs.disp, disp_pred)

                                # accumulate val loss
                                loss_val_accu += loss_val

                                # compute accuracy (3 pixel error for val dataset)
                                diff = torch.abs(disp_pred - inputs.disp)
                                acc_val_accu += torch.sum(diff < 3)

                        # gather train loss every eval_interval
                        loss_train_mean = loss_train_accu / self.config.eval_iter
                        losses_train.append(loss_train_mean.item())

                        # gather validation loss every eval_interval
                        loss_val_mean = loss_val_accu / self.config.eval_iter
                        losses_val.append(loss_val_mean.item())

                        # gather 3-pix accuracy train every eval_interval
                        acc_train_mean = acc_train_accu / float(
                            self.h
                            * self.w
                            * self.config.batch_size_val
                            * self.config.eval_iter
                        )
                        accs_train.append(acc_train_mean.item())

                        # gather 3-pix accuracy val every eval_interval
                        acc_val_mean = acc_val_accu / float(
                            self.h
                            * self.w
                            * self.config.batch_size_val
                            * self.config.eval_iter
                        )
                        accs_val.append(acc_val_mean.item())

                        tepoch.set_postfix(
                            trainloss=loss_train_mean.item(),
                            valloss=loss_val_mean.item(),
                            trainacc=acc_train_mean.item(),
                            valacc=acc_val_mean.item(),
                        )

                        # save best model where loss_val[i] < loss_val[i - 1]
                        if loss_val_mean < loss_val_prev:
                            print(
                                "found best model, "
                                + f"loss_val_curr: {loss_val_mean.item():.4f} "
                                + f"loss_val_prev: {loss_val_prev:.4f}, saving"
                            )
                            self.save_checkpoint(epoch, step, optimizer, best=True)
                            loss_val_prev = loss_val_mean.item()

                        ## once a while check predicted disparity on the val loader
                        left = inputs.left.to("cpu")
                        right = inputs.right.to("cpu")
                        disp = inputs.disp.to("cpu")
                        disp_pred = disp_pred.data.cpu().numpy()
                        # normalize to (0, 255), for visualization
                        img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
                        img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

                        # visualize
                        figsize = (16, 10)
                        vmin = -100
                        vmax = 100
                        sns.set_theme()
                        sns.set_theme(
                            context="paper", style="white", font_scale=2, palette="deep"
                        )

                        fig, axes = plt.subplots(
                            nrows=self.config.batch_size_val, ncols=4, figsize=figsize
                        )
                        if self.config.batch_size_val == 1:
                            # left image
                            axes[0].imshow(img_left[0].permute(1, 2, 0))
                            axes[0].set_title("Left")

                            # right image
                            axes[1].imshow(img_right[0].permute(1, 2, 0))
                            axes[1].set_title("Right")

                            # predicted disparity
                            axes[2].imshow(
                                disp_pred[0],
                                cmap="jet",
                                vmin=vmin,
                                vmax=vmax,
                            )
                            axes[2].set_title("Pred. disparity")

                            # disparity ground truth
                            temp = axes[3].imshow(
                                disp[0], cmap="jet", vmin=vmin, vmax=vmax
                            )
                            axes[3].set_title("Ground truth")

                            # colorbar
                            l_ax, b_ax, w_ax, h_ax = axes[3].get_position().bounds
                            cax = plt.gcf().add_axes(
                                [l_ax + w_ax + 0.03, b_ax, 0.03, h_ax]
                            )
                            cbar_ticks = np.arange(vmin, vmax + 1, 50)
                            cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
                            cbar.ax.set_yticklabels(cbar_ticks)
                        else:
                            for k in range(self.config.batch_size_val):
                                # left image
                                axes[k, 0].imshow(img_left[k].permute(1, 2, 0))
                                axes[k, 0].set_title("Left")

                                # right image
                                axes[k, 1].imshow(img_right[k].permute(1, 2, 0))
                                axes[k, 1].set_title("Right")

                                # predicted disparity
                                axes[k, 2].imshow(
                                    disp_pred[k],
                                    cmap="jet",
                                    vmin=vmin,
                                    vmax=vmax,
                                )
                                axes[k, 2].set_title("Pred. disparity")

                                # disparity ground truth
                                temp = axes[k, 3].imshow(
                                    disp[k], cmap="jet", vmin=vmin, vmax=vmax
                                )
                                axes[k, 3].set_title("Ground truth")

                                # colorbar
                                l_ax, b_ax, w_ax, h_ax = (
                                    axes[k, 3].get_position().bounds
                                )
                                cax = plt.gcf().add_axes(
                                    [l_ax + w_ax + 0.03, b_ax, 0.03, h_ax]
                                )
                                cbar_ticks = np.arange(vmin, vmax + 1, 50)
                                cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
                                cbar.ax.set_yticklabels(cbar_ticks)

                        # turn off axis for all subplots
                        for ax in axes.ravel():
                            ax.set_axis_off()

                        plt.savefig(
                            f"{self.pred_images_dir}/output_val.pdf",
                            dpi=600,
                            bbox_inches="tight",
                        )
                        plt.close()

                    # train mode
                    self.model.train()

            # save model each epoch
            self.save_checkpoint(epoch, step, optimizer)

        # save train and val losses
        np.save(f"{self.experiment_dir}/losses_train.npy", losses_train)
        np.save(f"{self.experiment_dir}/losses_val.npy", losses_val)
        np.save(f"{self.experiment_dir}/accs_train.npy", accs_train)
        np.save(f"{self.experiment_dir}/accs_val.npy", accs_val)

    def plotLine_learning_curve(self, save_flag):

        losses_train = np.load(f"{self.experiment_dir}/losses_train.npy")
        losses_val = np.load(f"{self.experiment_dir}/losses_val.npy")

        # average losses_train every eval_interval
        losses_train_avg = []
        for i in range(0, len(losses_train), self.config.eval_interval):
            i_start = i
            i_end = i_start + self.config.eval_interval
            losses_train_avg.append(np.mean(losses_train[i_start:i_end]))

        assert len(losses_val) == len(losses_train_avg)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (14, 4)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharex=True)

        # fig.text(0.5, 1.02, "Training loss", ha="center")
        # fig.text(-0.01, 0.5, "L1-loss", va="center", rotation=90)
        fig.text(0.5, -0.04, "Step x {}".format(self.config.eval_interval), ha="center")

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.25, hspace=0.25)

        axes.plot(losses_train_avg, linewidth=2)
        axes.plot(losses_val, linewidth=2)

        x_low = 0
        x_up = len(losses_val)
        x_step = 10
        y_low = 0
        y_up = 31
        y_step = 5

        axes.set_ylabel("L1-loss")
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_ylim(y_low, y_up)

        plt.legend(["train", "val"], loc="upper right")

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        if save_flag:
            plt.savefig(
                f"{self.experiment_dir}/learning_curve.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_learning_curve_v2(self, save_flag):

        losses_train = np.load(f"{self.experiment_dir}/losses_train.npy")
        losses_val = np.load(f"{self.experiment_dir}/losses_val.npy")
        accs_train = np.load(f"{self.experiment_dir}/accs_train.npy")
        accs_val = np.load(f"{self.experiment_dir}/accs_val.npy")

        assert len(losses_val) == len(losses_train)
        assert len(accs_val) == len(accs_train)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (16, 5)
        n_row = 1
        n_col = 2

        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharex=True)

        # fig.text(0.5, 1.02, "Training loss", ha="center")
        # fig.text(-0.01, 0.5, "L1-loss", va="center", rotation=90)
        fig.text(
            0.5,
            -0.04,
            f"Step (averaged across {self.config.batch_size * self.config.eval_iter} trials, "
            + f"sampled every {self.config.eval_interval} steps)",
            ha="center",
        )

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.25, hspace=0.25)

        axes[0].plot(losses_train, linewidth=2)
        axes[0].plot(losses_val, linewidth=2)
        axes[1].plot(accs_train, linewidth=2)
        axes[1].plot(accs_val, linewidth=2)

        if self.config.dataset == "sceneflow_monkaa":
            x_low = 0
            x_step = 25
            x_up = len(losses_val) + x_step

        elif self.config.dataset == "sceneflow_flying":
            x_low = 0
            x_step = 100
            x_up = len(losses_val) + x_step

        y_low = 0
        y_up = 31
        y_step = 5

        axes[0].set_ylabel("L1-loss")
        axes[0].set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[0].set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[0].set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[0].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[0].set_ylim(y_low, y_up)

        y_low = 0.5
        y_up = 1.05
        y_step = 0.1

        axes[1].set_ylabel("3-pix acc")
        axes[1].set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[1].set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[1].set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[1].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[1].set_ylim(y_low, y_up)

        plt.legend(["train", "val"], loc="lower right")

        # Hide the right and top spines
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)

        # On[0]ly show ticks on the left and bottom spines
        axes[0].yaxis.set_ticks_position("left")
        axes[0].xaxis.set_ticks_position("bottom")
        axes[1].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")

        if save_flag:
            plt.savefig(
                f"{self.experiment_dir}/learning_curve.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plot_learning_rate(self, train_loader, save_flag):
        # plot learning rate
        max_steps_lr = 10 * len(train_loader)
        max_steps = self.config.epochs * len(train_loader)
        lr = np.empty(max_steps, dtype=np.float32)
        for i in range(max_steps):
            lr[i] = self.get_lr(i, max_steps_lr)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "Helvetica",
                # "font.sans-serif": "Helvetica",
            }
        )

        figsize = (14, 4)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharex=True)
        fig.text(
            0.5,
            -0.04,
            "Step",
            ha="center",
        )

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.25, hspace=0.25)

        c = 10000  # multiplier
        axes.plot(lr * c, linewidth=2)

        x_low = 0
        x_step = 2500
        x_up = len(lr) + x_step
        y_low = 0
        y_step = 1
        y_up = self.config.max_lr * 10000 + y_step

        axes.set_ylabel(r"Learning rate ($\times 10^{-4}$)")
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_ylim(y_low, y_up)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        if save_flag:
            plt.savefig(
                f"{self.experiment_dir}/learning_rate.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def inference_val(self, data_loader_validation):

        inputs = next(iter(data_loader_validation))
        left = inputs["left"]
        right = inputs["right"]
        disp = inputs["disp"]
        ref = inputs["ref"]

        input_data = NestedTensor(
            left.to(self.config.device),
            right.to(self.config.device),
            disp.to(self.config.device),
            ref,
        )
        # inference
        self.model.eval()
        with torch.autocast(device_type=self.config.device, dtype=torch.float16):
            disp_pred = self.model(input_data)

        # visualize output
        # normalize to (0, 255), for visualization
        img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
        img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

        fig, axes = plt.subplots(
            nrows=self.config.batch_size_val, ncols=4, figsize=(12, 5)
        )
        for i in range(self.config.batch_size_val):
            # left image
            axes[i, 0].imshow(img_left[i].permute(1, 2, 0))
            axes[i, 0].set_title("Left")
            axes[i, 0].axis("off")

            # right image
            axes[i, 1].imshow(img_right[i].permute(1, 2, 0))
            axes[i, 1].set_title("Right")
            axes[i, 1].axis("off")

            # predicted disparity
            vmin = -50
            vmax = 50
            axes[i, 2].imshow(
                disp_pred[i].detach().cpu().numpy(), cmap="jet", vmin=vmin, vmax=vmax
            )
            axes[i, 2].set_title("Pred. disparity")
            axes[i, 2].axis("off")

            # disparity ground truth
            temp = axes[i, 3].imshow(disp[i], cmap="jet", vmin=vmin, vmax=vmax)
            axes[i, 3].set_title("Ground truth")
            axes[i, 3].axis("off")
            # colorbar
            l_ax, b_ax, w_ax, h_ax = axes[i, 3].get_position().bounds
            cax = plt.gcf().add_axes([l_ax + w_ax + 0.03, b_ax, 0.03, h_ax])
            cbar_ticks = np.arange(vmin, vmax + 1, 50)
            cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
            cbar.ax.set_yticklabels(cbar_ticks)

        plt.savefig(
            f"{self.pred_images_dir}/output_test.pdf",
            dpi=600,
            bbox_inches="tight",
        )
