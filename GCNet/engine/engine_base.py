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

import random
from dataset.loading import make_loader, make_train_eval_loader, batches_for_evaluation


# %%
class Engine:

    def __init__(self, config: GCNetconfig) -> None:

        self.config = config
        self.device = config.device
        random.seed(config.seed)
        np.random.seed(config.seed % 2**32)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = not config.deterministic
        torch.backends.cudnn.deterministic = config.deterministic
        if config.amp_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unknown amp_dtype")
        if torch.device(self.device).type == "cuda" and config.amp_dtype == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This GPU does not support BF16; choose float16")
        self.h = config.img_height
        self.w = config.img_width

        # dataset directory
        # parent_folder = "/media/wundari/S990Pro2_4TB"
        parent_folder = os.path.abspath(f"{os.curdir}/../..")
        if config.dataset == "sceneflow_flying":
            self.datadir = f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/"
        elif config.dataset == "sceneflow_monkaa":
            self.datadir = f"{parent_folder}/Dataset/SceneFlow_complete/Monkaa/"

        else:
            raise ValueError(f"Unsupported dataset: {config.dataset}")
        if config.dataset_directory:
            self.datadir = os.path.abspath(config.dataset_directory)

        # saving directory
        self.save_dir = os.path.join(
            "run", config.dataset, f"bino_interaction_{config.binocular_interaction}"
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # experiment directory
        runs = natsorted(glob.glob(os.path.join(self.save_dir, "experiment_*")))
        if config.load_state:
            run_id = config.experiment_id
        else:
            run_id = int(runs[-1].split("_")[-1]) + 1 if runs else 0
        self.experiment_dir = os.path.join(
            self.save_dir,
            f"experiment_{run_id}",
        )
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # save config file
        if not config.load_state:
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
                f"{self.experiment_dir}/{config.resume}",
                map_location=self.device,
                weights_only=True,
            )
            self._resume_checkpoint = checkpoint
            pretrained_dict = checkpoint["state_dict"]

            # fix the keys of the state dictionary
            unwanted_prefix = "_orig_mod."
            for k, v in list(pretrained_dict.items()):
                if k.startswith(unwanted_prefix):
                    pretrained_dict[k[len(unwanted_prefix) :]] = pretrained_dict.pop(k)
            self.model.load_state_dict(pretrained_dict)
        self.model.to(self.device)
        # compile model
        if config.compile_mode is not None:
            self.model = torch.compile(
                self.model, mode=config.compile_mode
            )  # use compile_mode = "default" for layer analysis
        self.model.to(self.device)

        # if self.train_or_eval_mode == "train":
        # self.model.train()  # training mode
        print(
            f"GCNet was successfully loaded to {self.device}\n"
            + f"Binocular interaction: {config.binocular_interaction}\n"
            + f"Seed: {config.seed}\n"
            + f"Compile mode: {config.compile_mode}\n"
            + f"Experiment dir: {self.experiment_dir}\n"
            + f"Dataset: {config.dataset}\n"
            + f"Batch size: {config.batch_size}\n"
            + f"Number of epochs: {config.epochs} epochs"
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

        loaders = (
            make_loader(dataset_train, self.config, training=True),
            make_loader(dataset_validation, self.config, seed_offset=1),
            make_loader(dataset_test, self.config, seed_offset=2),
        )
        manifest = {
            split: ds.left_data
            for split, ds in zip(
                ("train", "validation", "test"),
                (dataset_train, dataset_validation, dataset_test),
            )
        }
        with open(Path(self.experiment_dir) / "split_manifest.json", "w") as stream:
            json.dump(manifest, stream, indent=2)
        return loaders

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
            "state_dict": getattr(self.model, "_orig_mod", self.model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": self._scaler.state_dict(),
            "history": self.history,
            "best_loss": self.best_loss,
            "epoch_complete": not best,
            "config": self.config.to_dict(),
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
        use_fused = fused_available and torch.device(self.device).type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )

        return optimizer

    def get_lr(self, it, max_steps):
        if max_steps <= 0 or it < 0:
            raise ValueError(
                "Learning-rate schedule needs positive max_steps and nonnegative step"
            )
        if it >= max_steps - 1:
            return self.config.min_lr
        warmup = min(max(0, self.config.warmup_steps or 0), max(0, max_steps - 2))
        if it < warmup:
            return self.config.max_lr * (it + 1) / warmup
        ratio = min(1.0, max(0.0, (it - warmup) / max(1, max_steps - 1 - warmup)))
        return self.config.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (
            self.config.max_lr - self.config.min_lr
        )

    def _autocast(self):
        kind = torch.device(self.device).type
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.config.amp_dtype]
        return torch.autocast(
            kind, dtype=dtype, enabled=kind == "cuda" and dtype != torch.float32
        )

    def _to_device(self, batch):
        return NestedTensor(
            **{
                key: batch[key].to(self.device, non_blocking=True)
                for key in ("left", "right", "disp", "ref")
            }
        )

    def _evaluate_batches(self, loader, criterion):
        total = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device)
        pixels = 0
        previous_mode = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in batches_for_evaluation(loader, self.config.eval_iter):
                    inputs = self._to_device(batch)
                    with self._autocast():
                        prediction = self.model(inputs)
                    loss = criterion(prediction.float(), inputs.disp.float())
                    n = inputs.disp.numel()
                    total += loss * n
                    correct += (
                        (prediction.float() - inputs.disp).abs()
                        < self.config.px_error_threshold
                    ).sum()
                    pixels += n
        finally:
            self.model.train(previous_mode)
        result = total / pixels
        if not torch.isfinite(result):
            raise ValueError("Nonfinite evaluation loss; inspect disparity targets")
        return result.item(), (correct / pixels).item()

    def train(self, train_loader, val_loader):
        return self.train_v2(train_loader, val_loader)

    def train_v2(self, train_loader, val_loader):
        config = self.config
        if not len(train_loader) or config.epochs <= 0:
            raise ValueError("Training requires a nonempty loader and positive epochs")
        if config.log_interval <= 0 or config.eval_interval <= 0:
            raise ValueError("Logging and evaluation intervals must be positive")
        if config.clip_max_norm < 0:
            raise ValueError("clip_max_norm must be nonnegative")
        criteria = {"smooth_l1": nn.SmoothL1Loss, "l1": nn.L1Loss}
        if config.loss not in criteria:
            raise ValueError(f"Unknown loss: {config.loss}")
        criterion = criteria[config.loss]()
        optimizer = self.configure_optimizers(config.weight_decay, config.lr)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                torch.device(self.device).type == "cuda"
                and config.amp_dtype == "float16"
            ),
        )
        train_eval = make_train_eval_loader(train_loader, config)
        self.history = {
            name: []
            for name in (
                "losses_train",
                "losses_val",
                "accs_train",
                "accs_val",
                "eval_steps",
            )
        }
        self.best_loss = float("inf")
        start_epoch = config.start_epoch
        checkpoint = getattr(self, "_resume_checkpoint", None)
        if checkpoint is not None:
            if not checkpoint.get("epoch_complete", False):
                raise ValueError(
                    "Training resume requires an epoch-complete checkpoint from the new trainer; best/legacy checkpoints remain usable for inference"
                )
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            self.history = checkpoint["history"]
            self.best_loss = checkpoint["best_loss"]
            start_epoch = checkpoint["epoch"] + 1
        if not 0 <= start_epoch < config.epochs:
            raise ValueError("start_epoch must be within the training epoch range")
        total_steps = config.epochs * len(train_loader)
        self._scaler = scaler
        for epoch in range(start_epoch, config.epochs):
            # Epoch-specific seeds allow reproducible epoch-boundary resume,
            # including persistent workers (dataset RNG receives the epoch).
            epoch_seed = config.seed + epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed % 2**32)
            torch.manual_seed(epoch_seed)
            if train_loader.generator is not None:
                train_loader.generator.manual_seed(epoch_seed)
            sampler_generator = getattr(train_loader.sampler, "generator", None)
            if sampler_generator is not None:
                sampler_generator.manual_seed(epoch_seed)
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            self.model.train()
            progress = tqdm(train_loader)
            for idx, batch in enumerate(progress):
                step = epoch * len(train_loader) + idx
                inputs = self._to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    prediction = self.model(inputs)
                loss = criterion(prediction.float(), inputs.disp.float())
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite training loss at step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.clip_max_norm
                    )
                for group in optimizer.param_groups:
                    group["lr"] = self.get_lr(step, total_steps)
                scaler.step(optimizer)
                scaler.update()
                if step % config.log_interval == 0:
                    progress.set_description(
                        f"step {step}/{total_steps} loss {loss.item():.4f}"
                    )
                # Save actual optimizer-update count, including the final update.
                if step % config.eval_interval == 0 or step == total_steps - 1:
                    tr_loss, tr_acc = self._evaluate_batches(train_eval, criterion)
                    va_loss, va_acc = self._evaluate_batches(val_loader, criterion)
                    for name, value in zip(
                        self.history, (tr_loss, va_loss, tr_acc, va_acc, step + 1)
                    ):
                        self.history[name].append(value)
                    self._save_history()
                    if va_loss < self.best_loss:
                        self.best_loss = va_loss
                        self.save_checkpoint(epoch, step, optimizer, best=True)
            self.save_checkpoint(epoch, step, optimizer)
        return self.history

    def _save_history(self):
        for name, values in self.history.items():
            np.save(Path(self.experiment_dir) / f"{name}.npy", values)

    def plotLine_learning_curve(self, save_flag=False):
        root = Path(self.experiment_dir)
        losses = [np.load(root / f"losses_{split}.npy") for split in ("train", "val")]
        accs = [np.load(root / f"accs_{split}.npy") for split in ("train", "val")]
        path = root / "eval_steps.npy"
        steps = (
            np.load(path)
            if path.exists()
            else np.arange(len(losses[0])) * self.config.eval_interval
        )

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (16, 5)
        n_row = 1
        n_col = 2

        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=figsize, sharex=True)

        for values, label in zip(losses, ("Train", "Validation")):
            axes[0].plot(steps, values, linewidth=2, label=label)
        for values, label in zip(accs, ("Train", "Validation")):
            axes[1].plot(steps, values, linewidth=2, label=label)

        y_low = 0
        y_up = 31
        axes[0].set_ylabel(self.config.loss)
        axes[0].set_ylim(y_low, y_up)

        y_low = 0.5
        y_up = 1.05
        axes[1].set_ylabel(f"Accuracy (< {self.config.px_error_threshold} px)")
        axes[1].set_ylim(y_low, y_up)
        for ax in axes:
            ax.set_xlabel("Optimizer updates")
            ax.legend()
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.25)

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
            fig.savefig(root / "learning_curve.pdf", bbox_inches="tight")
            plt.close(fig)
        return fig

    def plot_learning_rate(self, train_loader, save_flag):
        # plot learning rate
        max_steps_lr = self.config.epochs * len(train_loader)
        max_steps = self.config.epochs * len(train_loader)
        lr = np.empty(max_steps, dtype=np.float32)
        for i in range(max_steps):
            lr[i] = self.get_lr(i, max_steps_lr)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

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
            ref.to(self.config.device),
        )
        # inference
        self.model.eval()
        with torch.no_grad(), self._autocast():
            disp_pred = self.model(input_data)

        # visualize output
        # normalize to (0, 255), for visualization
        img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
        img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

        fig, axes = plt.subplots(
            nrows=len(left), ncols=4, figsize=(12, 5), squeeze=False
        )
        for i in range(len(left)):
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
