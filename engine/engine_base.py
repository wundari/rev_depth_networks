# %%
# working dir: /home/wundari/rev_depth_networks

# %% load necessary modules
import torch
import torch.nn as nn
from torch.utils.data import default_collate

import numpy as np
import os
import glob
import sys
import inspect
import math
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from natsort import natsorted
from pathlib import Path

from dataset.scene_flow import SceneFlowFlyingThingsDataset
from dataset.scene_flow import SceneFlowMonkaaDataset
from dataset.loading import make_loader, make_train_eval_loader, batches_for_evaluation
from config.config_bnn import ConfigBNN
from config.config_gcnet import ConfigGCNet
from utilities.misc import NestedTensor


# %%
class EngineBase:

    def __init__(self, config: ConfigBNN | ConfigGCNet) -> None:

        self.config = config
        self.model_name = config.model_name
        self.device = config.device
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = config.deterministic
        torch.backends.cudnn.benchmark = not config.deterministic
        if config.amp_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unknown amp_dtype")
        if torch.device(self.device).type == "cuda" and config.amp_dtype == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This GPU does not support BF16; choose float16")
        self.h = config.img_height
        self.w = config.img_width

        # dataset directory
        # parent_folder = "/media/wundari/S990Pro2_4TB"
        parent_folder = os.path.abspath(f"{os.curdir}/..")
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
            f"{self.model_name}/run",
            config.dataset,
            f"bino_interaction_{config.binocular_interaction}",
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
        self.save_config()

        # pred_images directory
        self.pred_images_dir = os.path.join(self.experiment_dir, "pred_images")
        if not os.path.exists(self.pred_images_dir):
            os.makedirs(self.pred_images_dir)

        # load model
        self.model = self._build_model(config)
        # load pre-trained BNN, if provided
        if config.load_state:
            print(
                f"Load pretrained {self.model_name}:\n"
                + f"Binocular interaction: {config.binocular_interaction}\n"
                + f"Experiment id: {config.experiment_id}\n"
                + f"Model: {config.resume}"
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
        # compile model
        if config.compile_mode is not None:
            self.model = torch.compile(
                self.model, mode=config.compile_mode
            )  # use compile_mode = "default" for layer analysis
        self.model.to(self.device)

        # if self.train_or_eval_mode == "train":
        # self.model.train()  # training mode
        print(
            f"{self.model_name} was successfully loaded to {self.device}\n"
            + f"Binocular interaction: {config.binocular_interaction}\n"
            + f"Seed: {config.seed}\n"
            + f"Compile mode: {config.compile_mode}\n"
            + f"Experiment dir: {self.experiment_dir}\n"
            + f"Dataset: {config.dataset}\n"
            + f"Batch size train: {config.batch_size}\n"
            + f"Batch size validation: {config.batch_size_val}\n"
            + f"Number of epochs: {config.epochs} epochs"
        )

    def _build_model(self, config: ConfigBNN | ConfigGCNet) -> nn.Module:
        raise NotImplementedError

    def save_config(self):
        config_dict = self.config.to_dict()
        with open(f"{self.experiment_dir}/config.json", "w") as f:
            json.dump(config_dict, f)

    # ------------------------------------------------------------------ #
    # dataset
    # ------------------------------------------------------------------ #
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

        # record which files went into each split
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
        """
        Sanity-check one batch: plot a random patch and its disparity.
        """

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

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # optimizer / LR schedule / autocast
    # ------------------------------------------------------------------ #
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
        """
        Linear warmup, then cosine decay to min_lr over the full schedule.
        """

        if max_steps <= 0 or it < 0:
            raise ValueError(
                "Learning-rate schedule needs positive max_steps and nonnegative step"
            )

        # 1 linear warmup for warmup_iters steps
        if it < self.config.warmup_steps:
            return self.config.max_lr * (it + 1) / self.config.warmup_steps

        # 2 if it > lr_decay_iters, return min learning rate
        if it > max_steps - 1:
            return self.config.min_lr

        # 3 in between, use cosine decay down to min learning rate
        decay_ratio = min(
            1.0,
            max(
                0.0,
                (it - self.config.warmup_steps)
                / max(1.0, (max_steps - self.config.warmup_steps - 1)),
            ),
        )

        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)

    def _to_device(self, batch):
        return NestedTensor(
            **{
                key: batch[key].to(self.device, non_blocking=True)
                for key in ("left", "right", "disp", "ref")
            }
        )

    def _save_history(self):
        for name, values in self.history.items():
            np.save(Path(self.experiment_dir) / f"{name}.npy", values)

    @torch.no_grad()
    def _save_prediction_snapshot(self, loader):
        """
        Periodically save a qualitative left/right/pred/gt figure.
        """
        previous_mode = self.model.training
        self.model.eval()
        try:

            # randomly sample a batch from the loader
            dataset = loader.dataset
            n_show = loader.batch_size
            idx = random.sample(range(len(dataset)), n_show)
            batch = default_collate([dataset[i] for i in idx])

            inputs = self._to_device(batch)

            with torch.autocast(
                device_type=torch.device(self.device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(self.device).type == "cuda",
            ):
                prediction = self.model(inputs)

            left = inputs.left.detach().float().cpu()
            right = inputs.right.detach().float().cpu()
            disp = inputs.disp.detach().float().cpu()
            prediction = prediction.detach().float().cpu()

            img_left = ((left / left.amax()) * 128 + 127).to(torch.uint8)
            img_right = ((right / right.amax()) * 128 + 127).to(torch.uint8)

            n = left.shape[0]
            vmin, vmax = -50, 50
            fig, axes = plt.subplots(
                nrows=n, ncols=4, figsize=(12, 3 * n), squeeze=False
            )
            for i in range(n):
                axes[i, 0].imshow(img_left[i].permute(1, 2, 0))
                axes[i, 0].set_title("Left")
                axes[i, 1].imshow(img_right[i].permute(1, 2, 0))
                axes[i, 1].set_title("Right")
                axes[i, 2].imshow(prediction[i], cmap="jet", vmin=vmin, vmax=vmax)
                axes[i, 2].set_title("Pred. disparity")
                im = axes[i, 3].imshow(disp[i], cmap="jet", vmin=vmin, vmax=vmax)
                axes[i, 3].set_title("Ground truth")
                for ax in axes[i]:
                    ax.axis("off")

                l_ax, b_ax, w_ax, h_ax = axes[i, 3].get_position().bounds
                cax = fig.add_axes([l_ax + w_ax + 0.03, b_ax, 0.03, h_ax])
                fig.colorbar(im, cax=cax)

            fig.savefig(
                f"{self.pred_images_dir}/output_test.pdf",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)
        finally:
            self.model.train(previous_mode)

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_batches(self, loader, criterion):
        total_loss = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device)
        pixels = 0

        previous_mode = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                for batch in batches_for_evaluation(loader, self.config.n_iter_eval):
                    inputs = self._to_device(batch)
                    with torch.autocast(
                        device_type=torch.device(self.device).type,
                        dtype=torch.bfloat16,
                        enabled=torch.device(self.device).type == "cuda",
                    ):
                        prediction = self.model(inputs)

                    loss = criterion(inputs.disp.float(), prediction.float())
                    count = inputs.disp.numel()
                    total_loss += loss * count
                    correct += (
                        (prediction.float() - inputs.disp.float()).abs() < 3
                    ).sum()
                    pixels += count
        finally:
            self.model.train(previous_mode)

        loss_avg = total_loss / pixels
        if not torch.isfinite(loss_avg):
            raise ValueError("Nonfinite evaluation loss; inspect disparity targets")

        return loss_avg.item(), (correct / pixels).item()

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self, train_loader, val_loader, test_loader):

        if not len(train_loader) or self.config.epochs <= 0:
            raise ValueError("Training requires a nonempty loader and positive epochs")
        if self.config.log_interval <= 0 or self.config.eval_interval <= 0:
            raise ValueError("Logging and evaluation intervals must be positive")
        if self.config.clip_max_norm < 0:
            raise ValueError("clip_max_norm must be nonnegative")
        criteria = {"smooth_l1": nn.SmoothL1Loss, "l1": nn.L1Loss}
        if self.config.loss not in criteria:
            raise ValueError(f"Unknown loss: {self.config.loss}")

        criterion = criteria[self.config.loss]()

        optimizer = self.configure_optimizers(self.config.weight_decay, self.config.lr)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=torch.device(self.device).type == "cuda"
        )

        # create dataloader for evaluation on training dataset
        train_eval_loader = make_train_eval_loader(train_loader, self.config)
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
        start_epoch = self.config.start_epoch
        checkpoint = getattr(self, "_resume_checkpoint", None)
        if checkpoint is not None:
            if not checkpoint.get("epoch_complete", False):
                raise ValueError(
                    "Training resume requires an epoch-complete checkpoint from "
                    "the new trainer; best/legacy checkpoints remain usable for inference"
                )
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            self.history = checkpoint["history"]
            self.best_loss = checkpoint["best_loss"]
            start_epoch = checkpoint["epoch"] + 1

        if not 0 <= start_epoch < self.config.epochs:
            raise ValueError("start_epoch must be within the training epoch range")

        total_steps = self.config.epochs * len(train_loader)
        self._scaler = scaler

        for epoch in range(start_epoch, self.config.epochs):

            # Epoch-specific seeds allow reproducible epoch-boundary resume,
            # including persistent workers (dataset RNG receives the epoch).
            epoch_seed = self.config.seed + epoch
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
            # for i in range(iter_num, len(train_loader)):
            for idx, inputs in enumerate(progress):

                step = epoch * len(train_loader) + idx
                inputs = self._to_device(inputs)

                # zero the gradients
                optimizer.zero_grad(set_to_none=True)

                # forward pass
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                    enabled=torch.device(self.device).type == "cuda",
                ):
                    disp_pred = self.model(inputs)

                # compute loss
                loss = criterion(inputs.disp.float(), disp_pred.float())
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite training loss at step {step}")

                # compute 3-pix acc
                diff = torch.abs(disp_pred.float() - inputs.disp.float())
                acc_train = (diff < 3).float().mean()

                # terminate training if exploded
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    sys.exit(1)

                # backprop
                scaler.scale(loss).backward()

                # clip norm
                # if max_norm > 0:
                scaler.unscale_(optimizer)
                if self.config.clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_max_norm
                    )

                # update learning rate
                lr = self.get_lr(step, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # step optimizer
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                # logging
                if step % self.config.log_interval == 0:
                    progress.set_description(
                        f"step {step}/{total_steps} | train loss: {loss_value:.4f} |"
                        f"train 3-pix acc: {acc_train.item():.4f} | lr: {lr:.4e}"
                    )

                # once in a while evaluate validation loss
                if step % self.config.eval_interval == 0 or step == total_steps - 1:

                    tr_loss, tr_acc = self._evaluate_batches(
                        train_eval_loader, criterion
                    )
                    val_loss, val_acc = self._evaluate_batches(val_loader, criterion)

                    # record performance
                    for name, value in zip(
                        self.history, (tr_loss, val_loss, tr_acc, val_acc, step + 1)
                    ):
                        self.history[name].append(value)

                    # safe performance history
                    self._save_history()

                    # save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(epoch, step, optimizer, best=True)

                    # save predicted disparity maps from test dataloader
                    if (
                        self.config.save_snapshot
                        and step % self.config.snapshot_interval == 0
                    ):
                        self._save_prediction_snapshot(test_loader)

            # save model each epoch
            self.save_checkpoint(epoch, step, optimizer)

        return self.history

    def plotLine_learning_curve(self, save_flag):

        losses_train = np.load(f"{self.experiment_dir}/losses_train.npy")
        losses_val = np.load(f"{self.experiment_dir}/losses_val.npy")
        accs_train = np.load(f"{self.experiment_dir}/accs_train.npy")
        accs_val = np.load(f"{self.experiment_dir}/accs_val.npy")
        steps = np.load(f"{self.experiment_dir}/eval_steps.npy") // 100

        assert len(losses_val) == len(losses_train)
        assert len(accs_val) == len(accs_train)

        # start plotting
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
            f"Step (averaged across {self.config.batch_size * self.config.n_iter_eval} trials, "
            + f"sampled every {self.config.eval_interval} steps)",
            ha="center",
        )

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.25)

        axes[0].plot(steps, losses_train, linewidth=2, label="Train")
        axes[0].plot(steps, losses_val, linewidth=2, label="Validation")
        axes[1].plot(steps, accs_train, linewidth=2, label="Train")
        axes[1].plot(steps, accs_val, linewidth=2, label="Validation")

        if self.model_name == "BNN":
            y_low = 10
            y_up = 51
        else:  # GC_Net
            y_low = 0
            y_up = 31
        y_step = 5

        axes[0].set_ylabel(self.config.loss)
        axes[0].set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[0].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[0].set_ylim(y_low, y_up)

        if self.model_name == "BNN":
            y_low = 0.0
            y_up = 0.61
        else:  # GC_Net
            y_low = 0.5
            y_up = 1.05
        y_step = 0.1

        axes[1].set_ylabel("3-pix acc")
        axes[1].set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[1].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes[1].set_ylim(y_low, y_up)

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
            plt.close(fig)

    def plot_learning_rate(self, train_loader, save_flag):
        # plot learning rate
        total_steps = self.config.epochs * len(train_loader)
        lr = np.empty(total_steps, dtype=np.float32)
        for i in range(total_steps):
            lr[i] = self.get_lr(i, total_steps)

        # start plotting
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        # plt.rcParams["font.family"] = "sans-serif"
        # plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

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
            plt.close(fig)

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
