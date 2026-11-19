# %%
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import numpy as np
from contextlib import contextmanager
from tqdm import tqdm
from jaxtyping import Float, Int

import torch
from torch import nn, Tensor

import matplotlib.pyplot as plt
import seaborn as sns

from config.config_bi import ConfigBI
from engine.engine_bnn import EngineBNN
from engine.engine_gcnet_lr import EngineGCNet
from utilities.misc import NestedTensor

# %%


class BinocularIntegration:

    def __init__(self, engine: EngineBNN | EngineGCNet, config_bi: ConfigBI):

        self.engine = engine
        self.config = self.engine.config
        self.config_bi = config_bi

        self.model_name = engine.model_name
        self.interaction = engine.interaction
        self.model = getattr(
            self.engine.model, "_orig_mod", self.engine.model
        )  # remove "_orig_mod" phrases
        self.device = next(self.model.parameters()).device
        self.model.eval()

        # update epoch_iter_to load
        self.engine.update_epoch_iter_to_load()

        # update target layers
        self.update_target_layers()

        # set up output directories
        self.make_dirs()

        self.target_layer_indices = config_bi.target_layer_indices

        self.handles = []
        self.conv_types = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)

        # find target layers that are matched with layer_indices
        self.target_layers = self.find_target_layers(self.target_layer_indices)

        # print target layers
        self.print_target_layers()

        # allocate buffer for storing activations
        # self._buffer_mono: dict[str, Tensor] = {}
        self._buffer: dict[str, Tensor] = {}

    def update_target_layers(self):

        if self.model_name == "BNN":
            self.config_bi.target_layer_indices = list(range(3, 5))
        else:  # gcnet variants
            self.config_bi.target_layer_indices = list(range(19, 38))

    def make_dirs(self):

        # binocular integration directory
        self.bi_dir = os.path.join(
            self.engine.experiment_dir,
            f"bino_integration_iter_{self.config.iter_to_load}",
        )
        if not os.path.exists(self.bi_dir):
            os.makedirs(self.bi_dir)

    def find_target_layers(self, layer_indices: list) -> dict[str, nn.Module]:
        """
        find target layers in DNN whose indices match layer_indices

        Args:
            layer_indices (list): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            dict[str, nn.Module]: _description_
        """

        if not layer_indices:
            raise ValueError(
                "layer_indices should not be empty.\n"
                + "Please select at least one numbered layer [GC-Net: 19-37], BNN: [3, 3]."
            )

        target_layers = {}
        found_layers = set()
        for name, module in self.model.named_modules():

            # find layer indices
            ids = [
                int(m.group(1))
                for part in name.split(".")
                if (m := re.fullmatch(r"layer(\d+)[a-z]*", part))
            ]
            matched_layers = [n for n in ids if n in layer_indices]
            if matched_layers and isinstance(module, self.conv_types):
                target_layers[name] = module
                found_layers.update(matched_layers)

        missing_layers = set(layer_indices) - found_layers
        if missing_layers:
            raise ValueError(
                f"No convolution layer for layer_indices {sorted(missing_layers)}.\n"
                "BNN: [3-4], GC-Net: [19-37]"
            )

        return target_layers

    def hook_layers(self):
        """
        hook many layers
        """

        # prevent duplicate hooks
        self.close()

        for layer_name, module in self.target_layers.items():

            handle = module.register_forward_hook(self._hook_layer(layer_name))
            self.handles.append(handle)

    def _hook_layer(self, layer_name: str):
        """
        hook a single layer
        """

        def hook(module, args, output):
            if layer_name in self._buffer:
                raise RuntimeError(f"{layer_name} was executed more than once.")

            self._buffer[layer_name] = output.detach().clone()

        return hook

    def close(self):

        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    @contextmanager
    def _hooks_registered(self):
        """
        hooks are active only inside the "with" block
        """

        self.hook_layers()
        try:
            yield
        finally:
            self.close()
            self._buffer = {}

    def print_target_layers(self) -> None:

        print("==============================================================")
        print(f"List of target layers in {self.model_name}-{self.interaction}")
        print("==============================================================")
        for layer_name, module in self.target_layers.items():
            print(
                f"{layer_name} | type: {type(module).__name__} | out_channels={module.out_channels}"
            )
        print("==============================================================")

    def make_gray_image(
        self,
        image_shape: tuple[int, int, int, int],
        baseline: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> Float[Tensor, "B C H W"]:
        """
        Convert raw gray to normalized RGB; image shape is [1, 3, H, W].
        """

        if len(image_shape) != 4 or image_shape[1] != 3:
            raise ValueError("Expected floating RGB images [B, 3, H, W].")
        if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
            raise ValueError(
                "mean/std must have three entries and std must be positive."
            )

        values = [(baseline - m) / s for m, s in zip(mean, std)]
        img_dummy = torch.randn(image_shape, dtype=torch.float32, device=self.device)
        gray = img_dummy.new_tensor(values).view(1, 3, 1, 1)

        return gray.expand_as(img_dummy).contiguous()

    def replace_inputs(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"], closed_eye: str
    ) -> NestedTensor:

        if closed_eye not in ["left", "right", "both"]:
            raise TypeError('closed_eye must be "left" or "right" or "both"')

        if inputs.left.shape != inputs.right.shape or gray.shape != inputs.left.shape:
            raise ValueError("Left, right, and gray shapes must match.")

        if gray.device != inputs.left.device or gray.dtype != inputs.left.dtype:
            raise ValueError(
                "Gray and input images must have matching device and dtype."
            )

        return NestedTensor(
            left=gray if closed_eye in ("left", "both") else inputs.left,
            right=gray if closed_eye in ("right", "both") else inputs.right,
            disp=inputs.disp,
            ref=inputs.ref,
        )

    def _forward(self, inputs: NestedTensor) -> dict[str, Float[Tensor, "B C D H W"]]:
        """
        one forward pass.
        returns the hooked outputs, still on GPU
        """

        if inputs.ref is None:
            raise ValueError("inputs.ref must be specified +1 or -1 for each sample")

        self._buffer = {}
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=True
        ):
            self.model(inputs)

        missing_layer = set(self.target_layers) - set(self._buffer)
        if missing_layer:
            raise RuntimeError(
                f"Selected layers did not execute: {sorted(missing_layer)}"
            )

        activations = self._buffer
        self._buffer = {}
        return activations

    def _mono_sum_resp(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ) -> dict[str, Float[Tensor, "B C D H W"]]:
        """
        left + right - baseline, accumulated in fp32 on the GPU.

        Only one running total plus one in-flight activation set is resident,
        instead of three separate sets.
        """

        # gray = self.make_gray_image(inputs.left)

        # compute convolutional layer outputs when left eye is closed
        # (left_input = gray images)
        mono_right = self._forward(self.replace_inputs(inputs, gray, closed_eye="left"))

        total = {layer_name: act.float() for layer_name, act in mono_right.items()}

        del mono_right

        # + mono_left_resp (right eye receives gray); - baseline (both eyes receive gray)
        for closed_eye, sign in (("right", +1), ("both", -1)):
            acts = self._forward(self.replace_inputs(inputs, gray, closed_eye))

            for layer_name, act in acts.items():
                total[layer_name].add_(act, alpha=sign)

        return total

    def _bino_resp(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ) -> dict[str, Float[Tensor, "B C D H W"]]:
        """
        bino_activation - baseline
        """

        # gray = self.make_gray_image(inputs.left)
        baseline = self._forward(self.replace_inputs(inputs, gray, closed_eye="both"))
        activations_raw = self._forward(inputs)
        bino_resp = {}
        for layer_name, act in activations_raw.items():
            bino_resp[layer_name] = act - baseline[layer_name]

        return bino_resp

    def _to_cpu(
        self, activations: dict[str, Tensor]
    ) -> dict[str, Float[Tensor, "B C D H W"]]:

        host = {}
        for layer_name, act in activations.items():

            act = act.to(dtype=torch.float32)

            host[layer_name] = torch.empty(act.shape, dtype=act.dtype, pin_memory=True)
            host[layer_name].copy_(act, non_blocking=True)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return host

    @torch.inference_mode()
    def get_mono_sum_resp(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ) -> dict[str, Float[Tensor, "B C D H W"]]:

        # gray = self.make_gray_image(inputs.left)

        with self._hooks_registered():
            return self._to_cpu(self._mono_sum_resp(inputs, gray))

    @torch.inference_mode()
    def get_activations(
        self, inputs: NestedTensor
    ) -> dict[str, Float[Tensor, "B C D H W"]]:

        with self._hooks_registered():
            return self._to_cpu(self._forward(inputs))

    @torch.inference_mode()
    def compute_mono_and_bino_resp(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ):

        with self._hooks_registered():
            # each GPU result is freed as soon as it has been copied to the host

            # compute the sum of monocular left and right
            mono_sum_resp = self._mono_sum_resp(inputs, gray)

            # compute binocular response
            bino_resp = self._bino_resp(inputs, gray)

        # return mono_left, mono_right, baseline, mono_sum_resp, bino_resp
        return mono_sum_resp, bino_resp

    @torch.inference_mode()
    def get_activation_shape(self, inputs: NestedTensor) -> dict[str, tuple[int, ...]]:
        """
        inspect the shape of target_layers
        """

        out = self.get_activations(inputs)

        # record output shape
        out_shapes = {}
        for layer_name, act in out.items():
            out_shapes[layer_name] = tuple(act.shape)

        return out_shapes

    def _integration_ratio(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ) -> dict[str, Int[Tensor, "C D H W"]]:

        integration_ratio = {}
        tol = 1e-6
        with self._hooks_registered():

            # get bino_resp
            bino_resp = self._bino_resp(inputs, gray)

            # get mono_sum_resp
            mono_sum_resp = self._mono_sum_resp(inputs, gray)

            for layer_name, bino_act in bino_resp.items():

                delta = bino_act - mono_sum_resp[layer_name]
                ratio = delta.abs() / (
                    bino_act.abs() + mono_sum_resp[layer_name].abs() + tol
                )
                # ratio = bino_act / (mono_sum_resp[layer_name] + tol)

                # sum ratio across batch, for later averaging across dataloader
                integration_ratio[layer_name] = ratio.sum(dim=0, dtype=torch.float32)

        return integration_ratio

    @torch.inference_mode()
    def compute_integration_ratio(self, test_loader):
        """
        Estimate probability P(bino_resp > mono_sum_resp) for each neuron.
        """

        # allocate buffer
        integration_ratio_total = None
        sample_total = 0
        for inputs in tqdm(test_loader):

            # move inputs to gpu
            inputs = self.engine._to_device(inputs)

            # make gray image
            gray = self.make_gray_image(inputs.left.shape)

            # compute sample size
            sample_total += inputs.left.shape[0]

            # count the number of coupled neurons in the current batch
            current_batch_ratio = self._integration_ratio(inputs, gray)

            if integration_ratio_total == None:
                integration_ratio_total = current_batch_ratio
            else:
                for layer_name, ratio in current_batch_ratio.items():

                    if integration_ratio_total[layer_name].shape != ratio.shape:
                        raise ValueError(f"Activation shape changed for {layer_name}.")

                    integration_ratio_total[layer_name].add_(ratio)

        if sample_total == 0:
            raise ValueError("test_loader contains no samples.")

        # compute ratio
        integration_ratio = {}
        for layer_name, ratio in integration_ratio_total.items():
            integration_ratio[layer_name] = (
                ratio.to(dtype=torch.float32) / sample_total
            ).to(device="cpu")

        # save
        filename = f"{self.bi_dir}/integration_ratio.npy"
        np.save(filename, integration_ratio)

        return integration_ratio

    def _integration_count(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ) -> dict[str, Int[Tensor, "C D H W"]]:

        integration_count = {}
        tol = 1e-6
        with self._hooks_registered():

            # get bino_resp
            bino_resp = self._bino_resp(inputs, gray)

            # get mono_sum_resp
            mono_sum_resp = self._mono_sum_resp(inputs, gray)

            for layer_name, bino_act in bino_resp.items():

                delta = bino_act - mono_sum_resp[layer_name]
                ratio = delta.abs() / (
                    bino_act.abs() + mono_sum_resp[layer_name].abs() + tol
                )
                # ratio = bino_act / (mono_sum_resp[layer_name] + tol)

                # sum ratio across batch, for later averaging across dataloader
                integration_count[layer_name] = torch.sum(
                    ratio > 0.5, dim=0, dtype=torch.float32
                )

        return integration_count

    @torch.inference_mode()
    def compute_integration_count(self, test_loader):
        """
        Estimate probability P(bino_resp > mono_sum_resp) for each neuron.
        """

        # allocate buffer
        integration_count_total = None
        sample_total = 0
        for inputs in tqdm(test_loader):

            # move inputs to gpu
            inputs = self.engine._to_device(inputs)

            # make gray image
            gray = self.make_gray_image(inputs.left.shape)

            # compute sample size
            sample_total += inputs.left.shape[0]

            # count the number of coupled neurons in the current batch
            current_batch_count = self._integration_count(inputs, gray)

            if integration_count_total == None:
                integration_count_total = current_batch_count
            else:
                for layer_name, count in current_batch_count.items():

                    if integration_count_total[layer_name].shape != count.shape:
                        raise ValueError(f"Activation shape changed for {layer_name}.")

                    integration_count_total[layer_name].add_(count)

        if sample_total == 0:
            raise ValueError("test_loader contains no samples.")

        # compute ratio
        integration_count = {}
        for layer_name, count in integration_count_total.items():
            integration_count[layer_name] = (
                count.to(dtype=torch.float32) / sample_total
            ).to(device="cpu")

        # save
        filename = f"{self.bi_dir}/integration_count.npy"
        np.save(filename, integration_count)

        return integration_count

    def plotLine_integration_ratio(self, save_flag: bool = True):

        # load data
        filename = f"{self.bi_dir}/integration_ratio.npy"
        integration_ratio = np.load(filename, allow_pickle=True).item()

        integration_mean = np.empty(len(self.target_layers), dtype=np.float32)
        integration_std = np.empty_like(integration_mean)
        for i, (layer_name, prob) in enumerate(integration_ratio.items()):

            integration_mean[i] = prob.mean()
            integration_std[i] = prob.std()

        # plot line
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (12, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"{self.model_name} - {self.interaction}: Binocular integration ratio across layers\n",
            ha="center",
        )
        fig.text(-0.04, 0.5, "Integration ratio", va="center", rotation=90)
        fig.text(0.5, -0.04, "Layer", ha="center")
        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#0072B2", "#D62728", "#009E73", "#9467BD"]

        if self.model_name == "BNN":
            x = np.arange(3, 5)
        else:  # GC-Net variants
            x = np.arange(19, 38)
        axes.plot(
            x,
            integration_mean,
            linewidth=2,
            color=colors[0],
            # label=label,
            # linestyle=linestyle,
        )
        axes.plot(x, integration_mean, "o", markersize=12, color=colors[0])
        axes.fill_between(
            x,
            integration_mean - integration_std,
            integration_mean + integration_std,
            color=colors[0],
            alpha=0.2,
        )

        if self.model_name == "BNN":
            x_low = 3
            x_up = 5
            x_step = 1
        else:  # GC-Net variants
            x_low = 19
            x_up = 38
            x_step = 2
        y_low = 0.0
        y_up = 1.05
        y_step = 0.2

        # plot chance level
        axes.hlines(
            [0.5], xmin=x_low - 1, xmax=x_up, colors="red", linestyles="--", linewidth=3
        )

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 1, x_up)
        axes.set_ylim(y_low, y_up)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save figure
        if save_flag == 1:
            fig.savefig(
                f"{self.bi_dir}/integration_ratio_{self.engine.interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_integration_count(self, save_flag: bool = True):

        # load data
        filename = f"{self.bi_dir}/integration_count.npy"
        integration_count = np.load(filename, allow_pickle=True).item()

        integration_mean = np.empty(len(self.target_layers), dtype=np.float32)
        integration_std = np.empty_like(integration_mean)
        for i, (layer_name, prob) in enumerate(integration_count.items()):

            integration_mean[i] = prob.mean()
            integration_std[i] = prob.std()

        # plot line
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (12, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"{self.model_name} - {self.interaction}: Binocular integration count across layers\n",
            ha="center",
        )
        fig.text(-0.04, 0.5, "Integration count", va="center", rotation=90)
        fig.text(0.5, -0.04, "Layer", ha="center")
        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#0072B2", "#D62728", "#009E73", "#9467BD"]

        if self.model_name == "BNN":
            x = np.arange(3, 5)
        else:  # GC-Net variants
            x = np.arange(19, 38)
        axes.plot(
            x,
            integration_mean,
            linewidth=2,
            color=colors[0],
            # label=label,
            # linestyle=linestyle,
        )
        axes.plot(x, integration_mean, "o", markersize=12, color=colors[0])
        axes.fill_between(
            x,
            integration_mean - integration_std,
            integration_mean + integration_std,
            color=colors[0],
            alpha=0.2,
        )

        if self.model_name == "BNN":
            x_low = 3
            x_up = 5
            x_step = 1
        else:  # GC-Net variants
            x_low = 19
            x_up = 38
            x_step = 2
        y_low = 0.0
        y_up = 1.05
        y_step = 0.2

        # plot chance level
        axes.hlines(
            [0.5], xmin=x_low - 1, xmax=x_up, colors="red", linestyles="--", linewidth=3
        )

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 1, x_up)
        axes.set_ylim(y_low, y_up)

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save figure
        if save_flag == 1:
            fig.savefig(
                f"{self.bi_dir}/integration_count_{self.engine.interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )
