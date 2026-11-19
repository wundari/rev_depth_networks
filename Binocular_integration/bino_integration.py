# %%
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import numpy as np
from contextlib import contextmanager
from tqdm import tqdm
from typing import Callable
from jaxtyping import Float

import torch
from torch import nn, Tensor

import matplotlib.pyplot as plt
import seaborn as sns

from config.config_bi import ConfigBI
from engine.engine_bnn import EngineBNN
from engine.engine_gcnet_lr import EngineGCNet
from utilities.misc import NestedTensor


# %%
class _StopForward(Exception):
    """Raised from the hook of the last target layer to skip the rest of the network."""


class BinocularIntegration:
    """
    For every target conv layer and every neuron (C, D, H, W) compute

        bino_resp     = A(L, R)       - A(gray, gray)
        mono_sum_resp = (A(gray, R)   - A(gray, gray))      # left eye closed
                        + (A(L, gray)   - A(gray, gray))      # right eye closed
        ratio         = |bino - mono_sum| / (|bino| + |mono_sum| + tol)

    and average `ratio` (and `ratio > thresh`) over the dataset.

    amp_dtype: None -> fp32 (convs still use TF32 on the 5090). torch.bfloat16 is
                faster but the quantisation error of bf16 outputs (~0.4 %) is
                comparable to the small differences this analysis looks at.
    early_exit: stop the forward pass as soon as the last hooked layer has run.
    """

    TOL = 1e-6

    def __init__(self, engine: EngineBNN | EngineGCNet, config_bi: ConfigBI):

        self.engine = engine
        self.config = self.engine.config
        self.config_bi = config_bi
        self.amp_dtype = config_bi.amp_dtype
        self.early_exit = config_bi.early_exit

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
        self.layer_index_of: dict[str, int] = {}  # conv name -> numbered layer
        self.target_layers = self.find_target_layers(self.target_layer_indices)

        # print target layers
        self.print_target_layers()

        # hook state (set per forward pass)
        self._sink: Callable[[str, Tensor], None] | None = None
        self._seen_layer: set[str] = set()
        self._gray_cache: dict[tuple, Tensor] = {}

    def _configure_backends(self):
        if self.device.type != "cuda":
            return
        torch.backends.cudnn.benchmark = True  # fixed shapes -> autotune once
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

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
            os.makedirs(self.bi_dir, exist_ok=True)

    def find_target_layers(self, layer_indices: list) -> dict[str, nn.Module]:
        """
        Find convolution layers whose "layer<N>" index is in layer_indices.

        Raises:
            ValueError: if layer_indices is empty or an index has no convolution.

        Returns:
            dict[str, nn.Module]: module name -> convolution module.
        """

        if not layer_indices:
            raise ValueError(
                "layer_indices should not be empty.\n"
                + "Please select at least one numbered layer [GC-Net: 19-37], BNN: [3, 3]."
            )

        target_layers = {}
        found_layers = set()
        for layer_name, module in self.model.named_modules():

            # find layer indices
            ids = [
                int(m.group(1))
                for part in layer_name.split(".")
                if (m := re.fullmatch(r"layer(\d+)[a-z]*", part))
            ]
            matched_layers = [n for n in ids if n in layer_indices]
            if matched_layers and isinstance(module, self.conv_types):
                target_layers[layer_name] = module
                self.layer_index_of[layer_name] = matched_layers[0]
                found_layers.update(matched_layers)

        missing_layers = set(layer_indices) - found_layers
        if missing_layers:
            raise ValueError(
                f"No convolution layer for layer_indices {sorted(missing_layers)}.\n"
                "BNN: [3-4], GC-Net: [19-37]"
            )

        return target_layers

    def print_target_layers(self) -> None:

        print("==============================================================")
        print(f"List of target layers in {self.model_name}-{self.interaction}")
        print("==============================================================")
        for layer_name, module in self.target_layers.items():
            print(
                f"{layer_name} | type: {type(module).__name__} | out_channels={module.out_channels}"
            )
        print("==============================================================")

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
        hook a single layer.
        The hook hands the raw conv output to the current sink *immediately*.
        The sink must consume it right away (copy / accumulate / reduce), because
        later in-place ops in the network may overwrite it.
        """

        # def hook(module, args, output):
        #     if layer_name in self._buffer:
        #         raise RuntimeError(f"{layer_name} was executed more than once.")

        #     self._buffer[layer_name] = output.detach().clone()

        def hook(module, args, output):
            if layer_name in self._seen_layer:
                raise RuntimeError(f"{layer_name} was executed more than once.")
            self._seen_layer.add(layer_name)

            self._sink(layer_name, output.detach())

            if self.early_exit and len(self._seen_layer) == len(self.target_layers):
                raise _StopForward

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
            self._sink = None
            self._seen_layer = set()

    def make_gray_image(
        self,
        image_shape: tuple[int, int, int, int],
        dtype: torch.dtype = torch.float32,
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
        gray = torch.tensor(values, dtype=dtype, device=self.device).view(1, 3, 1, 1)

        return gray.expand(image_shape).contiguous()

    def _get_gray(self, like: Tensor) -> Float[Tensor, "B C H W"]:
        """
        cached gray image with the same shape/dtype as `like`
        """

        key = (tuple(like.shape), like.dtype)
        if key not in self._gray_cache:
            self._gray_cache[key] = self.make_gray_image(like.shape, dtype=like.dtype)

        return self._gray_cache[key]

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

    def _run(self, inputs: NestedTensor, sink: Callable[[str, Tensor], None]) -> None:
        """
        one forward pass; every hooked conv output is passed to "sink"

        """

        if not self.handles:
            raise RuntimeError(
                "Hooks are not registered; use 'with self._hooks_registered()'."
            )

        if inputs.ref is None:
            raise ValueError("inputs.ref must be specified +1 or -1 for each sample")

        self._sink = sink
        self._seen_layer = set()

        try:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype or torch.bfloat16,
                enabled=True,
            ):
                self.model(inputs)

        except _StopForward:
            pass
        finally:
            self._sink = None

        missing_layer = set(self.target_layers) - set(self._seen_layer)
        if missing_layer:
            raise RuntimeError(
                f"Selected layers did not execute: {sorted(missing_layer)}"
            )

    def _forward(self, inputs: NestedTensor) -> dict[str, Float[Tensor, "B C D H W"]]:
        """
        one forward pass.
        returns the hooked outputs, still on GPU
        """

        out: dict[str, Tensor] = {}

        def sink(layer_name: str, output: Tensor):
            out[layer_name] = output.to(torch.float32, copy=True)

        self._run(inputs, sink)

        # if inputs.ref is None:
        #     raise ValueError("inputs.ref must be specified +1 or -1 for each sample")

        # self._buffer = {}
        # with torch.autocast(
        #     device_type=self.device.type, dtype=torch.bfloat16, enabled=True
        # ):
        #     self.model(inputs)

        # missing_layer = set(self.target_layers) - set(self._buffer)
        # if missing_layer:
        #     raise RuntimeError(
        #         f"Selected layers did not execute: {sorted(missing_layer)}"
        #     )

        # activations = self._buffer
        # self._buffer = {}
        return out

    def _baseline_and_mono_sum_resp(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ) -> tuple[
        dict[str, Float[Tensor, "B C D H W"]], dict[str, Float[Tensor, "B C D H W"]]
    ]:
        """
        baseline = activation(gray, gray)
        mono_left = activation(L, gray)
        mono_right = activation(gray, R)
        mono_sum_resp = (mono_left - baseline) + (mono_right - baseline)

        run 3 forward passes; only `baseline` and `mono_sum_resp` stay in memory.
        """

        baseline = self._forward(self.replace_inputs(inputs, gray, "both"))

        # allocate buffer for mono_sum_resp
        mono_sum_resp: dict[str, Tensor] = {}

        # define a "sink" function used in the hooking process
        def sink(layer_name: str, output: Tensor):
            if layer_name not in mono_sum_resp:
                mono_sum_resp[layer_name] = (
                    output.float() - baseline[layer_name]
                )  # new activation tensor
            else:
                mono_sum_resp[layer_name].add_(output).sub_(baseline[layer_name])

        for closed_eye in ("left", "right"):
            self._run(self.replace_inputs(inputs, gray, closed_eye), sink)

        # # compute convolutional layer outputs when left eye is closed
        # # (left_input = gray images)
        # mono_right = self._forward(self.replace_inputs(inputs, gray, closed_eye="left"))

        # total = {layer_name: act.float() for layer_name, act in mono_right.items()}

        # del mono_right

        # # + mono_left_resp (right eye receives gray); - baseline (both eyes receive gray)
        # for closed_eye, sign in (("right", +1), ("both", -1)):
        #     acts = self._forward(self.replace_inputs(inputs, gray, closed_eye))

        #     for layer_name, act in acts.items():
        #         total[layer_name].add_(act, alpha=sign)

        return baseline, mono_sum_resp

    def _bino_integration(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"], thresh: float
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        run 4 forward passes in total.
        The ratio is reduced over the batch inside the hook,
        so the full bino_resp / delta tensors never coexist for all layers.
        """

        # compute baseline and mono_sum_resp
        baseline, mono_sum_resp = self._baseline_and_mono_sum_resp(inputs, gray)

        # compute integration ratio and integration count
        integration_ratio: dict[str, Tensor] = {}
        integration_count: dict[str, Tensor] = {}

        def sink(layer_name: str, output: Tensor):
            mono_sum_act = mono_sum_resp[layer_name]
            bino_act = output.float() - baseline[layer_name]

            delta = bino_act - mono_sum_act
            den = bino_act.abs_().add_(mono_sum_act.abs()).add_(self.TOL)
            ratio = delta.abs_().div_(den)

            integration_ratio[layer_name] = ratio.sum(dim=0)
            integration_count[layer_name] = (ratio > thresh).sum(
                dim=0, dtype=torch.float32
            )

        self._run(inputs, sink)

        # # gray = self.make_gray_image(inputs.left)
        # baseline = self._forward(self.replace_inputs(inputs, gray, closed_eye="both"))
        # activations_raw = self._forward(inputs)
        # bino_resp = {}
        # for layer_name, act in activations_raw.items():
        #     bino_resp[layer_name] = act - baseline[layer_name]

        return integration_ratio, integration_count

    @staticmethod
    def _to_cpu(
        activations: dict[str, Tensor],
    ) -> dict[str, Float[Tensor, "B C D H W"]]:

        host = {}
        for layer_name, act in activations.items():

            act = act.to(device="cpu", dtype=torch.float32)
            host[layer_name] = act

        return host

    @torch.inference_mode()
    def get_activations(
        self, inputs: NestedTensor, to_cpu: bool = True
    ) -> dict[str, Float[Tensor, "B C D H W"]]:

        with self._hooks_registered():
            acts = self._forward(inputs)

        return self._to_cpu(acts) if to_cpu else acts

    @torch.inference_mode()
    def get_activation_shape(self, inputs: NestedTensor) -> dict[str, tuple[int, ...]]:
        """
        inspect the shape of target_layers
        """

        acts = self.get_activations(inputs, to_cpu=False)

        return {layer_name: tuple(act.shape) for layer_name, act in acts.items()}

    @torch.inference_mode()
    def compute_mono_sum_and_bino_resp(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ):

        with self._hooks_registered():

            # compute the baseline and the sum of monocular left and right
            baseline, mono_sum_resp = self._baseline_and_mono_sum_resp(inputs, gray)

            # compute binocular response
            bino_resp: dict[str, Tensor] = {}

            def sink(layer_name: str, output: Tensor):
                bino_resp[layer_name] = output.float() - baseline[layer_name]

            self._run(inputs, sink)

        # return mono_left, mono_right, baseline, mono_sum_resp, bino_resp
        return mono_sum_resp, bino_resp

    @torch.inference_mode()
    def compute_bino_integration(self, dataloader, thresh: float = 0.5):
        # ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:

        integration_ratio_total: dict[str, Tensor] | None = None
        integration_count_total: dict[str, Tensor] | None = None
        sample_total = 0

        with self._hooks_registered():
            for inputs in tqdm(dataloader):

                # move inputs to gpu
                inputs = self.engine._to_device(inputs)

                # make gray images
                gray = self._get_gray(inputs.left)

                # count sample
                sample_total += inputs.left.shape[0]

                # compute integration_ratio and integration_count
                ratio, count = self._bino_integration(inputs, gray, thresh)

                # move ratio and count to buffer
                if integration_ratio_total is None:
                    integration_ratio_total = ratio
                    integration_count_total = count
                else:
                    for layer_name in ratio:
                        if (
                            integration_ratio_total[layer_name].shape
                            != ratio[layer_name].shape
                        ):
                            raise ValueError(
                                f"Activation shape changed for {layer_name}"
                            )

                        integration_ratio_total[layer_name].add_(ratio[layer_name])
                        integration_count_total[layer_name].add_(count[layer_name])

        if sample_total == 0:
            raise ValueError("dataloader contains no samples")

        integration_ratio = {
            k: (v / sample_total).cpu() for k, v in integration_ratio_total.items()
        }
        integration_count = {
            k: (v / sample_total).cpu() for k, v in integration_count_total.items()
        }

        # save data
        torch.save(integration_ratio, f"{self.bi_dir}/bino_integration_ratio.pt")
        torch.save(integration_count, f"{self.bi_dir}/bino_integration_count.pt")

        # return integration_ratio, integration_count

    def plotLine_integration(self, metric: str = "ratio", save_flag: bool = True):

        if metric not in ("ratio", "count"):
            raise ValueError("metric must be 'ratio' or 'count'")

        # load data
        filename = f"{self.bi_dir}/bino_integration_{metric}.pt"
        data = torch.load(filename, map_location="cpu")

        # pool all convs that belong to the same numbered layer
        per_layer: dict[int, list[tuple[int, float, float]]] = {}
        for layer_name, t in data.items():
            t = t.flatten()
            per_layer.setdefault(self.layer_index_of[layer_name], []).append(
                (t.numel(), t.mean().item(), t.var(unbiased=False).item())
            )

        x, mean, std = [], [], []
        for idx in sorted(per_layer):
            parts = per_layer[idx]
            n = sum(p[0] for p in parts)
            m = sum(p[0] * p[1] for p in parts) / n
            ex2 = sum(p[0] * (p[2] + p[1] ** 2) for p in parts) / n
            x.append(idx)
            mean.append(m)
            std.append(max(ex2 - m**2, 0.0) ** 0.5)
        x, mean, std = np.array(x), np.array(mean), np.array(std)

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
            f"{self.model_name} - {self.interaction}: "
            f"Binocular integration {metric} across layers",
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
            mean,
            "-o",
            linewidth=2,
            markersize=12,
            color=colors[0],
            # label=label,
            # linestyle=linestyle,
        )
        # axes.plot(x, integration_mean, "o", markersize=12, color=colors[0])
        axes.fill_between(
            x,
            mean - std,
            mean + std,
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

        # plot reference line
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
                f"{self.bi_dir}/bino_integration_{metric}_{self.engine.interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )
