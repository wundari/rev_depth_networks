# %%
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import numpy as np
from contextlib import contextmanager
from tqdm import tqdm
from typing import Callable
from jaxtyping import Float
from scipy.stats import sem

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
        self.interaction = engine.binocular_interaction
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

    def update_network_config(
        self, interaction: str, seed: int, epoch: int, iter: int
    ) -> None:

        # update experiment dirs based on the new interaction
        self.engine.update_network_config(interaction, seed, epoch, iter)

        # update bi_dir:
        self.make_dirs()

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

        return out

    def _mono_and_bino_resp_spatial_avg(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ):
        """
        compute mono_left, mono_right, and binocular responses, spatially averaged across Height
        and Width axis.

        activation_left = activation(L, gray)
        activation_right = activation(gray, R)
        activation_bino = activation(L, R)

        mono_left = activation_left - baseline
        mono_right = activation_right - baseline
        bino_resp = activation_bino - baseline

        """

        baseline = self._forward(self.replace_inputs(inputs, gray, closed_eye="both"))

        # allocate buffer for bino_resp
        mono_left_resp: dict[str, Tensor] = {}
        mono_right_resp: dict[str, Tensor] = {}
        bino_resp: dict[str, Tensor] = {}

        # define a "sink" function used in the hooking process for computing mono_resp
        def sink_mono_left(layer_name: str, output: Tensor):
            mono_left_resp[layer_name] = (output.float() - baseline[layer_name]).mean(
                dim=(-2, -1)
            )

        def sink_mono_right(layer_name: str, output: Tensor):
            mono_right_resp[layer_name] = (output.float() - baseline[layer_name]).mean(
                dim=(-2, -1)
            )

        # define a "sink" function used in the hooking process for computing bino_resp
        def sink_bino(layer_name: str, output: Tensor):
            bino_resp[layer_name] = (output.float() - baseline[layer_name]).mean(
                dim=(-2, -1)
            )

        self._run(self.replace_inputs(inputs, gray, closed_eye="right"), sink_mono_left)
        self._run(self.replace_inputs(inputs, gray, closed_eye="left"), sink_mono_right)
        self._run(inputs, sink_bino)

        return mono_left_resp, mono_right_resp, bino_resp

    def _bino_sum_and_mul_resp_spatial_avg(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ):
        """
        compute binocular summation (mono_left + mono_right) and
        binocular multiplication (mono_left * mono_right)

        """

        baseline = self._forward(self.replace_inputs(inputs, gray, closed_eye="both"))

        # allocate buffer
        mono_left_resp: dict[str, Tensor] = {}
        bino_sum_resp: dict[str, Tensor] = {}
        bino_mul_resp: dict[str, Tensor] = {}

        # define sink function used in the hooking process for computing bino_sum and bino_mul
        def sink_left(layer_name: str, output: Tensor):
            mono_left_resp[layer_name] = output.float() - baseline[layer_name]

        def sink_right(layer_name: str, output: Tensor):
            mono_right = output.float() - baseline[layer_name]  # mono_right_resp

            # remove baseline[layer_name]
            # baseline.pop(layer_name)

            # move mono_left_resp[layer_name] to a variable then frees it
            mono_left = mono_left_resp[layer_name]

            # compute bino_sum and bino_mul, spatially averaged across H and W axis
            bino_sum_resp[layer_name] = (mono_left + mono_right).mean(dim=(-2, -1))
            bino_mul_resp[layer_name] = (mono_left * mono_right).mean(dim=(-2, -1))

        self._run(
            self.replace_inputs(inputs, gray, closed_eye="right"), sink_left
        )  # mono_left_resp
        self._run(self.replace_inputs(inputs, gray, closed_eye="left"), sink_right)

        return bino_sum_resp, bino_mul_resp

    def _baseline_and_bino_sum_resp(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ) -> tuple[
        dict[str, Float[Tensor, "B C D H W"]], dict[str, Float[Tensor, "B C D H W"]]
    ]:
        """
        baseline = activation(gray, gray)
        activation_left = activation(L, gray)
        activation_right = activation(gray, R)
        bino_sum_resp = (activation_left - baseline) + (activation_right - baseline)

        run 3 forward passes; only `baseline` and `mono_sum_resp` stay in memory.
        """

        baseline = self._forward(self.replace_inputs(inputs, gray, closed_eye="both"))

        # allocate buffer for bino_sum_resp (binocular summation = mono_resp_left + mono_resp_right)
        bino_sum_resp: dict[str, Tensor] = {}

        # define a "sink" function used in the hooking process for computing bino_sum_resp
        def sink(layer_name: str, output: Tensor):
            if layer_name not in bino_sum_resp:
                bino_sum_resp[layer_name] = (
                    output.float() - baseline[layer_name]
                )  # new activation tensor
            else:
                bino_sum_resp[layer_name].add_(output).sub_(baseline[layer_name])

        for closed_eye in ("left", "right"):
            self._run(self.replace_inputs(inputs, gray, closed_eye), sink)

        return baseline, bino_sum_resp

    def _bino_integration(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ) -> dict[str, Tensor]:
        """
        Compute binocular-integration ratio
        binocular-integration ratio: ratio between binocular_resp / binocular_summation

        run 4 forward passes in total.
        The ratio is reduced over the batch inside the hook,
        so the full bino_resp / delta tensors never coexist for all layers.
        """

        # compute baseline and binocular summation resp  (mono_left + mono_right)
        baseline, bino_sum_resp = self._baseline_and_bino_sum_resp(inputs, gray)

        # compute integration ratio and integration count
        integration_ratio: dict[str, Tensor] = {}
        # integration_count: dict[str, Tensor] = {}

        # def sink(layer_name: str, output: Tensor):
        #     bino_sum_layer = bino_sum_resp[
        #         layer_name
        #     ]  # binocular summation: mono_left + mono_right
        #     bino_resp_layer = output.float() - baseline[layer_name]  # binocular resp

        #     delta = bino_resp_layer - bino_sum_layer
        #     den = bino_resp_layer.abs_().add_(bino_sum_layer.abs_()).add_(self.TOL)
        #     # ratio = delta.abs_().div_(den)
        #     ratio = delta.div_(den)

        #     integration_ratio[layer_name] = ratio.sum(dim=0)
        #     integration_count[layer_name] = (ratio > thresh).sum(
        #         dim=0, dtype=torch.float32
        #     )

        def sink(layer_name: str, output: Tensor):
            bino_sum_layer = bino_sum_resp[
                layer_name
            ]  # binocular summation: mono_left + mono_right
            bino_resp_layer = output.float() - baseline[layer_name]  # binocular resp

            delta = bino_resp_layer.abs_() - bino_sum_layer.abs_()
            den = bino_resp_layer.add_(bino_sum_layer).add_(
                self.TOL
            )  # here, bino_resp_layer and bino_sum_layer are
            # absolute by .abs_() from the previous step
            ratio = delta.div_(den)

            integration_ratio[layer_name] = ratio.sum(dim=0)
            # integration_count[layer_name] = (ratio > thresh).sum(
            #     dim=0, dtype=torch.float32
            # )

        self._run(inputs, sink)

        return integration_ratio  # , integration_count

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
    def compute_bino_sum_and_bino_resp(
        self, inputs: NestedTensor, gray: Float[Tensor, "B C H W"]
    ):
        """
        compute binocular summation (mono_left + mono_right) and binocular response

        """

        with self._hooks_registered():

            # compute the baseline and the sum of monocular left and right
            baseline, bino_sum_resp = self._baseline_and_bino_sum_resp(inputs, gray)

            # compute binocular response
            bino_resp: dict[str, Tensor] = {}

            def sink(layer_name: str, output: Tensor):
                bino_resp[layer_name] = output.float() - baseline[layer_name]

            self._run(inputs, sink)

        # return mono_left, mono_right, baseline, bino_sum_resp, bino_resp
        return bino_sum_resp, bino_resp

    @torch.inference_mode()
    def compute_mono_and_bino_resp_spatial_avg(self, dataloader):
        """
        Compute monocular and binocular responses.
        Because it takes a lot of memory, it's better to average spatially
        along Height and Width dimension

        """

        mono_left_resp: dict[str, list[Tensor]] = {}
        mono_right_resp: dict[str, list[Tensor]] = {}
        bino_resp: dict[str, list[Tensor]] = {}

        with self._hooks_registered():
            for inputs in tqdm(dataloader):

                # move inputs to gpu
                inputs = self.engine._to_device(inputs)

                # make gray images
                gray = self._get_gray(inputs.left)

                # compute monocular and binocular responses
                mono_left, mono_right, bino = self._mono_and_bino_resp_spatial_avg(
                    inputs, gray
                )

                # move this batch to CPU and store it
                for layer_name, act in mono_left.items():
                    mono_left_resp.setdefault(layer_name, []).append(act.cpu())
                    mono_right_resp.setdefault(layer_name, []).append(
                        mono_right[layer_name].cpu()
                    )
                    bino_resp.setdefault(layer_name, []).append(bino[layer_name].cpu())

                del mono_left, mono_right, bino  # free GPU memory before the next batch

        # concatenate batches along the sample dimension
        def concat(store: dict[str, list[Tensor]]) -> dict[str, Tensor]:
            return {
                layer_name: torch.cat(acts, dim=0) for layer_name, acts in store.items()
            }

        mono_left_resp = concat(mono_left_resp)
        mono_right_resp = concat(mono_right_resp)
        bino_resp = concat(bino_resp)

        # save data:
        torch.save(mono_left_resp, f"{self.bi_dir}/mono_left_resp.pt")
        torch.save(mono_right_resp, f"{self.bi_dir}/mono_right_resp.pt")
        torch.save(bino_resp, f"{self.bi_dir}/bino_resp.pt")

    @torch.inference_mode()
    def compute_bino_sum_and_bino_mul_spatial_avg(self, dataloader):
        """
        Compute binocular summation and binocular multiplication.
        Because it takes a lot of memory, it's better to average
        spatially along Height and Width dimension

        """

        bino_sum_resp: dict[str, list[Tensor]] = {}
        bino_mul_resp: dict[str, list[Tensor]] = {}

        with self._hooks_registered():
            for inputs in tqdm(dataloader):

                # move inputs to gpu
                inputs = self.engine._to_device(inputs)

                # make gray images
                gray = self._get_gray(inputs.left)

                # compute monocular and binocular responses
                bino_sum, bino_mul = self._bino_sum_and_mul_resp_spatial_avg(
                    inputs, gray
                )

                # move this batch to CPU and store it
                for layer_name, act in bino_sum.items():
                    bino_sum_resp.setdefault(layer_name, []).append(act.cpu())
                    bino_mul_resp.setdefault(layer_name, []).append(
                        bino_mul[layer_name].cpu()
                    )

                del (
                    bino_sum,
                    bino_mul,
                )  # free GPU memory before the next batch

        # concatenate batches along the sample dimension
        def concat(store: dict[str, list[Tensor]]) -> dict[str, Tensor]:
            return {
                layer_name: torch.cat(acts, dim=0) for layer_name, acts in store.items()
            }

        bino_sum_resp = concat(bino_sum_resp)
        bino_mul_resp = concat(bino_mul_resp)

        # save data:
        torch.save(bino_sum_resp, f"{self.bi_dir}/bino_sum_resp.pt")
        torch.save(bino_mul_resp, f"{self.bi_dir}/bino_mul_resp.pt")

    @torch.inference_mode()
    def compute_bino_integration(self, dataloader):
        # ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:

        integration_ratio_total: dict[str, Tensor] | None = None
        # integration_count_total: dict[str, Tensor] | None = None
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
                ratio = self._bino_integration(inputs, gray)

                # move ratio and count to buffer
                if integration_ratio_total is None:
                    integration_ratio_total = ratio
                    # integration_count_total = count
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
                        # integration_count_total[layer_name].add_(count[layer_name])

        if sample_total == 0:
            raise ValueError("dataloader contains no samples")

        integration_ratio = {
            k: (v / sample_total).cpu() for k, v in integration_ratio_total.items()
        }
        # integration_count = {
        #     k: (v / sample_total).cpu() for k, v in integration_count_total.items()
        # }

        # save data
        torch.save(integration_ratio, f"{self.bi_dir}/bino_integration_ratio.pt")
        # torch.save(integration_count, f"{self.bi_dir}/bino_integration_count.pt")

    def _plotLine_correlation(self, ax, corr_x_vs_y, subtitle):

        # compute the mean of correlation across # neurons
        corr_mean = np.empty(len(self.target_layers), dtype=np.float32)
        corr_std = np.empty_like(corr_mean)
        for i, (layer_name, corr) in enumerate(corr_x_vs_y.items()):
            corr_mean[i] = corr.mean()
            corr_std[i] = corr.std()

        if self.model_name == "BNN":
            x = np.arange(3, 5)
        else:  # GC-Net variants
            x = np.arange(19, 38)
        ax.plot(x, corr_mean, "-o", linewidth=2, markersize=12, color="black")
        ax.fill_between(
            x,
            corr_mean - corr_std,
            corr_mean + corr_std,
            color="black",
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
        y_low = -1.0
        y_up = 1.1
        y_step = 0.25

        # plot reference line
        ax.hlines(
            [0.0], xmin=x_low - 1, xmax=x_up, colors="red", linestyles="--", linewidth=3
        )

        ax.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        ax.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        ax.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        ax.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # title
        ax.set_title(subtitle)

        ax.set_xlim(x_low - 1, x_up)
        ax.set_ylim(y_low, y_up)

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

    def plotLine_bino_correlation_interaction(self, save_flag: bool = True):

        # load data
        filename = f"{self.bi_dir}/mono_left_resp.pt"
        mono_left_resp = torch.load(filename, map_location="cpu")
        filename = f"{self.bi_dir}/mono_right_resp.pt"
        mono_right_resp = torch.load(filename, map_location="cpu")
        filename = f"{self.bi_dir}/bino_resp.pt"
        bino_resp = torch.load(filename, map_location="cpu")
        filename = f"{self.bi_dir}/bino_sum_resp.pt"
        bino_sum_resp = torch.load(filename, map_location="cpu")
        filename = f"{self.bi_dir}/bino_mul_resp.pt"
        bino_mul_resp = torch.load(filename, map_location="cpu")

        # compute correlation
        def corr_xy(
            x: Float[Tensor, "B n_var"], y: Float[Tensor, "B n_var"], eps: float = 1e-8
        ) -> Float[Tensor, "n_var"]:

            # mean-center
            x = x - x.mean(dim=0, keepdim=True)
            y = y - y.mean(dim=0, keepdim=True)
            cov = (x * y).sum(dim=0) / (
                x.pow(2).sum(dim=0).sqrt() * y.pow(2).sum(dim=0).sqrt() + eps
            )

            return cov

        # compute correlation
        corr_sum_vs_bino: dict[str, Tensor] = {}
        corr_mul_vs_bino: dict[str, Tensor] = {}
        corr_left_vs_right: dict[str, Tensor] = {}
        for layer_name, bino in bino_resp.items():

            B = bino.shape[0]
            y = bino.reshape(B, -1)

            # binocular summation vs binocular resp
            x = bino_sum_resp[layer_name].reshape(B, -1)
            corr = corr_xy(x, y)
            corr_sum_vs_bino[layer_name] = corr

            # binocular multiplication vs binocular resp
            x = bino_mul_resp[layer_name].reshape(B, -1)
            corr = corr_xy(x, y)
            corr_mul_vs_bino[layer_name] = corr

            # mono_left vs mono_right
            x = mono_left_resp[layer_name].reshape(B, -1)
            y = mono_right_resp[layer_name].reshape(B, -1)
            corr = corr_xy(x, y)
            corr_left_vs_right[layer_name] = corr

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (12, 18)
        n_row = 3
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.01,
            f"{self.model_name} - {self.interaction} (seed: {self.config.experiment_id}): "
            f"Correlation across layers",
            ha="center",
        )
        fig.text(-0.04, 0.5, f"Pearson correlation", va="center", rotation=90)
        fig.text(0.5, -0.04, "Layer", ha="center")
        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        self._plotLine_correlation(axes[0], corr_mul_vs_bino, "bino_mul vs bino_resp")
        self._plotLine_correlation(axes[1], corr_sum_vs_bino, "bino_sum vs bino_resp")
        self._plotLine_correlation(
            axes[2], corr_left_vs_right, "mono_left_resp vs mono_right_resp"
        )

        # save figure
        if save_flag == 1:
            fig.savefig(
                f"{self.bi_dir}/bino_correlation_{self.engine.binocular_interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_bino_correlation_all_interactions(self, save_flag: bool = True):

        # load data
        integration_mean_all: dict[str, np.ndarray] = {}
        integration_std_all: dict[str, np.ndarray] = {}
        for interaction in self.config.interactions:
            mean_seeds = np.empty(
                (len(self.config.seed_to_analyse), len(self.target_layers)),
                dtype=np.float32,
            )
            for s, seed in enumerate(self.config.seed_to_analyse):

                # interaction = "bem"
                # s = 0
                # seed = bi.config.seed_to_analyse[s]
                if interaction == "default":
                    epoch2load, iter2load = self.config.epoch_iter_to_load_default[s]
                elif interaction == "bem":
                    epoch2load, iter2load = self.config.epoch_iter_to_load_bem[s]
                elif interaction == "cmm":
                    epoch2load, iter2load = self.config.epoch_iter_to_load_cmm[s]
                else:  # sum_diff
                    epoch2load, iter2load = self.config.epoch_iter_to_load_sum_diff[s]

                # update network configuration and directory addresses
                self.update_network_config(interaction, seed, epoch2load, iter2load)

                filename = f"{self.bi_dir}/bino_integration_{metric}.pt"
                data = torch.load(filename, map_location="cpu")

                # pool all convs that belong to the same numbered layer
                per_layer: dict[int, list[tuple[int, float, float]]] = {}
                for layer_name, t in data.items():
                    t = t.flatten()
                    per_layer.setdefault(self.layer_index_of[layer_name], []).append(
                        (t.numel(), t.mean().item(), t.var(unbiased=False).item())
                    )

                x, mean = [], []
                for idx in sorted(per_layer):
                    parts = per_layer[idx]
                    n = sum(p[0] for p in parts)
                    m = sum(p[0] * p[1] for p in parts) / n
                    x.append(idx)
                    mean.append(m)
                x, mean = np.array(x), np.array(mean)

                # move to buffer
                mean_seeds[s] = mean

            # for each binocular interaction, average over seeds
            integration_mean_all[interaction] = mean_seeds.mean(axis=0)
            integration_std_all[interaction] = sem(mean_seeds, axis=0)

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
            f"{self.model_name} - all interactions: "
            f"Binocular integration {metric} across layers",
            ha="center",
        )
        fig.text(-0.04, 0.5, f"Integration {metric}", va="center", rotation=90)
        fig.text(0.5, -0.04, "Layer", ha="center")
        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        for i, interaction in enumerate(self.config.interactions):
            mean = integration_mean_all[interaction]
            err = integration_std_all[interaction]
            axes.plot(
                x,
                mean,
                "-o",
                linewidth=2,
                markersize=12,
                color=colors[i],
                label=labels[i],
                # linestyle=linestyle,
            )
            # axes.plot(x, integration_mean, "o", markersize=12, color=colors[0])
            axes.fill_between(
                x,
                mean - err,
                mean + err,
                color=colors[i],
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
        y_low = -0.5
        y_up = 0.51
        y_step = 0.1

        # plot reference line
        axes.hlines(
            [0.0],
            xmin=x_low - 1,
            xmax=x_up,
            colors="red",
            linestyles="--",
            linewidth=3,
        )

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 1, x_up)
        axes.set_ylim(y_low, y_up)

        # legend
        plt.legend(
            loc="upper right",
            fontsize=16,
            frameon=True,
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save figure
        if save_flag == 1:
            save_dir = f"{self.bi_dir}/../../../bino_interaction_group/plots"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            fig.savefig(
                f"{save_dir}/bino_integration_{metric}_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def _plotScatter_correlation(
        self,
        resp_x,
        resp_y,
        title,
        xlabel,
        ylabel,
        n_points,
        save_flag: bool = True,
    ):

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        figsize = (15, 15)
        if self.model_name == "BNN":
            n_row = 1
            n_col = 2
        else:  # gcnet variants
            n_row = 5
            n_col = 4

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )
        axes = axes.ravel()

        g = torch.Generator().manual_seed(0)
        for ax, (layer_name, left) in zip(axes, resp_x.items()):

            # flatten ALL samples and units into two vectors, then pick a random subset
            x = left.flatten()
            y = resp_y[layer_name].flatten()
            idx = torch.randint(
                0, x.numel(), (n_points,), generator=g
            )  # no huge randperm
            x, y = x[idx].numpy(), y[idx].numpy()

            ax.scatter(x, y, s=2, c="k", alpha=0.3, linewidths=0, rasterized=True)
            ax.set_aspect("equal")

            # data-driven, symmetric limits (ignore extreme outliers)
            lim = np.percentile(np.abs(np.concatenate([x, y])), 99.9)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

            # reference lines: zero axes and the identity line
            ax.axhline(0, color="red", linestyle="--", linewidth=2)
            ax.axvline(0, color="red", linestyle="--", linewidth=2)
            ax.axline((0, 0), slope=1, color="red", linestyle="--", linewidth=2)

            ax.set_title(layer_name, fontsize=12)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

        # hide unused panels (20 axes for 19 layers)
        for ax in axes[len(resp_x) :]:
            ax.set_visible(False)

        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        fig.suptitle(
            f"{self.model_name} - {self.interaction} (seed: {self.config.experiment_id}): "
            f"{title}",
            y=1.0,
        )
        fig.tight_layout()

        # save figure
        if save_flag == 1:
            fig.savefig(
                f"{self.bi_dir}/{title}_{self.engine.binocular_interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotScatter_bino_correlation_interaction(
        self, n_points: int = 20000, save_flag: bool = True
    ):

        # load data
        filename = f"{self.bi_dir}/mono_left_resp.pt"
        mono_left_resp = torch.load(filename, map_location="cpu")
        filename = f"{self.bi_dir}/mono_right_resp.pt"
        mono_right_resp = torch.load(filename, map_location="cpu")
        filename = f"{self.bi_dir}/bino_resp.pt"
        bino_resp = torch.load(filename, map_location="cpu")
        filename = f"{self.bi_dir}/bino_sum_resp.pt"
        bino_sum_resp = torch.load(filename, map_location="cpu")
        filename = f"{self.bi_dir}/bino_mul_resp.pt"
        bino_mul_resp = torch.load(filename, map_location="cpu")

        # mono_left vs mono_right
        self._plotScatter_correlation(
            resp_x=mono_left_resp,
            resp_y=mono_right_resp,
            title="mono_left_resp_vs_mono_right_resp",
            xlabel="mono_left_resp",
            ylabel="mono_right_resp",
            n_points=n_points,
            save_flag=save_flag,
        )

        # bino_mul vs bino_resp
        self._plotScatter_correlation(
            resp_x=bino_mul_resp,
            resp_y=bino_resp,
            title="bino_mul_resp_vs_bino_resp",
            xlabel="bino_mul_resp",
            ylabel="bino_resp",
            n_points=n_points,
            save_flag=save_flag,
        )

        # mono_left vs bino_resp
        self._plotScatter_correlation(
            resp_x=bino_sum_resp,
            resp_y=bino_resp,
            title="bino_sum_resp_vs_bino_resp",
            xlabel="bino_sum_resp",
            ylabel="bino_resp",
            n_points=n_points,
            save_flag=save_flag,
        )

    def plotLine_bino_integration_interaction(
        self, metric: str = "ratio", save_flag: bool = True
    ):

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
        fig.text(-0.04, 0.5, f"Integration {metric}", va="center", rotation=90)
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
        y_low = -0.5
        y_up = 0.51
        y_step = 0.1

        # plot reference line
        axes.hlines(
            [0.0], xmin=x_low - 1, xmax=x_up, colors="red", linestyles="--", linewidth=3
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
                f"{self.bi_dir}/bino_integration_{metric}_{self.engine.binocular_interaction}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_bino_integration_all_interactions(
        self, metric: str = "ratio", save_flag: bool = True
    ):

        if metric not in ("ratio", "count"):
            raise ValueError("metric must be 'ratio' or 'count'")

        # load data
        integration_mean_all: dict[str, np.ndarray] = {}
        integration_std_all: dict[str, np.ndarray] = {}
        for interaction in self.config.interactions:
            mean_seeds = np.empty(
                (len(self.config.seed_to_analyse), len(self.target_layers)),
                dtype=np.float32,
            )
            for s, seed in enumerate(self.config.seed_to_analyse):

                # interaction = "bem"
                # s = 0
                # seed = bi.config.seed_to_analyse[s]
                if interaction == "default":
                    epoch2load, iter2load = self.config.epoch_iter_to_load_default[s]
                elif interaction == "bem":
                    epoch2load, iter2load = self.config.epoch_iter_to_load_bem[s]
                elif interaction == "cmm":
                    epoch2load, iter2load = self.config.epoch_iter_to_load_cmm[s]
                else:  # sum_diff
                    epoch2load, iter2load = self.config.epoch_iter_to_load_sum_diff[s]

                # update network configuration and directory addresses
                self.update_network_config(interaction, seed, epoch2load, iter2load)

                filename = f"{self.bi_dir}/bino_integration_{metric}.pt"
                data = torch.load(filename, map_location="cpu")

                # pool all convs that belong to the same numbered layer
                per_layer: dict[int, list[tuple[int, float, float]]] = {}
                for layer_name, t in data.items():
                    t = t.flatten()
                    per_layer.setdefault(self.layer_index_of[layer_name], []).append(
                        (t.numel(), t.mean().item(), t.var(unbiased=False).item())
                    )

                x, mean = [], []
                for idx in sorted(per_layer):
                    parts = per_layer[idx]
                    n = sum(p[0] for p in parts)
                    m = sum(p[0] * p[1] for p in parts) / n
                    x.append(idx)
                    mean.append(m)
                x, mean = np.array(x), np.array(mean)

                # move to buffer
                mean_seeds[s] = mean

            # for each binocular interaction, average over seeds
            integration_mean_all[interaction] = mean_seeds.mean(axis=0)
            integration_std_all[interaction] = sem(mean_seeds, axis=0)

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
            f"{self.model_name} - all interactions: "
            f"Binocular integration {metric} across layers",
            ha="center",
        )
        fig.text(-0.04, 0.5, f"Integration {metric}", va="center", rotation=90)
        fig.text(0.5, -0.04, "Layer", ha="center")
        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        for i, interaction in enumerate(self.config.interactions):
            mean = integration_mean_all[interaction]
            err = integration_std_all[interaction]
            axes.plot(
                x,
                mean,
                "-o",
                linewidth=2,
                markersize=12,
                color=colors[i],
                label=labels[i],
                # linestyle=linestyle,
            )
            # axes.plot(x, integration_mean, "o", markersize=12, color=colors[0])
            axes.fill_between(
                x,
                mean - err,
                mean + err,
                color=colors[i],
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
        y_low = -0.5
        y_up = 0.51
        y_step = 0.1

        # plot reference line
        axes.hlines(
            [0.0],
            xmin=x_low - 1,
            xmax=x_up,
            colors="red",
            linestyles="--",
            linewidth=3,
        )

        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        axes.set_xlim(x_low - 1, x_up)
        axes.set_ylim(y_low, y_up)

        # legend
        plt.legend(
            loc="upper right",
            fontsize=16,
            frameon=True,
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save figure
        if save_flag == 1:
            save_dir = f"{self.bi_dir}/../../../bino_interaction_group/plots"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            fig.savefig(
                f"{save_dir}/bino_integration_{metric}_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )
