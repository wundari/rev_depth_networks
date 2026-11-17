# %%
import torch
from torch import nn, Tensor

import re
from contextlib import contextmanager
from tqdm import tqdm
from jaxtyping import Float

from config.config_gcnet_lr import ConfigGCNet
from config.config_at import ConfigAT
from engine.engine_gcnet_lr import EngineGCNet
from utilities.misc import NestedTensor

# %%


class ActivationTracker:

    def __init__(self, engine: EngineGCNet, config: ConfigAT):

        self.engine = engine
        self.config = config

        self.model_name = engine.model_name
        self.interaction = engine.interaction
        self.model = getattr(
            self.engine.model, "_orig_mod", self.engine.model
        )  # remove "_orig_mod" phrases
        self.device = next(self.model.parameters()).device
        self.model.eval()

        self.target_layer_indices = config.target_layer_indices

        self.handles = []
        self.conv_types = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)

        # find target layers that are matched with layer_indices
        self.target_layers = self.find_target_layers(self.target_layer_indices)

        # print target layers
        self.print_target_layers()

        # allocate buffer for storing activations
        # self._buffer_mono: dict[str, Tensor] = {}
        self._buffer: dict[str, Tensor] = {}

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

    def print_target_layers(self):

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
        image: Float[Tensor, "B C H W"],
        baseline: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> Float[Tensor, "B C H W"]:
        """
        Convert raw gray to normalized RGB; image shape is [1, 3, H, W].
        """

        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Expected floating RGB images [B, 3, H, W].")
        if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
            raise ValueError(
                "mean/std must have three entries and std must be positive."
            )

        values = [(baseline - m) / s for m, s in zip(mean, std)]
        gray = image.new_tensor(values).view(1, 3, 1, 1)

        return gray.expand_as(image).contiguous()

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

    def _forward(self, inputs: NestedTensor) -> dict[str, Tensor]:
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

    def _mono_sum_resp(self, inputs: NestedTensor) -> dict[str, Tensor]:
        """
        left + right - baseline, accumulated in fp32 on the GPU.

        Only one running total plus one in-flight activation set is resident,
        instead of three separate sets.
        """

        gray = self.make_gray_image(inputs.left)

        # compute convolutional layer outputs when left eye is closed
        # (left_input = gray images)
        mono_right = self._forward(self.replace_inputs(inputs, gray, closed_eye="left"))

        total = {name: act.float() for name, act in mono_right.items()}

        del mono_right

        # + mono_left_resp (right eye receives gray); - baseline (both eyes receive gray)
        for closed_eye, sign in (("right", +1), ("both", -1)):
            acts = self._forward(self.replace_inputs(inputs, gray, closed_eye))

            for name, act in acts.items():
                total[name].add_(act, alpha=sign)

        return total

    # def monocular_sum(
    #     self,
    #     mono_left: dict[str, Float[Tensor, "B C D H W"]],
    #     mono_right: dict[str, Float[Tensor, "B C D H W"]],
    #     baseline: dict[str, Float[Tensor, "B C D H W"]],
    # ) -> dict[str, Float[Tensor, "B C D H W"]]:
    #     """
    #     Compute left + right - baseline
    #     """

    #     if mono_left.keys() != mono_right.keys() or mono_left.keys() != baseline.keys():
    #         raise ValueError("Activation layer names must match.")

    #     out = {}
    #     for k in mono_left:
    #         if (
    #             mono_left[k].shape != mono_right[k].shape
    #             or mono_left[k].shape != baseline[k].shape
    #         ):
    #             raise ValueError(f"Activation shapes differ for {k}.")

    #         # clone prevents modifying left when it is already fp32.
    #         value = mono_left[k].to(dtype=torch.float32, copy=True)
    #         value.add_(mono_right[k]).sub_(baseline[k])
    #         out[k] = value

    #     return out

    # @torch.inference_mode()
    # def get_activations(self, inputs: NestedTensor):

    #     self._buffer = {}
    #     try:

    #         self.hook_layers()

    #         with torch.autocast(
    #             device_type=self.device.type, dtype=torch.bfloat16, enabled=True
    #         ):
    #             self.model(inputs)

    #         missing_layer = set(self.target_layers) - set(self._buffer)
    #         if missing_layer:
    #             raise RuntimeError(
    #                 f"Selected layers did not execute: {sorted(missing_layer)}"
    #             )

    #         return dict(self._buffer)

    #     finally:
    #         self.close()
    #         self._buffer = {}

    # @torch.inference_mode()
    # def get_mono_sum_resp(self, inputs: NestedTensor) -> dict[str, Tensor]:

    #     # create gray image
    #     gray = self.make_gray_image(inputs.left)

    #     try:
    #         self.hook_layers()

    #         with torch.autocast(
    #             device_type=self.device.type, dtype=torch.bfloat16, enabled=True
    #         ):

    #             # compute convolutional layer outputs when left eye is closed
    #             # (left_input = gray images)
    #             modified_inputs = self.replace_inputs(inputs, gray, closed_eye="left")
    #             mono_right = self.get_activations(modified_inputs)

    #             # compute convolutional layer outputs when right eye is closed
    #             # (right_input = gray images)
    #             modified_inputs = self.replace_inputs(inputs, gray, closed_eye="right")
    #             mono_left = self.get_activations(modified_inputs)

    #             # compute baseline (both eyes receive gray images)
    #             modified_inputs = self.replace_inputs(inputs, gray, closed_eye="both")
    #             baseline = self.get_activations(modified_inputs)

    #             # compute the sum of monocular left and right
    #             self._buffer = self.monocular_sum(mono_left, mono_right, baseline)

    #         missing_layer = set(self.target_layers) - set(self._buffer)
    #         if missing_layer:
    #             raise RuntimeError(
    #                 f"Selected layers did not execute: {sorted(missing_layer)}"
    #             )

    #         return dict(self._buffer)

    #     finally:
    #         self.close()
    #         self._buffer = {}

    def _to_cpu(self, activations: dict[str, Tensor]) -> dict[str, Tensor]:

        host = {}
        for layer_name, act in activations.items():

            act = act.to(dtype=torch.float32)

            host[layer_name] = torch.empty(act.shape, dtype=act.dtype, pin_memory=True)
            host[layer_name].copy_(act, non_blocking=True)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return host

    @torch.inference_mode()
    def get_mono_sum_resp(self, inputs: NestedTensor) -> dict[str, Tensor]:

        with self._hooks_registered():
            return self._to_cpu(self._mono_sum_resp(inputs))

    @torch.inference_mode()
    def get_activations(self, inputs: NestedTensor) -> dict[str, Tensor]:

        with self._hooks_registered():
            return self._to_cpu(self._forward(inputs))

    @torch.inference_mode()
    def compute_mono_and_bino_resp(self, inputs: NestedTensor):

        with self._hooks_registered():
            # each GPU result is freed as soon as it has been copied to the host

            # compute the sum of monocular left and right
            mono_sum_resp = self._to_cpu(self._mono_sum_resp(inputs))

            # compute binocular response
            bino_resp = self._to_cpu(self._forward(inputs))

        # return mono_left, mono_right, baseline, mono_sum_resp, bino_resp
        return mono_sum_resp, bino_resp

    def gather(self, test_loader):

        for i, inputs in enumerate(tqdm(test_loader)):

            inputs = self.engine._to_device(inputs)
            mono_sum_resp, bino_resp = self.compute_mono_and_bino_resp(inputs)


# %%

config = ConfigGCNet()
config_at = ConfigAT()
config.load_state = True
engine = EngineGCNet(config)

tracker = ActivationTracker(engine, config_at)

# prepare dataset
train_loader, validation_loader, test_loader = engine.prepare_dataset()

inputs = next(iter(test_loader))
inputs = engine._to_device(inputs)

mono_sum_resp, bino_resp = tracker.compute_mono_and_bino_resp(inputs)
# mono_left, mono_right, baseline, mono_sum_resp, bino_resp = (
#     tracker.compute_monocular_response(inputs)
# )

# %%
import matplotlib.pyplot as plt

layer_name = "decoder.layer19.0"
# plt.imshow(mono_left[layer_name][0, 0, 48])
# plt.imshow(mono_right[layer_name][0, 0, 70])
# plt.imshow(baseline[layer_name][0, 0, 10])
plt.imshow(mono_sum_resp[layer_name][0, 0, 48])
plt.imshow(bino_resp[layer_name][0, 0, 0])
# tracker.hook_layers()
# tracker.get_activations(inputs)
# activations = tracker._buffer


# %%
