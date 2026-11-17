# %%
from config.config_gcnet_lr import ConfigGCNet
from engine.engine_gcnet_lr import EngineGCNet


import torch
from torch import nn, Tensor
from utilities.misc import NestedTensor

from jaxtyping import Float

from timeit import default_timer as timer
from datetime import datetime
from tqdm import tqdm

import copy
import re

torch.backends.cudnn.benchmark = True  # best for fixed input sizes
# Use the precision API available in the installed PyTorch version.
if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
else:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

# %%


def make_gray_image(
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
        raise ValueError("mean/std must have three entries and std must be positive.")
    values = [(baseline - m) / s for m, s in zip(mean, std)]
    gray = image.new_tensor(values).view(1, 3, 1, 1)
    return gray.expand_as(image).contiguous()


def replace_inputs(
    inputs: NestedTensor, gray: Float[Tensor, "B C H W"], closed_eye: str
) -> NestedTensor:

    if closed_eye not in ["left", "right", "both"]:
        raise TypeError('closed_eye must be "left" or "right" or "both"')

    if inputs.left.shape != inputs.right.shape or gray.shape != inputs.left.shape:
        raise ValueError("Left, right, and gray shapes must match.")

    if gray.device != inputs.left.device or gray.dtype != inputs.left.dtype:
        raise ValueError("Gray and input images must have matching device and dtype.")

    return NestedTensor(
        left=gray if closed_eye in ("left", "both") else inputs.left,
        right=gray if closed_eye in ("right", "both") else inputs.right,
        disp=inputs.disp,
        ref=inputs.ref,
    )


class ActivationExtractor:
    """
    Captures raw outputs of all conv layers numbered in `layers`.

    Layers are located and hooks registered ONCE; each call only runs a forward
    pass. Returned tensors are independent CPU copies (Conv2d: [B,C,H,W]; Conv3d: [B,C,D,H,W]).

    Usage:
        with ActivationExtractor(engine.model) as extract:
            acts = extract(inputs)   # inputs must already be on the device
    """

    def __init__(self, model: nn.Module, layers=range(19, 21)):
        # Eager model, so hooks also work with an engine-compiled model.
        self.model = getattr(model, "_orig_mod", model)
        self.device = next(self.model.parameters()).device
        self.conv_types = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)

        self.targets = self._find_targets(frozenset(layers))
        self._buffer: dict[str, Tensor] = {}
        self._capturing = False

        self._modes = []
        self._handles = []
        self._active = False

    def _find_targets(self, layers) -> dict[str, nn.Module]:
        if not layers:
            raise ValueError("Select at least one numbered layer.")
        targets, found = {}, set()
        for name, module in self.model.named_modules():
            ids = [
                int(m.group(1))
                for part in name.split(".")
                if (m := re.fullmatch(r"layer(\d+)[a-z]*", part))
            ]
            selected = [n for n in ids if n in layers]
            if selected and isinstance(module, self.conv_types):
                targets[name] = module
                found.update(selected)

        missing = set(layers) - found
        if missing:
            raise ValueError(
                f"No convolution found for layers {sorted(missing)}. "
                "The current BNN uses layers 1-4; use GC_Net_LR for 19-37."
            )
        return targets

    def _make_hook(self, name: str):
        def capture(module, args, output):
            if not self._capturing:
                return
            if name in self._buffer:
                raise RuntimeError(f"{name} executed more than once.")
            # Blocking copy protects against subsequent in-place operations and
            # avoids retaining complete activation volumes on the GPU.
            self._buffer[name] = output.detach().to("cpu", copy=True)

        return capture

    @torch.inference_mode()
    def __call__(self, inputs: NestedTensor) -> dict[str, Tensor]:
        if not self._active:
            raise RuntimeError("Use ActivationExtractor inside a with block.")
        if self._capturing:
            raise RuntimeError("Concurrent or recursive extraction is unsupported.")
        if inputs.ref is None:
            raise ValueError("inputs.ref must specify +1 or -1 for each sample.")
        if inputs.left.device != self.device or inputs.right.device != self.device:
            raise ValueError("Move input images to the model device first.")

        self._buffer = {}
        self._capturing = True
        try:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=True,
            ):
                self.model(inputs)
        except BaseException:
            self._buffer.clear()
            raise
        finally:
            self._capturing = False

        unexecuted = set(self.targets) - self._buffer.keys()
        if unexecuted:
            self._buffer.clear()
            raise RuntimeError(f"Layers did not execute: {sorted(unexecuted)}")

        out, self._buffer = self._buffer, {}
        return out

    def close(self):
        if not self._active:
            return
        self._capturing = False
        self._buffer.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, training in self._modes:
            module.training = training
        self._modes.clear()
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Extractor is already active.")
        self._modes = [(m, m.training) for m in self.model.modules()]
        self._active = True
        try:
            self.model.eval()
            for name, module in self.targets.items():
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()


def monocular_sum(left, right, baseline) -> dict[str, Tensor]:
    """Compute left + right - baseline in fp32; keep the result in fp32."""
    if left.keys() != right.keys() or left.keys() != baseline.keys():
        raise ValueError("Activation layer names must match.")
    out = {}
    for k in left:
        if left[k].shape != right[k].shape or left[k].shape != baseline[k].shape:
            raise ValueError(f"Activation shapes differ for {k}.")
        # clone prevents modifying left when it is already fp32.
        value = left[k].to(dtype=torch.float32, copy=True)
        value.add_(right[k]).sub_(baseline[k])
        out[k] = value
    return out


# %% set up DNN

# The guard also prevents dataset workers from rerunning the experiment.
# if __name__ == "__main__":

config = ConfigGCNet()
config.load_state = True
engine = EngineGCNet(config)

# prepare dataset
train_loader, validation_loader, test_loader = engine.prepare_dataset()

# %% compute activations
if SAVE_DIR is not None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if any(SAVE_DIR.glob("batch_*.pt")):
        raise FileExistsError(f"{SAVE_DIR} already contains activation batches.")

time_start = datetime.now().strftime("%H:%M:%S")
t_start = timer()

results = {"left": [], "right": [], "baseline": [], "sum": [], "bino": []}

with ActivationExtractor(engine.model) as extract, torch.inference_mode():
    for idx, inputs in enumerate(tqdm(test_loader)):
        inputs = next(iter(test_loader))
        inputs = engine._to_device(inputs)
        gray = make_gray_image(inputs.left)

        # with ActivationExtractor(engine.model) as extract, torch.inference_mode():
        # Each extraction copies outputs to CPU before the next condition.
        act_left = extract(replace_inputs(inputs, gray, closed_eye="right"))
        act_right = extract(replace_inputs(inputs, gray, closed_eye="left"))
        act_base = extract(replace_inputs(inputs, gray, closed_eye="both"))
        act_sum = monocular_sum(act_left, act_right, act_base)

        record = {"sum": act_sum}
        del act_left, act_right, act_base, act_sum  # record owns the CPU tensors

        # binocular response
        record["bino"] = extract(inputs)

        # Keep reference direction and capture settings with every batch.
        record["metadata"] = {
            "batch_index": idx,
            "ref": inputs.ref.detach().cpu().clone(),
            "binocular_interaction": extract.model.decoder.binocular_interaction,
            "capture": "raw_convolution_output",
            "gray_baseline": 0.5,
            "gray_mean": (0.485, 0.456, 0.406),
            "gray_std": (0.229, 0.224, 0.225),
        }
        if SAVE_DIR is not None:
            torch.save(record, SAVE_DIR / f"batch_{idx:05d}.pt")
        else:
            for key in results:
                results[key].append(record[key])
        del record  # release the previous batch before starting another

t_end = timer()
time_end = datetime.now().strftime("%H:%M:%S")
print(
    f"Computing activations has completed: [{time_start} -> {time_end}] "
    f"[{(t_end - t_start) / 60.0:.2f} mins]\n"
)

# In disk mode these lists are empty: load batch_*.pt files from SAVE_DIR.
# Names used by the original script
activations_left = results["left"]
activations_right = results["right"]
activations_baseline = results["baseline"]
activations_sum = results["sum"]
activations_bino = results["bino"]
# %%

engine.model.eval()
progress = tqdm(test_loader)
activations_left = []
activations_right = []
activations_baseline = []
activations_sum = []
activations_bino = []
for idx, inputs in enumerate(progress):

    # move inputs to gpu
    inputs = engine._to_device(inputs)

    # create blank / gray images
    gray = make_gray_image(inputs.left)

    # compute monocular left response
    modified_inputs = replace_inputs(inputs, gray, closed_eye="right")
    with torch.autocast(
        device_type=torch.device(engine.device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(engine.device).type == "cuda",
    ):
        activation_left = get_neuron_activation(engine.model, modified_inputs)
        activations_left.append(activation_left)

    # compute monocular right response
    modified_inputs = replace_inputs(inputs, gray, closed_eye="left")
    with torch.autocast(
        device_type=torch.device(engine.device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(engine.device).type == "cuda",
    ):
        activation_right = get_neuron_activation(engine.model, modified_inputs)
        activations_right.append(activation_right)

    # compute baseline
    modified_inputs = replace_inputs(inputs, gray, closed_eye="both")
    with torch.autocast(
        device_type=torch.device(engine.device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(engine.device).type == "cuda",
    ):
        activation_baseline = get_neuron_activation(engine.model, modified_inputs)
        activations_baseline.append(activation_baseline)

    # compute monocular sum response
    with torch.autocast(
        device_type=torch.device(engine.device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(engine.device).type == "cuda",
    ):
        activation_sum = {}
        for layer_name, activation in activation_left.items():

            activation_sum[layer_name] = (
                activation
                + activation_right[layer_name]
                - activation_baseline[layer_name]
            )
        activations_sum.append(activation_sum)

    # compute binocular response
    with torch.autocast(
        device_type=torch.device(engine.device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(engine.device).type == "cuda",
    ):
        activations_bino.append(get_neuron_activation(engine.model, inputs))

t_end = timer()
now = datetime.now()
time_end = now.strftime("%H:%M:%S")
dur = (t_end - t_start) / 60.0
print(
    f"Computing activations has completed: [{time_start} -> {time_end}] [{dur:.2f} mins] \n"
)
# %%

inputs = next(iter(test_loader))
inputs = engine._to_device(inputs)
left = inputs.left

# get monocular right response
gray = make_gray_image(inputs.left)
modified_inputs = replace_inputs(inputs, gray, closed_eye="left")
activations_right = get_neuron_activation(engine.model, modified_inputs)

# get monocular left response
modified_inputs = replace_inputs(inputs, gray, closed_eye="right")
activations_left = get_neuron_activation(engine.model, modified_inputs)

# get binocular response
activations_bino = get_neuron_activation(engine.model, inputs)

# get baseline response (response to blank or gray stereo images)
modified_inputs = replace_inputs(inputs, gray, closed_eye="both")
activations_baseline = get_neuron_activation(engine.model, modified_inputs)

# get monocular sum response
activations_sum = {}
for layer_name, activation_left in activations_left.items():

    activations_sum[layer_name] = (
        activation_left
        + activations_right[layer_name]
        - activations_baseline[layer_name]
    )


for name, activation_right in activations_right.items():
    print(name, tuple(activation_right.shape))

import matplotlib.pyplot as plt

layer_name = "decoder.layer30.0"
activation_bino = activations_bino[layer_name]  # [B, n_channels, n_disp_channels, h, w]
activation_right = activations_right[
    layer_name
]  # [B, n_channels, n_disp_channels, h, w]
activation_left = activations_left[layer_name]  # [B, n_channels, n_disp_channels, h, w]
activation_sum = activations_sum[layer_name]
activation_baseline = activations_baseline[layer_name]

plt.imshow(activation_right[0, 0, 0])
plt.imshow(activation_left[0, 0, 0])
plt.imshow(activation_bino[0, 0, 0])
plt.imshow(activation_sum[0, 0, 0])

# identify strong and weak coupled neurons
# strongly-coupled neurons: binocular response > monocular_sum response
# weakly-coupled neurons: binocular response < monocular_sum response
# Hypothesis: weakly-coupled neurons should carry more information about monocular representation.

# strongly-coupled neurons
plt.imshow(activation_bino[1, 1, 0] > activation_sum[1, 1, 0])
plt.title(f"Strongly-coupled neurons at {layer_name}: bino > mono_sum")

# weakly-coupled neurons
plt.imshow(activation_bino[1, 1, 0] < activation_sum[1, 1, 0])
plt.title(f"Weakly-coupled neurons at {layer_name}: bino < mono_sum")

plt.hist(activation_bino[0, 0, 0].flatten())
plt.hist(activation_sum[0, 0, 0].flatten())
