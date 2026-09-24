# %% load necessary modules
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
from torch import Tensor

from RDS_analysis.rds_analysis_v2 import (
    RDSAnalysis,
    _generate_rds_condition,
)
from RDS.DataHandler_RDS import DatasetRDS

from utilities.utils import *
from utilities.misc import NestedTensor

import os
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearnex import patch_sklearn

patch_sklearn(verbose=False)
from sklearnex.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import sem

from jaxtyping import Float, Bool
from config.config_bnn import ConfigBNN
from config.config_gcnet import ConfigGCNet


# %%
class RDS_LayerAct(RDSAnalysis):

    def __init__(self, config: ConfigBNN | ConfigGCNet) -> None:

        super().__init__(config)

        self.rds_bank_seed = 3407
        self.pool_shape = tuple(
            (1, 1)
        )  # spatial average across [h, w] in each DNN layer

        # BNN layer dimensions
        # | Layer(s)            | Shape                     |
        # | encoder.in_conv[0]  | [B, 32, 128, 256) — 4D [B, feat_channel, h, w]
        # | encoder.layer2[0]   | [B, 32, 128, 256) — 4D [B, feat_channel, h, w]
        # | decoder.layer3[0]   | [B, 32, 96, 128, 256) — 5D [B, feat_channel, disp_channel, h, w]
        # | decoder.layer4	    | [B, 1, 192, 256, 512) — 5D [B, feat_channel, disp_channel, h, w]

        # GCNet layer dimensions
        # | Layer(s)             | Shape [B, feat_channel, disp_channel, h, w] |
        # | -------------------- | ----------------------- |
        # | Cost volume          | `[B, 64, 96, 128, 256]` |
        # | 19–20                | `[B, 32, 96, 128, 256]` |
        # | 21–23                | `[B, 64, 48, 64, 128]`  |
        # | 24–26                | `[B, 64, 24, 32, 64]`   |
        # | 27–29                | `[B, 64, 12, 16, 32]`   |
        # | 30–32                | `[B, 128, 6, 8, 16]`    |
        # | 33a                  | `[B, 64, 12, 16, 32]`   |
        # | 34a                  | `[B, 64, 24, 32, 64]`   |
        # | 35a                  | `[B, 64, 48, 64, 128]`  |
        # | 36a                  | `[B, 32, 96, 128, 256]` |
        # | 37, before squeezing | `[B, 1, 192, 256, 512]` |

        # reset target layer names, important for hooking
        if self.model_name == "BNN":
            self.target_list = [
                self.model.encoder.in_conv[0],
                self.model.encoder.layer2[0],
                self.model.decoder.layer3[0],
                self.model.decoder.layer4,
            ]
            self.layer_name = ["encoder1", "encoder2", "layer3", "layer4"]

        elif self.model_name == "GC_Net":
            # self.target_list = [
            #     self.model.decoder.layer19[0],
            #     self.model.decoder.layer20[0],
            #     self.model.decoder.layer21[0],
            #     self.model.decoder.layer22[0],
            #     self.model.decoder.layer23[0],
            #     self.model.decoder.layer24[0],
            #     self.model.decoder.layer25[0],
            #     self.model.decoder.layer26[0],
            #     self.model.decoder.layer27[0],
            #     self.model.decoder.layer28[0],
            #     self.model.decoder.layer29[0],
            #     self.model.decoder.layer30[0],
            #     self.model.decoder.layer31[0],
            #     self.model.decoder.layer32[0],
            #     self.model.decoder.layer33a[0],
            #     self.model.decoder.layer34a[0],
            #     self.model.decoder.layer35a[0],
            #     self.model.decoder.layer36a[0],
            #     self.model.decoder.layer37,
            # ]
            self.layer_name = [
                *(f"layer{i}" for i in range(19, 33)),
                *(f"layer{i}a" for i in range(33, 37)),
                "layer37",
            ]

            self.target_list = [
                (
                    getattr(self.model.decoder, name)[0]
                    if name != "layer37"
                    else self.model.decoder.layer37
                )
                for name in self.layer_name
            ]

        self._create_layer_act_dir()

    def _create_layer_act_dir(self):

        # create folders for layer activations
        self.layer_act_dir = os.path.join(
            self.experiment_dir, f"layer_act_analysis_iter_{self.config.iter_to_load}"
        )
        if not os.path.exists(self.layer_act_dir):
            os.makedirs(self.layer_act_dir)

    def update_network_config(
        self, interaction: str, seed: int, epoch: int, iter: int
    ) -> None:
        """
        Update the network configuration and directories for storing
        the results
        """

        # old config, for printing purposes
        interaction_old = self.binocular_interaction
        seed_old = self.seed
        epoch_old = self.epoch
        iter_old = self.iter
        # batch_size_rds_old = self.batch_size_rds

        # update binocular_interaction, seed, epoch, iter, and model_pretrained in
        # the class and config
        self.binocular_interaction = interaction
        self.config.binocular_interaction = interaction
        self.seed = seed
        self.config.seed = seed
        self.config.experiment_id = seed
        self.epoch = epoch
        self.config.epoch_to_load = epoch
        self.iter = iter
        self.config.iter_to_load = iter
        self.model_pretrained = f"epoch_{self.epoch}_iter_{self.iter}_model_best.pth.tar"  # pretrained file name, e.g: epoch_1_model.pth.tar
        self.config.model_pretrained = self.model_pretrained

        # update the experiment directories based on the new interaction
        self.experiment_dir = (
            f"{self.model_name}/run/{self.dataset}/"
            + f"bino_interaction_{self.binocular_interaction}/"
            + f"experiment_{self.seed}"
        )

        # update layer_act dir
        self._create_layer_act_dir()

        # update directory for storing plots of a given interaction
        # (average across seeds)
        self.plot_dir = f"{self.experiment_dir}/../plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # update folders for rds analysis
        self.make_rds_dirs()

        print(
            "==============================================================\n"
            + f"Updating {self.model_name} config:\n"
            + "==============================================================\n"
            + f"Binocular interaction: {interaction_old} => {self.config.binocular_interaction}\n"
            + f"Seed: {seed_old} => {self.config.seed}\n"
            + f"Epoch: {epoch_old} => {self.config.epoch_to_load}\n"
            + f"Iter: {iter_old} => {self.config.iter_to_load}\n"
            + f"Experiment directory: {self.experiment_dir}\n"
            + f"RDS directory: {self.rds_dir}\n"
            + f"Cosine-similarity directory: {self.layer_act_dir}\n"
            + "==============================================================\n"
        )

    def create_disp_indices(
        self, n_disp_channel: int
    ) -> Float[Tensor, "batch disp_channel h w"]:

        # disp_indices = (
        #     torch.arange(-n_disp_channel // 2, n_disp_channel // 2)
        #     .view(1, -1, 1, 1)
        #     .pin_memory()
        #     .to(self.config.device, non_blocking=True)
        # )
        disp_indices = torch.linspace(
            -1, 1, n_disp_channel, device=self.device, dtype=torch.float32
        ).view(1, -1, 1, 1)

        return disp_indices

    def _pool_activation(self, value):
        """Spatial pooling only, followed by flattening into probe features.

        5D: [B,C,D,H,W] -> [B, C, D, h_pool, w_pool] -> [B, C*D*h_pool*w_pool]
        4D: [B,C,H,W]   -> [B, C, h_pool, w_pool]   -> [B, C*h_pool*w_pool]
        3D: [B,H,W]     -> [B, 1, h_pool, w_pool]   -> [B, h_pool*w_pool] (prediction)

        pool_shape = (1, 1) gives spatial means. Optional fractional ROI bounds
        select activation positions, not a restriction on their receptive fields.
        """
        if value.ndim not in (3, 4, 5):
            raise ValueError(f"Unexpected activation shape: {tuple(value.shape)}")
        # if self.roi is not None:
        #     height, width = value.shape[-2:]
        #     y0, y1, x0, x1 = self.roi
        #     # Include positions overlapping the ROI; keep at least one bin.
        #     top, bottom = int(np.floor(y0 * height)), int(np.ceil(y1 * height))
        #     left, right = int(np.floor(x0 * width)), int(np.ceil(x1 * width))
        #     value = value[..., top:bottom, left:right]

        size = tuple(
            min(n, limit) for n, limit in zip(value.shape[-2:], self.pool_shape)
        )

        # Pool before copying to CPU; never retain all full GPU cost volumes.
        if value.ndim == 5:
            value = F.adaptive_avg_pool3d(value, (value.shape[2], *size))

        elif value.ndim == 4:
            value = F.adaptive_avg_pool2d(value, size)

        elif value.ndim == 3:  # Actual predicted disparity, N x H x W.
            value = F.adaptive_avg_pool2d(value.unsqueeze(1), size)

        return value.flatten(1)

    @torch.inference_mode()
    def compute_layer_activations(
        self,
        input_data: NestedTensor,
        target: nn.Module,
        pooled: bool = False,
        include_prediction: bool = False,
    ) -> dict:
        """
        compute layer activation in respond to left and right inputs.

        Args:
            image_left_gpu (_type_): _description_
            image_right_gpu (_type_): _description_
            target (nn.Module): the target layer to be computed.
                it can be either in this form:
                - target = model.layer35a
                - target = model.layer35a[0] # the convolutional output
                - target = [model.layer35a]
                - target = [model.layer19, model.layer20, ...] # many layers

        Returns:
            module_outputs (list): a list containing the target layer
                    activations.

        """

        targets = list(target) if isinstance(target, (list, tuple)) else [target]
        outputs = {module: [] for module in targets}
        handles = []
        modes = {module: module.training for module in self.model.modules()}

        def capture(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("Select a module whose output is a tensor")
            value = (
                self._pool_activation(output.detach()) if pooled else output.detach()
            )
            # copy now, before a later in-place ReLU can change it.
            outputs[module].append(
                value.to(device="cpu", dtype=torch.float32, copy=True)
            )

        try:
            for module in targets:
                handles.append(module.register_forward_hook(capture))

            self.model.eval()
            device_type = torch.device(self.device).type
            dtype = getattr(torch, self.config.amp_dtype)
            with torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=device_type == "cuda" and dtype != torch.float32,
            ):
                prediction = self.model(input_data)

            result = {}
            for module, calls in outputs.items():
                if not calls:
                    raise RuntimeError(
                        "A selected hook did not fire; rebuild targets after changing models"
                    )
                result[module] = torch.cat(calls, dim=1) if len(calls) > 1 else calls[0]
                # result[module] = calls[0]

            if include_prediction:
                value = self._pool_activation(prediction) if pooled else prediction
                result["prediction"] = value.to(
                    device="cpu", dtype=torch.float32, copy=True
                )
            return result

        finally:
            for handle in handles:
                handle.remove()
            for module, mode in modes.items():
                module.training = mode

    def _generate_rds_loader(
        self,
        dotMatch,
        dotDens,
        background_flag,
        pedestal_flag,
        rds_bank=None,
    ):
        print(
            "==============================================================\n"
            + "Generate RDS dataloader: \n"
            + "==============================================================\n"
            + f"Dot Match: {dotMatch}\n"
            + f"Dot Density: {dotDens}\n"
            + f"Use RDS background: {background_flag}\n"
            + f"Use pedestal: {pedestal_flag}\n"
            + "==============================================================\n"
        )

        # reuse the bank's condition dataset, or use exactly its seed
        # derivation and per-trial generation. No shared global RNG or worker seed.
        if rds_bank is not None:
            bank = rds_bank.dataset
            if bank.bank_seed != self.rds_bank_seed:
                raise ValueError("Bank seed differs from the analysis bank_seed")
            matches = np.flatnonzero(np.isclose(bank.dot_matches, dotMatch))
            densities = np.flatnonzero(np.isclose(bank.dot_densities, dotDens))
            if len(matches) != 1 or len(densities) != 1:
                raise ValueError("Condition absent or ambiguous in supplied RDS bank")
            index = int(matches[0]) * len(bank.dot_densities) + int(densities[0])
            dataset = bank.datasets[index]
            # Existing banks have no background/pedestal metadata. The caller
            # must supply a bank generated with the same flags and normalization.
        else:
            seed = np.random.SeedSequence(
                [
                    self.rds_bank_seed,
                    int(round((dotMatch + 1) * 10_000)),
                    int(round(dotDens * 10_000)),
                ]
            )
            seeds = [
                int(child.generate_state(1, dtype=np.uint32)[0])
                for child in seed.spawn(self.n_rds_each_disp)
            ]
            left, right, labels = _generate_rds_condition(
                dotMatch,
                dotDens,
                self.disp_ct_pix_list,
                seeds,
                background_flag,
                pedestal_flag,
            )
            dataset = DatasetRDS(left, right, labels, transform=self.transform_data)

        expected = np.repeat(self.disp_ct_pix_list, self.n_rds_each_disp)
        if not np.array_equal(np.asarray(dataset.rds_label), expected):
            raise ValueError(
                "Expected disparity-major labels and configured sample counts"
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size_rds,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=torch.device(self.device).type == "cuda",
            generator=torch.Generator().manual_seed(self.rds_bank_seed),
        )

    def _configure_activation_bank(
        self,
        rds_bank: DataLoader,
        background_flag=None,
        pedestal_flag=None,
    ):
        """
        Validate the sequential bank and select an isolated cache directory.
        Images must use the same normalization as create_rds_bank().
        """

        if not isinstance(rds_bank, DataLoader):
            raise TypeError(
                "rds_bank must be the DataLoader returned by create_rds_bank"
            )
        if (
            type(rds_bank.sampler) is not SequentialSampler
            or type(rds_bank.batch_sampler) is not BatchSampler
            or rds_bank.drop_last
            or rds_bank.batch_size is None
        ):
            raise ValueError(
                "Bank requires sequential, fixed-size batches and drop_last=False"
            )

        dataset = rds_bank.dataset
        dotMatch_list = tuple(float(x) for x in self.dotMatch_list)
        dotDens_list = tuple(float(x) for x in self.dotDens_list)
        conditions = [(m, d) for m in dotMatch_list for d in dotDens_list]
        expected = np.repeat(self.disp_ct_pix_list, self.n_rds_each_disp)
        n_samples_per_cond = len(
            expected
        )  # the number of RDSs for each dotMatch and dotDens
        # = len(self.disp_ct_pix_list) * self.n_rds_each_disp

        if (
            not conditions
            or tuple(dataset.condition_shape)
            != (len(dotMatch_list), len(dotDens_list), n_samples_per_cond)
            or len(dataset.datasets) != len(conditions)
            or len(dataset) != n_samples_per_cond * len(conditions)
        ):
            raise ValueError("Bank layout/sample counts do not match this analysis")

        names = [f"{m:.2f}_{d:.2f}" for m, d in conditions]
        if len(set(names)) != len(names) or not np.isfinite(conditions).all():
            raise ValueError(
                "Condition values must be finite and distinct at two decimal places"
            )

        for data in dataset.datasets:
            if len(data) != n_samples_per_cond or not np.array_equal(
                data.rds_label, expected
            ):
                raise ValueError(
                    "Each condition must have the configured disparity-major labels"
                )

        flags = {}
        for name, supplied in (
            ("background_flag", background_flag),
            ("pedestal_flag", pedestal_flag),
        ):
            stored = getattr(dataset, name, None)
            if stored is None and supplied is None:
                raise ValueError(f"Older bank lacks {name}; supply it explicitly")
            if (
                stored is not None
                and supplied is not None
                and bool(stored) != bool(supplied)
            ):
                raise ValueError(f"Supplied {name} disagrees with the bank")
            flags[name] = bool(stored if stored is not None else supplied)

        self.background_flag = flags["background_flag"]
        self.pedestal_flag = flags["pedestal_flag"]
        self.rds_bank_seed = int(dataset.bank_seed)
        self._bank_context = dict(
            conditions=conditions,
            samples_per_condition=n_samples_per_cond,
            batch_size=self.batch_size_rds,
            reference=1,
        )

        # Keep fixed-reference bank outputs separate from legacy label-based swaps.
        if not hasattr(self, "_bank_output_root"):
            self._bank_output_root = Path(self.layer_act_dir)
        digest = hashlib.sha256(
            json.dumps(
                self._metadata(),
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:16]

        self.layer_act_dir = str(self._bank_output_root / f"rds_bank_{digest}")
        Path(self.layer_act_dir).mkdir(parents=True, exist_ok=True)

        return conditions, n_samples_per_cond

    def _metadata(self):
        return dict(
            schema=2,
            representation="pooled_convolution_outputs",
            layer_names=self.layer_name,
            pool_shape=self.pool_shape,
            bank_seed=self.rds_bank_seed,
            background=self.background_flag,
            pedestal=self.pedestal_flag,
            batch_size=self.batch_size_rds,
            amp_dtype=self.config.amp_dtype,
            disparities=list(self.disp_ct_pix_list),
            samples_per_disparity=self.n_rds_each_disp,
            checkpoint=self.config.model_pretrained,
        )

    @torch.inference_mode()
    def compute_layer_act_rds(
        self,
        rds_bank: DataLoader,
        background_flag: bool = True,
        pedestal_flag: bool = False,
    ) -> None:
        """
        compute the layer activations in response to RDSs for a given dot match
        and dot density.

        Args:
            dotMatch (float): dot match level; between 0 (ards) to 1 (crds)
            dotDens (float): dot density level; between 0.1 to 0.9
            background_flag ([binary 1/0]): a binary flag indicating
                    whether the RDS is surrounded by cRDS background (1) or not (0)
        """

        # BNN layer dimensions
        # | Layer(s)            | Shape                     |
        # | encoder.in_conv[0]  | [B, 32, 128, 256) — 4D [B, feat_channel, h, w]
        # | encoder.layer2[0]   | [B, 32, 128, 256) — 4D [B, feat_channel, h, w]
        # | decoder.layer3[0]   | [B, 32, 96, 128, 256) — 5D [B, feat_channel, disp_channel, h, w]
        # | decoder.layer4	    | [B, 1, 192, 256, 512) — 5D [B, feat_channel, disp_channel, h, w]

        # GCNet layer dimensions
        # | Layer(s)             | Shape [B, feat_channel, disp_channel, h, w] |
        # | -------------------- | ----------------------- |
        # | Cost volume          | `[B, 64, 96, 128, 256]` |
        # | 19–20                | `[B, 32, 96, 128, 256]` |
        # | 21–23                | `[B, 64, 48, 64, 128]`  |
        # | 24–26                | `[B, 64, 24, 32, 64]`   |
        # | 27–29                | `[B, 64, 12, 16, 32]`   |
        # | 30–32                | `[B, 128, 6, 8, 16]`    |
        # | 33a                  | `[B, 64, 12, 16, 32]`   |
        # | 34a                  | `[B, 64, 24, 32, 64]`   |
        # | 35a                  | `[B, 64, 48, 64, 128]`  |
        # | 36a                  | `[B, 32, 96, 128, 256]` |
        # | 37, before squeezing | `[B, 1, 192, 256, 512]` |

        # layer_act_dict = {
        #     "layer19": np.empty((n_samples_per_cond, 128, 256), dtype=np.float32),
        #     "layer20": np.empty((n_samples_per_cond, 128, 256), dtype=np.float32),
        #     "layer21": np.empty((n_samples_per_cond, 64, 128), dtype=np.float32),
        #     "layer22": np.empty((n_samples_per_cond, 64, 128), dtype=np.float32),
        #     "layer23": np.empty((n_samples_per_cond, 64, 128), dtype=np.float32),
        #     "layer24": np.empty((n_samples_per_cond, 32, 64), dtype=np.float32),
        #     "layer25": np.empty((n_samples_per_cond, 32, 64), dtype=np.float32),
        #     "layer26": np.empty((n_samples_per_cond, 32, 64), dtype=np.float32),
        #     "layer27": np.empty((n_samples_per_cond, 16, 32), dtype=np.float32),
        #     "layer28": np.empty((n_samples_per_cond, 16, 32), dtype=np.float32),
        #     "layer29": np.empty((n_samples_per_cond, 16, 32), dtype=np.float32),
        #     "layer30": np.empty((n_samples_per_cond, 8, 16), dtype=np.float32),
        #     "layer31": np.empty((n_samples_per_cond, 8, 16), dtype=np.float32),
        #     "layer32": np.empty((n_samples_per_cond, 8, 16), dtype=np.float32),
        #     "layer33a": np.empty((n_samples_per_cond, 16, 32), dtype=np.float32),
        #     "layer34a": np.empty((n_samples_per_cond, 32, 64), dtype=np.float32),
        #     "layer35a": np.empty((n_samples_per_cond, 64, 128), dtype=np.float32),
        #     "layer36a": np.empty((n_samples_per_cond, 128, 256), dtype=np.float32),
        #     "layer37": np.empty((n_samples_per_cond, 256, 512), dtype=np.float32),
        # }
        # n_samples_per_cond: the number of RDSs for each dotMatch and dotDens
        # = len(disp_ct_pix_list) * n_rds_each_disp

        # validate rds_bank ordering, labels, batch settings, and stimulus metadata.
        # Also selects the separate bank-analysis output directory.
        conditions, n_samples_per_cond = self._configure_activation_bank(
            rds_bank,
            background_flag=background_flag,
            pedestal_flag=pedestal_flag,
        )
        # conditions = [
        #     (dotMatch, dotDens)
        #     for dotMatch in self.dotMatch_list
        #     for dotDens in self.dotDens_list
        # ]
        # n_samples_per_cond = (
        #     len(self.disp_ct_pix_list) * self.n_rds_each_disp
        # )  # the number of RDSs for each dotMatch and dotDens

        if n_samples_per_cond % self.batch_size_rds != 0:
            raise ValueError(
                f"n_rds_each_disp ({self.n_rds_each_disp} must be divisible by RDS batch size {self.batch_size_rds}.\n"
                "Please change the batch size of RDS!"
            )

        if len(self.layer_name) != len(self.target_list):
            raise ValueError("layer_name and target_list must have equal lengths")

        # pre-allocate array
        expected_disp_labels = np.repeat(
            self.disp_ct_pix_list, self.n_rds_each_disp
        )  # for cross-checking the order of disparity label in the rds dataset
        disp_labels = np.empty(n_samples_per_cond, dtype=np.int8)
        total = len(rds_bank.dataset)
        features = {}
        # features = None
        paths = {}

        # iterate through the data and compute activations
        tepoch = tqdm(
            rds_bank,
            desc=f"Layer activation to RDSs",
        )
        count = 0
        global_count = 0
        for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):
            # (inputs_left, inputs_right, disps) = next(iter(rds_bank))

            # generate disparity direction
            ref = disps / 10.0

            # build nested tensor
            # input_data = NestedTensor(
            #     left=inputs_left.to(self.config.device, non_blocking=True),
            #     right=inputs_right.to(self.config.device, non_blocking=True),
            #     ref=ref.pin_memory().to(self.config.device, non_blocking=True),
            # )

            # swap left and right inputs if ref < 0
            if ref.mean() > 0:
                input_data = NestedTensor(
                    left=inputs_left.to(self.config.device, non_blocking=True),
                    right=inputs_right.to(self.config.device, non_blocking=True),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )
            else:
                input_data = NestedTensor(
                    left=inputs_right.to(self.config.device, non_blocking=True),
                    right=inputs_left.to(self.config.device, non_blocking=True),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )

            # compute activation for the whole layers
            captured = self.compute_layer_activations(
                input_data, self.target_list, pooled=True, include_prediction=True
            )

            # collect layer activation temporarily until it reaches n_samples_per_cond
            id_start = count
            id_end = id_start + self.batch_size_rds
            for layer_name, module in zip(
                self.layer_name, self.target_list, strict=True
            ):
                act = captured[module].numpy()

                # check values
                if not np.isfinite(act).all():
                    raise ValueError(f"Nonfinite activations at {layer_name}")

                # check dimensions
                if act.ndim != 2 or act.shape[0] != self.batch_size_rds:
                    raise ValueError(
                        f"Expected [B, n_feat] at {layer_name}; got {act.shape}"
                    )

                if layer_name not in features:
                    features[layer_name] = np.empty(
                        (n_samples_per_cond, act.shape[1]), dtype=np.float32
                    )
                features[layer_name][id_start:id_end] = act

            # save target disparity label
            disp_labels[id_start:id_end] = disps.cpu().numpy()

            # accumulate counter
            count += self.batch_size_rds
            global_count += self.batch_size_rds

            ## when buffer reaches n_samples_per_cond,
            # grouping layer activation according to dotMatch and dotDens then save
            if count % n_samples_per_cond == 0:

                # ensure the order of disparity labels
                if not np.array_equal(disp_labels, expected_disp_labels):
                    raise ValueError("Condition labels do not match the expected order")

                # safe
                rds_cond_idx = global_count // n_samples_per_cond - 1
                dotMatch, dotDens = conditions[rds_cond_idx]
                suffix = f"dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy"
                path = Path(self.layer_act_dir) / ("act_rds_" + suffix)
                features["_metadata"] = self._metadata()

                np.save(path, features)
                np.save(
                    Path(self.layer_act_dir) / ("targetDisp_rds_" + suffix), disp_labels
                )
                paths[(dotMatch, dotDens)] = path

                # reset buffer and counter
                features = {}
                disp_labels = np.empty(n_samples_per_cond, dtype=np.int8)
                count = 0

        if global_count != total:
            raise ValueError(f"Loader yielded {global_count} samples; expected {total}")

            # with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            # # get layer activation and average across feature channels
            # # [batch, feat_channel, disp_channel, h, w] => [batch, disp_channel, h, w]
            # out = module_outputs[layer].mean(dim=1)

            # # compute the prob (normalized across disp_channel)
            # out = F.softmax(-out, dim=1)  # [batch, disp_channel, h, w]

            # # create disparity indices tensor
            # n_disp_channel = out.shape[1]
            # disp_indices = self.create_disp_indices(n_disp_channel)

            # # compute the expected activation across disp channels
            # if i == len(self.target_list) - 1:
            #     layer_act = torch.sum(
            #         out * disp_indices * input_data.ref.view(-1, 1, 1, 1), dim=1
            #     )  # [batch, h, w]
            # else:
            #     layer_act = torch.sum(
            #         out * disp_indices, dim=1
            #     )  # [batch, h, w]

            # store in the dict
            # layer_act_dict[self.layer_name[l]][i] = layer_act.cpu().detach().numnpy()
            # layer_act_dict[self.layer_name[i]][id_start:id_end] = (
            #     layer_act.cpu().detach().numpy()
            # )

        #     features["_metadata"] = self._metadata()

        # # save file
        # # for layer in self.layer_name:
        # suffix = f"dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy"
        # np.save(
        #     f"{self.layer_act_dir}/act_rds_{suffix}",
        #     features,
        # )

        # np.save(
        #     f"{self.layer_act_dir}/targetDisp_rds_{suffix}",
        #     disp_labels,
        # )

    def compute_layer_act_rds_all(self, background_flag: bool) -> None:
        """
        compute layer activation for every rds types and dot density
        """

        for dotDens in self.dotDens_list:
            for dotMatch in self.dotMatch_list:

                print(
                    f"compute layer activation for RDS: "
                    + f"dotDens {dotDens:.2f}, "
                    + f"dotMatch {dotMatch:.2f}"
                )
                # compute layer activation for rds
                self.compute_layer_act_rds(dotMatch, dotDens, background_flag)

    def compute_layer_act_all_seeds(self, rds_bank: DataLoader):

        pass

    # def xDecode_layer_activation(
    #     self,
    #     dotDens: float,
    #     split_train: float,
    #     n_bootstrap: int,
    # ) -> None:
    #     """
    #     single bootstrap cross-decoding for a given dot density

    #     Args:
    #         dotDens (float): dot density
    #         split_train (float): proportion of training dataset
    #         n_bootstrap (int): the number of bootstrap iterations

    #     Returns:
    #         score_ards, score_hmrds, score_crds
    #             (n_bootstrap, len(self.layer_name))
    #             float32: cross-decoding score for each rds type
    #     """

    #     # dotDens = 0.4
    #     n_samples_per_cond = (
    #         2 * self.n_rds_each_disp
    #     )  # number of rds in total, the "2" comes from near and far disp
    #     n_train = int(split_train * n_samples_per_cond)  # number of training dataset

    #     # load layer activation data and target disparity label
    #     # ards
    #     dotMatch = 0.0
    #     layer_act_ards = np.load(
    #         f"{self.layer_act_dir}/act_rds"
    #         + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
    #         allow_pickle=True,
    #     ).item()
    #     targetDisp_ards = np.load(
    #         f"{self.layer_act_dir}/targetDisp_rds"
    #         + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
    #         allow_pickle=True,
    #     )

    #     # hmrds
    #     dotMatch = 0.5
    #     layer_act_hmrds = np.load(
    #         f"{self.layer_act_dir}/act_rds"
    #         + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
    #         allow_pickle=True,
    #     ).item()
    #     targetDisp_hmrds = np.load(
    #         f"{self.layer_act_dir}/targetDisp_rds"
    #         + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
    #         allow_pickle=True,
    #     )

    #     # crds
    #     dotMatch = 1.0
    #     layer_act_crds = np.load(
    #         f"{self.layer_act_dir}/act_rds"
    #         + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
    #         allow_pickle=True,
    #     ).item()
    #     # layer_act_crds = np.load(
    #     #     f"{rdsl.layer_act_dir}/act_rds"
    #     #     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
    #     # allow_pickle=True,
    #     # ).item()
    #     targetDisp_crds = np.load(
    #         f"{self.layer_act_dir}/targetDisp_rds"
    #         + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
    #         allow_pickle=True,
    #     )
    #     # targetDisp_crds = np.load(
    #     #     f"{rdsl.layer_act_dir}/targetDisp_rds"
    #     #     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
    #     # allow_pickle=True,
    #     # )

    #     score_ards = np.empty(
    #         (n_bootstrap, len(self.layer_name)),
    #         dtype=np.float32,
    #     )
    #     score_hmrds = np.empty(
    #         (n_bootstrap, len(self.layer_name)),
    #         dtype=np.float32,
    #     )
    #     score_crds = np.empty(
    #         (n_bootstrap, len(self.layer_name)),
    #         dtype=np.float32,
    #     )

    #     # loop over each DNN layer
    #     for i, layer in enumerate(self.layer_name):

    #         # Pre-compute row-averaged activations for each sample
    #         crds_avg = layer_act_crds[layer].mean(axis=-2)  # [n_samples_per_cond, w]
    #         ards_avg = layer_act_ards[layer].mean(axis=-2)  # [n_samples_per_cond, w]
    #         hmrds_avg = layer_act_hmrds[layer].mean(axis=-2)  # [n_samples_per_cond, w]

    #         # compute global mean and std for normalization across batch with crds
    #         x_mean = crds_avg.mean()  # axis=0, keepdims=True)  # [1, w]
    #         x_std = crds_avg.std()  # axis=0, keepdims=True)  # [1, w]
    #         # prevent division by zero
    #         # x_std[x_std == 0] = 1e-6

    #         # standardize/normalize activations across samples
    #         crds_norm = (crds_avg - x_mean) / x_std  # [n_samples_per_cond, w]
    #         ards_norm = (ards_avg - x_mean) / x_std  # [n_samples_per_cond, w]
    #         hmrds_norm = (hmrds_avg - x_mean) / x_std  # [n_samples_per_cond, w]

    #         # bootstrap loop
    #         for i_bootstrap in range(n_bootstrap):

    #             print(
    #                 f"Cross-decoding for rds dotDens {dotDens:.2f}, "
    #                 + f"{layer}, bootstrap: {i_bootstrap}/{n_bootstrap}"
    #             )

    #             # generate random numbers for splitting train and test dataset
    #             idx = np.random.permutation(n_samples_per_cond)
    #             idx_train = idx[:n_train]
    #             idx_test = idx[n_train:]

    #             # set up cRDS training dataset
    #             x_train = crds_norm[idx_train]  # [n_train, w]
    #             y_train = targetDisp_crds[idx_train]  # [n_train]

    #             # train classifier
    #             clf = SVC(kernel="linear", cache_size=1000)
    #             clf.fit(x_train, y_train)

    #             # evaluate on ards
    #             score_ards[i_bootstrap, i] = clf.score(ards_norm, targetDisp_ards)

    #             # evaluate on hmrds
    #             score_hmrds[i_bootstrap, i] = clf.score(hmrds_norm, targetDisp_hmrds)

    #             # evaluate on crds test subset
    #             x_test = crds_norm[idx_test]  # [n_test, w]
    #             y_test = targetDisp_crds[idx_test]  # [n_test]
    #             # fit
    #             score_crds[i_bootstrap, i] = clf.score(x_test, y_test)

    #             # clean up svm classifier
    #             del clf
    #         # gc.collect()

    #     # save xDecode
    #     np.save(
    #         f"{self.layer_act_dir}/xDecode_score_ards_dotDens_{dotDens:.2f}_bootstrap",
    #         score_ards,
    #     )
    #     np.save(
    #         f"{self.layer_act_dir}/xDecode_score_hmrds_dotDens_{dotDens:.2f}_bootstrap",
    #         score_hmrds,
    #     )
    #     np.save(
    #         f"{self.layer_act_dir}/xDecode_score_crds_dotDens_{dotDens:.2f}_bootstrap",
    #         score_crds,
    #     )

    @staticmethod
    def _cosine_sim_rows(
        disp_direction1: Float[np.ndarray, "B n_feat"],
        disp_direction2: Float[np.ndarray, "B n_feat"],
        tol: float,
    ) -> Float[np.ndarray, "B"]:
        """
        Row-wise cosine similarity; zero/near-zero axes have undefined direction (NaN).
        """

        first_norm = np.linalg.norm(disp_direction1, axis=1)
        second_norm = np.linalg.norm(disp_direction2, axis=1)
        valid = (first_norm > tol) & (second_norm > tol)
        scores = np.full(disp_direction1.shape[0], np.nan, dtype=np.float64)

        # Normalize before taking the dot product; avoid an all-pairs matrix.
        scores[valid] = np.einsum(
            "ij, ij -> i",
            disp_direction1[valid] / first_norm[valid, None],
            disp_direction2[valid] / second_norm[valid, None],
        )
        return np.clip(scores, -1.0, 1.0)

    @staticmethod
    def _contrast_weights(
        mask: Bool[np.ndarray, "n_bootstrap n_samples_per_cond"],
        label: Bool[np.ndarray, "n_samples_per_cond"],
    ) -> Float[np.ndarray, "n_bootstrap n_samples_per_cond"]:

        is_near = label > 0
        is_far = label < 0
        mask_near = is_near[None, :]
        mask_far = is_far[None, :]
        near = (mask & mask_near) / mask_near.sum(axis=1, keepdims=True)
        far = (mask & mask_far) / mask_far.sum(axis=1, keepdims=True)
        return near - far

    def compute_cosine_similarity(
        self,
        dotDens: float,
        split_train: float,
        n_bootstrap: int,
        split_seed: int = 3407,
    ):

        # load layer activation
        data, labels = {}, {}
        for condition, dotMatch in (("ards", 0.0), ("hmrds", 0.5), ("crds", 1.0)):
            suffix = f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy"
            data[condition] = np.load(
                Path(self.layer_act_dir) / ("act_rds" + suffix),
                allow_pickle=True,
            ).item()
            if data[condition].get("_metadata") != self._metadata():
                raise ValueError(
                    "Legacy or incompatible activation files; regenerate all conditions"
                )
            labels[condition] = np.load(
                Path(self.layer_act_dir) / ("targetDisp_rds" + suffix)
            )

        y = labels["crds"]
        expected = np.repeat(self.disp_ct_pix_list, self.n_rds_each_disp)
        if not np.array_equal(y, expected) or len(np.unique(y)) != 2:
            raise ValueError(
                "This near/far analysis requires exactly two disparity classes"
            )
        if not all(np.array_equal(y, value) for value in labels.values()):
            raise ValueError("Condition labels/order do not match")

        groups = self._split_groups(len(y))
        if len(np.unique(groups)) < 2:
            raise ValueError(
                "Too few independent groups; increase samples or reduce inference batch size"
            )
        splitter = GroupShuffleSplit(
            n_splits=n_bootstrap,
            train_size=split_train,
            random_state=split_seed,
        )

        # create near/far membership matrices [n_bootstrap, n_sample]
        # Shared by every layer: turns an (n_layers * n_bootstrap)-iteration
        mask_train = np.zeros((n_bootstrap, len(y)), dtype=bool)
        for repeat, (train, test) in enumerate(
            splitter.split(np.zeros(len(y)), y, groups)
        ):
            if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                raise ValueError("Both disparity classes must occur in each fold")

            mask_train[repeat, train] = True

        # array allocations
        conditions = ("crds", "hmrds", "ards")
        pairs = {
            "crds_ards": ("crds", "ards"),
            "crds_hmrds": ("crds", "hmrds"),
            "hmrds_ards": ("hmrds", "ards"),
        }
        weights_train = self._contrast_weights(
            mask_train, y
        )  # [n_bootstrap, n_samples_per_cond]
        weights_test = self._contrast_weights(
            ~mask_train, y
        )  # [n_bootstrap, n_samples_per_cond]
        shape = (n_bootstrap, len(self.layer_name))
        alignment = {pair: np.full(shape, np.nan, dtype=np.float32) for pair in pairs}
        reliability = {
            cond: np.full(shape, np.nan, dtype=np.float32) for cond in conditions
        }

        chunk_size = 128
        tol = 1e-12
        for layer_idx, layer_name in enumerate(
            tqdm(self.layer_name, desc="Cosine similarity")
        ):

            x = {condition: data[condition][layer_name] for condition in conditions}
            # sanity check
            for condition, value in x.items():
                if (
                    value.ndim != 2
                    or value.shape[0] != len(y)
                    or value.shape[1] == 0
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        f"Invalid feature arrays for {condition} at {layer_name}"
                    )

            if len({value.shape[1] for value in x.values()}) != 1:
                raise ValueError(
                    f"Feature dimensionality differs between conditions at {layer_name}"
                )

            # Float64 accumulation; subtract a shared reference row per
            # condition to reduce cancellation error. Contrast weights sum to
            # zero (+1 total on near rows, -1 total on far rows), so shifting
            # every row of a condition by the same constant vector does not
            # change the weighted near-minus-far result.
            centered = {}
            for condition, values in x.items():
                value = np.asarray(
                    values, dtype=np.float64
                )  # [n_samples_per_cond, n_feat]
                centered[condition] = value - value[0]

            for begin in range(0, n_bootstrap, chunk_size):
                # begin = 0
                end = min(begin + chunk_size, n_bootstrap)

                train_axes = {}
                for condition in conditions:

                    train_ax = (
                        weights_train[begin:end] @ centered[condition]
                    )  # [chunk_size, n_samples_per_cond] x [n_samples_per_cond, n_feat]
                    # = [chunck_size, n_feat], n_feat = n_feat_channel * n_disp_channel
                    test_ax = (
                        weights_test[begin:end] @ centered[condition]
                    )  # [chunk_size, n_samples_per_cond] x [n_samples_per_cond, n_feat]
                    # = [chunck_size, n_feat]
                    train_axes[condition] = train_ax

                    # compute reliability: cosine_similarity between training and test set
                    # cos_sim = _cosine_sim_rows(train_ax, test_ax, tol)
                    reliability[condition][begin:end, layer_idx] = (
                        self._cosine_sim_rows(train_ax, test_ax, tol)
                    )

                # compute alignment: cosine similarity between rds conditions
                for pair, (rds1, rds2) in pairs.items():
                    alignment[pair][begin:end, layer_idx] = self._cosine_sim_rows(
                        train_axes[rds1], train_axes[rds2], tol
                    )
            del centered

        for pair, score in alignment.items():
            np.save(
                Path(self.layer_act_dir)
                / f"cosineSim_align_{pair}_dotDens_{dotDens:.2f}_bootstrap.npy",
                score,
            )
        for condition, score in reliability.items():
            np.save(
                Path(self.layer_act_dir)
                / f"cosineSim_reliab_{condition}_dotDens_{dotDens:.2f}_bootstrap.npy",
                score,
            )

        return {"alignment": alignment, "reliability": reliability}

    def _load_cosine_simi_data(self, dotDens: float):

        rds_pairs = ("crds_ards", "crds_hmrds", "hmrds_ards")
        rds_conditions = ("crds", "hmrds", "ards")

        alignment = {
            pair: np.load(
                Path(self.layer_act_dir)
                / f"cosineSim_align_{pair}_dotDens_{dotDens:.2f}_bootstrap.npy"
            )
            for pair in rds_pairs
        }  # {rds_pair: [n_bootstrap, n_layer]}
        reliability = {
            condition: np.load(
                Path(self.layer_act_dir)
                / f"cosineSim_reliab_{condition}_dotDens_{dotDens:.2f}_bootstrap.npy"
            )
            for condition in rds_conditions
        }  # reliability: # {rds_cond: [n_bootstrap, n_layer]}

        if any(
            value.ndim != 2 or value.shape[1] != len(self.layer_name)
            for value in (*alignment.values(), *reliability.values())
        ):
            raise ValueError(
                "Cosine similarity score arrays do not match the selected layers"
            )

        return alignment, reliability

    @staticmethod
    def _cosine_simi_summary(
        similarity_score: Float[np.ndarray, "n_bootstrap n_layer"],
    ):
        """
        Mean, split SD, valid count; undefined angles remain missing.
        """

        valid = np.isfinite(similarity_score)
        count = valid.sum(axis=0)
        total = np.where(valid, similarity_score, 0.0).sum(axis=0)
        mean = np.divide(
            total,
            count,
            out=np.full(similarity_score.shape[1], np.nan),
            where=count > 0,
        )
        squared = np.where(valid, (similarity_score - mean) ** 2, 0.0).sum(axis=0)
        variance = np.divide(
            squared, count, out=np.full_like(mean, np.nan), where=count > 0
        )
        return mean, np.sqrt(variance), count

    def plot_cosine_similarity(self, save_flag=True):
        """
        Plot one dotDens or all configured densities

        """

        dotDens_list = list(self.dotDens_list)

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        n_row = len(dotDens_list)
        n_col = 2
        figsize = (8 * n_col, 4 * len(dotDens_list))
        fig, axes = plt.subplots(
            nrows=n_row,
            ncols=n_col,
            squeeze=False,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )
        fig.text(
            0.5,
            1.01,
            f"Cosine Similarity ({self.model_name} / {self.binocular_interaction})",
            ha="center",
        )
        fig.text(-0.05, 0.5, "Cosine similarity", va="center", rotation=90)
        fig.text(0.5, -0.02, "Layer", ha="center")
        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        colors = ["#6a5acd", "#00CED1", "#333333"]

        x = np.arange(len(self.layer_name))
        for row, dotDens in enumerate(dotDens_list):

            # load cosine similarity data
            alignment, reliability = self._load_cosine_simi_data(dotDens)
            # dotDens = 0.1
            # alignment, reliability = rdsl._load_cosine_simi_data(dotDens)

            # plot alignment
            valid_counts = []
            for i, (pair, simi_score) in enumerate(alignment.items()):

                mean, sd, valid_count = self._cosine_simi_summary(simi_score)

                # pair = "crds_ards"
                # simi_score = alignment[pair]
                # mean, sd, valid_count = rdsl._cosine_simi_summary(simi_score)
                valid_counts.extend(valid_count.tolist())

                ## plot the one standard deviation for rds1 vs rds2
                y = np.array(mean)
                axes[row, 0].plot(
                    x, y, linewidth=2, color=colors[i], label=pair.replace("_", " vs ")
                )
                axes[row, 0].plot(x, y, "o", markersize=8, color=colors[i])
                axes[row, 0].fill_between(
                    x,
                    y - sd,
                    y + sd,
                    color=colors[i],
                    alpha=0.2,
                )
                axes[row, 0].text(
                    0.01,
                    0.02,
                    f"Valid splits/layer: {min(valid_counts)}–{max(valid_counts)}/{self.n_bootstrap}",
                    transform=axes[row, 0].transAxes,
                    fontsize=8,
                )
                axes[row, 0].legend(fontsize=10, frameon=False)

                # plot midline
                axes[row, 0].axhline(0.0, color="red", linestyle="--", linewidth=2)

                # Hide the right and top spines
                axes[row, 0].spines["right"].set_visible(False)
                axes[row, 0].spines["top"].set_visible(False)

                # Only show ticks on the left and bottom spines
                axes[row, 0].yaxis.set_ticks_position("left")
                axes[row, 0].xaxis.set_ticks_position("bottom")

            # label x-axis
            axes[row, 0].set_xticks(x)
            axes[row, 0].set_xticklabels(x)

            # ylim
            axes[row, 0].set_ylim(-1, 1)

            # title
            axes[row, 0].set_title(f"Alignment: dot density {dotDens:.2f}")

            # plot reliability
            for i, (cond, reli_score) in enumerate(reliability.items()):

                mean, sd, valid_count = self._cosine_simi_summary(reli_score)
                # pair = "crds_ards"
                # simi_score = alignment[pair]
                # mean, sd, valid_count = rdsl._cosine_simi_summary(simi_score)
                valid_counts.extend(valid_count.tolist())

                ## plot the one standard deviation for rds1 vs rds2
                y = np.array(mean)
                axes[row, 1].plot(x, y, linewidth=2, color=colors[i], label=cond)
                axes[row, 1].plot(x, y, "o", markersize=8, color=colors[i])
                axes[row, 1].fill_between(
                    x,
                    y - sd,
                    y + sd,
                    color=colors[i],
                    alpha=0.2,
                )
                axes[row, 1].text(
                    0.01,
                    0.02,
                    f"Valid splits/layer: {min(valid_counts)}–{max(valid_counts)}/{self.n_bootstrap}",
                    transform=axes[row, 1].transAxes,
                    fontsize=8,
                )
                axes[row, 1].legend(fontsize=10, frameon=False)

                # plot midline
                axes[row, 1].axhline(0.0, color="red", linestyle="--", linewidth=2)

                # Hide the right and top spines
                axes[row, 1].spines["right"].set_visible(False)
                axes[row, 1].spines["top"].set_visible(False)

                # Only show ticks on the left and bottom spines
                axes[row, 1].yaxis.set_ticks_position("left")
                axes[row, 1].xaxis.set_ticks_position("bottom")

            # label x-axis
            axes[row, 1].set_xticks(x)
            axes[row, 1].set_xticklabels(x)

            # ylim
            axes[row, 1].set_ylim(-1, 1)

            # title
            axes[row, 1].set_title(f"Reliability: dot density {dotDens:.2f}")

        # fig.suptitle(f"{self.model_name}: near–far disparity axes")
        self._save_plot(fig, f"plot_cosine_similarity_all.pdf", save_flag)

    def _load_cosine_similarity_all_seeds(self, interaction: str):
        """
        Load cosine similarity scores for all seeds and average them across bootstrap

        Returns:
            _type_: _description_
        """

        # allocate dict
        rds_pairs = ("crds_ards", "crds_hmrds", "hmrds_ards")
        rds_conditions = ("crds", "hmrds", "ards")
        alignment_all_seeds = {
            pair: np.empty(
                (
                    len(self.config.seed_to_analyse),
                    len(self.config.dotDens_list),
                    len(self.layer_name),
                ),
                dtype=np.float32,
            )
            for pair in rds_pairs
        }
        reliability_all_seeds = {
            rds_cond: np.empty(
                (
                    len(self.config.seed_to_analyse),
                    len(self.config.dotDens_list),
                    len(self.layer_name),
                ),
                dtype=np.float32,
            )
            for rds_cond in rds_conditions
        }

        # gather cosine-similarity data all seeds
        for s, seed in enumerate(self.config.seed_to_analyse):

            if interaction == "default":
                epoch, iter = self.config.epoch_iter_to_load_default[s]
            elif interaction == "bem":
                epoch, iter = self.config.epoch_iter_to_load_bem[s]
            elif interaction == "cmm":
                epoch, iter = self.config.epoch_iter_to_load_cmm[s]
            else:  # sum_diff
                epoch, iter = self.config.epoch_iter_to_load_sum_diff[s]

            # update network configuration and directory addresses
            self.update_network_config(interaction, seed, epoch, iter)

            for dd, dotDens in enumerate(dotDens_list):

                # load cosine similarity data
                alignment, reliability = self._load_cosine_simi_data(dotDens)
                # alignment: {rds_pair: [n_bootstrap, n_layer]}
                # reliability: # {rds_cond: [n_bootstrap, n_layer]}

                # gather alignment
                for i, (rds_pair, simi_score) in enumerate(alignment.items()):

                    # average across bootstrap
                    mean, _, _ = self._cosine_simi_summary(simi_score)  # [n_layer]
                    alignment_all_seeds[rds_pair][s, dd] = mean

                # gather reliability
                for i, (rds_cond, reli_score) in enumerate(reliability.items()):

                    # average across bootstrap
                    mean, _, _ = self._cosine_simi_summary(reli_score)  # [n_layer]
                    reliability_all_seeds[rds_cond][s, dd] = mean

        return alignment_all_seeds, reliability_all_seeds

    def plot_cosine_similarity_all_seeds(
        self, interaction: str, save_flag: bool = True
    ):

        ## load cosine similarity scores all seeds
        # alignment_all_seeds = {rds_pair: [len(self.config.seed_to_analyse),
        #                                   len(self.config.dotDens_list),
        #                                   len(self.layer_name)]}
        # reliability_all_seeds = {rds_cond: [len(self.config.seed_to_analyse),
        #                                     len(self.config.dotDens_list),
        #                                     len(self.layer_name)]}
        alignment_all_seeds, reliability_all_seeds = (
            self._load_cosine_similarity_all_seeds(interaction)
        )
        rds_pairs = ("crds_ards", "crds_hmrds", "hmrds_ards")
        rds_conditions = ("crds", "hmrds", "ards")
        dotDens_list = list(self.dotDens_list)

        # average across seed_to_analyse
        alignment_avg = {
            pair: np.empty(
                (len(self.config.dotDens_list), len(self.layer_name)), dtype=np.float32
            )
            for pair in rds_pairs
        }
        reliability_avg = {
            cond: np.empty(
                (len(self.config.dotDens_list), len(self.layer_name)), dtype=np.float32
            )
            for cond in rds_conditions
        }
        alignment_sem = {
            pair: np.empty(
                (len(self.config.dotDens_list), len(self.layer_name)), dtype=np.float32
            )
            for pair in rds_pairs
        }
        reliability_sem = {
            cond: np.empty(
                (len(self.config.dotDens_list), len(self.layer_name)), dtype=np.float32
            )
            for cond in rds_conditions
        }
        for i in range(len(rds_pairs)):

            rds_pair = rds_pairs[i]
            alignment_avg[rds_pair] = alignment_all_seeds[rds_pair].mean(
                axis=0
            )  # [dotDens_list, layer_name]
            alignment_sem[rds_pair] = sem(
                alignment_all_seeds[rds_pair], axis=0
            )  # [dotDens_list, layer_name]

            rds_cond = rds_conditions[i]
            reliability_avg[rds_cond] = reliability_all_seeds[rds_cond].mean(
                axis=0
            )  # [dotDens_list, layer_name]
            reliability_sem[rds_cond] = sem(
                reliability_all_seeds[rds_cond], axis=0
            )  # [dotDens_list, layer_name]

        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

        n_row = len(dotDens_list)
        n_col = 2
        figsize = (8 * n_col, 4 * len(dotDens_list))
        fig, axes = plt.subplots(
            nrows=n_row,
            ncols=n_col,
            squeeze=False,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )
        fig.text(
            0.5,
            1.01,
            f"Cosine Similarity ({self.model_name} / {self.binocular_interaction})",
            ha="center",
        )
        fig.text(-0.05, 0.5, "Cosine similarity", va="center", rotation=90)
        fig.text(0.5, -0.02, "Layer", ha="center")
        fig.tight_layout()

        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        colors = ["#6a5acd", "#00CED1", "#333333"]

        x = np.arange(len(self.layer_name))
        for row, dotDens in enumerate(dotDens_list):

            # plot alignment
            for i, (pair, simi_score) in enumerate(alignment_avg.items()):

                ## plot the sem for rds1 vs rds2
                y = np.array(simi_score)
                y_sem = np.array(alignment_sem[pair])
                axes[row, 0].plot(
                    x, y, linewidth=2, color=colors[i], label=pair.replace("_", " vs ")
                )
                axes[row, 0].plot(x, y, "o", markersize=8, color=colors[i])
                axes[row, 0].fill_between(
                    x,
                    y - y_sem,
                    y + y_sem,
                    color=colors[i],
                    alpha=0.2,
                )

                axes[row, 0].legend(fontsize=10, frameon=False)

                # plot midline
                axes[row, 0].axhline(0.0, color="red", linestyle="--", linewidth=2)

                # Hide the right and top spines
                axes[row, 0].spines["right"].set_visible(False)
                axes[row, 0].spines["top"].set_visible(False)

                # Only show ticks on the left and bottom spines
                axes[row, 0].yaxis.set_ticks_position("left")
                axes[row, 0].xaxis.set_ticks_position("bottom")

            # label x-axis
            axes[row, 0].set_xticks(x)
            axes[row, 0].set_xticklabels(x)

            # title
            axes[row, 0].set_title(f"Alignment: dot density {dotDens:.2f}")

            # plot reliability
            for i, (cond, reli_score) in enumerate(reliability_avg.items()):

                ## plot the sem for rds1 vs rds2
                y = np.array(reli_score)
                y_sem = np.array(reliability_sem[cond])
                axes[row, 1].plot(x, y, linewidth=2, color=colors[i], label=cond)
                axes[row, 1].plot(x, y, "o", markersize=8, color=colors[i])
                axes[row, 1].fill_between(
                    x,
                    y - y_sem,
                    y + y_sem,
                    color=colors[i],
                    alpha=0.2,
                )

                axes[row, 1].legend(fontsize=10, frameon=False)

                # plot midline
                axes[row, 1].axhline(0.0, color="red", linestyle="--", linewidth=2)

                # Hide the right and top spines
                axes[row, 1].spines["right"].set_visible(False)
                axes[row, 1].spines["top"].set_visible(False)

                # Only show ticks on the left and bottom spines
                axes[row, 1].yaxis.set_ticks_position("left")
                axes[row, 1].xaxis.set_ticks_position("bottom")

            # label x-axis
            axes[row, 1].set_xticks(x)
            axes[row, 1].set_xticklabels(x)

            # title
            axes[row, 1].set_title(f"Reliability: dot density {dotDens:.2f}")

        # fig.suptitle(f"{self.model_name}: near–far disparity axes")
        self._save_plot(
            fig, f"plot_cosine_similarity_avg_seeds_{interaction}.pdf", save_flag
        )

    def _split_groups(self, n_samples_per_cond: int):
        """Keep shared generation trials AND inference batches in one fold.

        BatchNorm uses current batch statistics in this repository, even in eval.
        Splitting images from one inference batch between folds leaks context.
        """
        parent = np.arange(n_samples_per_cond)

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def join(i, j):
            parent[root(i)] = root(j)

        for i in range(n_samples_per_cond):
            join(i, (i // self.batch_size_rds) * self.batch_size_rds)
            join(i, i % self.n_rds_each_disp)
        return np.array([root(i) for i in range(n_samples_per_cond)])

    def xDecode_layer_activation(
        self,
        dotDens: float,
        split_train: float,
        n_bootstrap: int,
        split_seed: int = 3407,
        C: float = 1.0,
    ):
        """Repeated grouped holdout; legacy 'bootstrap' filenames are retained.

        Standardization is fitted only on cRDS training features. Test indices
        are shared across conditions and layers; no tuning on aRDS scores.
        """
        if not 0 < split_train < 1 or n_bootstrap < 1 or C <= 0:
            raise ValueError("Require 0 < split_train < 1, positive repeats and C")
        data, labels = {}, {}
        for condition, match in (("ards", 0.0), ("hmrds", 0.5), ("crds", 1.0)):
            suffix = f"_dotDens_{dotDens:.2f}_dotMatch_{match:.2f}.npy"
            data[condition] = np.load(
                Path(self.layer_act_dir) / ("act_rds" + suffix),
                allow_pickle=True,
            ).item()
            if data[condition].get("_metadata") != self._metadata():
                raise ValueError(
                    "Legacy or incompatible activation files; regenerate all conditions"
                )
            labels[condition] = np.load(
                Path(self.layer_act_dir) / ("targetDisp_rds" + suffix)
            )
        y = labels["crds"]
        expected = np.repeat(self.disp_ct_pix_list, self.n_rds_each_disp)
        if not np.array_equal(y, expected) or len(np.unique(y)) != 2:
            raise ValueError(
                "This near/far analysis requires exactly two disparity classes"
            )
        if not all(np.array_equal(y, value) for value in labels.values()):
            raise ValueError("Condition labels/order do not match")

        groups = self._split_groups(len(y))
        if len(np.unique(groups)) < 2:
            raise ValueError(
                "Too few independent groups; increase samples or reduce inference batch size"
            )
        splits = list(
            GroupShuffleSplit(
                n_splits=n_bootstrap,
                train_size=split_train,
                random_state=split_seed,
            ).split(np.zeros(len(y)), y, groups)
        )
        for train, test in splits:
            if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                raise ValueError("Both disparity classes must occur in each fold")
        scores = {
            condition: np.empty((n_bootstrap, len(self.layer_name)), np.float32)
            for condition in data
        }
        for layer_index, layer in enumerate(
            tqdm(self.layer_name, desc="Cross-decoding")
        ):
            x = {condition: values[layer] for condition, values in data.items()}
            if any(
                value.ndim != 2
                or value.shape[0] != len(y)
                or not np.isfinite(value).all()
                for value in x.values()
            ):
                raise ValueError(f"Invalid feature arrays for {layer}")
            for repeat, (train, test) in enumerate(splits):
                # train-only, per-feature scaling, including constant
                # features (StandardScaler handles zero variance safely).
                classifier = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C))
                classifier.fit(x["crds"][train], y[train])
                for condition in scores:
                    scores[condition][repeat, layer_index] = classifier.score(
                        x[condition][test],
                        labels[condition][test],
                    )
        for condition, score in scores.items():
            np.save(
                Path(self.layer_act_dir)
                / f"xDecode_score_{condition}_dotDens_{dotDens:.2f}_bootstrap.npy",
                score,
            )
        # Save settings and exact folds so a score is traceable.
        np.savez(
            Path(self.layer_act_dir) / f"xDecode_splits_dotDens_{dotDens:.2f}.npz",
            train=(
                np.stack([train for train, _ in splits])
                if len({len(t) for t, _ in splits}) == 1
                else np.array([t for t, _ in splits], dtype=object)
            ),
            test=(
                np.stack([test for _, test in splits])
                if len({len(t) for _, t in splits}) == 1
                else np.array([t for _, t in splits], dtype=object)
            ),
            split_seed=split_seed,
            split_train=split_train,
            C=C,
        )
        return scores

    def _load_scores(self, density):
        scores = {
            condition: np.load(
                Path(self.layer_act_dir)
                / f"xDecode_score_{condition}_dotDens_{density:.2f}_bootstrap.npy"
            )
            for condition in ("ards", "hmrds", "crds")
        }
        if any(
            value.ndim != 2 or value.shape[1] != len(self.layer_name)
            for value in scores.values()
        ):
            raise ValueError("Score arrays do not match the selected layers")
        return scores

    def _plot_density(self, ax, density):
        x = np.arange(len(self.layer_name))
        ax.axhline(0.5, color="black", linestyle="--", label="Chance")
        for condition, score in self._load_scores(density).items():
            ax.errorbar(x, score.mean(0), yerr=score.std(0), label=condition)
        ax.set(
            title=f"Dot density {density:.2f}",
            ylim=(0, 1),
            xticks=x,
            xticklabels=self.layer_name,
            ylabel="Accuracy",
        )
        ax.tick_params(axis="x", labelrotation=90)

    def _save_plot(self, fig, filename, save_flag):
        if save_flag:
            folder = Path(self.layer_act_dir) / "Plots"
            folder.mkdir(exist_ok=True)
            fig.savefig(folder / filename, dpi=600, bbox_inches="tight")

    def plotLine_xDecode_across_layers_at_dotDens(self, dotDens, save_flag):
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        self._plot_density(ax, dotDens)
        ax.legend()
        fig.suptitle(f"{self.model_name}: cRDS-trained probes (bars: split SD)")
        self._save_plot(fig, f"plotLine_xDecode_dotDens_{dotDens:.2f}.pdf", save_flag)
        return fig

    def plotLine_xDecode_across_layers(self, save_flag):
        # LAYER FIX: dynamic density count and layer labels, including BNN.
        n = len(self.dotDens_list)
        if n == 0:
            raise ValueError("No dot densities configured")
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7 * ncols, 5 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        for index, ax in enumerate(axes.flat):
            if index < n:
                self._plot_density(ax, self.dotDens_list[index])
            else:
                ax.set_visible(False)
        axes.flat[0].legend()
        fig.suptitle(f"{self.model_name}: cRDS-trained probes (bars: split SD)")
        self._save_plot(fig, "plotLine_xDecode.pdf", save_flag)
        return fig

    def plotHeat_xDecode(self, save_flag):
        fig, axes = plt.subplots(1, 3, figsize=(16, 8), constrained_layout=True)
        all_scores = [self._load_scores(density) for density in self.dotDens_list]
        for ax, condition in zip(axes, ("ards", "hmrds", "crds"), strict=True):
            values = np.stack(
                [score[condition].mean(0) for score in all_scores], axis=1
            )
            im = ax.imshow(values, vmin=0, vmax=1, cmap="coolwarm", aspect="auto")
            ax.set(
                title=f"cRDS → {condition}",
                xlabel="Dot density",
                xticks=np.arange(len(self.dotDens_list)),
                xticklabels=[f"{density:.2f}" for density in self.dotDens_list],
                yticks=np.arange(len(self.layer_name)),
                yticklabels=self.layer_name,
            )
            fig.colorbar(im, ax=ax)
        self._save_plot(fig, "plotHeatmap_xDecode.pdf", save_flag)
        return fig


# %%
