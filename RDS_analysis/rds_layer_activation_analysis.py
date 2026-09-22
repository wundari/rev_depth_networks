# %% load necessary modules
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor

from RDS_analysis.rds_analysis_v2 import RDSAnalysis, _generate_rds_condition
from RDS.DataHandler_RDS import DatasetRDS

from utilities.utils import *
from utilities.misc import NestedTensor

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearnex import patch_sklearn

patch_sklearn(verbose=False)
from sklearnex.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jaxtyping import Float
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
        self, dotMatch: float, dotDens: float, background_flag: bool
    ) -> None:
        """
        compute the layer activations in response to RDSs for a given dot match
        and dot density.

        Args:
            dotMatch (float): dot match level; between 0 (ards) to 1(crds)
            dotDens (float): dot density level; between 0.1 to 0.9
            background_flag ([binary 1/0]): a binary flag indicating
                    whether the RDS is surrounded by cRDS background (1) or not (0)
        """

        # create dataloader for RDS
        rds_loader = self._generate_rds_loader(
            dotMatch, dotDens, background_flag, self.pedestal_flag
        )

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
        #     "layer19": np.empty((n_samples, 128, 256), dtype=np.float32),
        #     "layer20": np.empty((n_samples, 128, 256), dtype=np.float32),
        #     "layer21": np.empty((n_samples, 64, 128), dtype=np.float32),
        #     "layer22": np.empty((n_samples, 64, 128), dtype=np.float32),
        #     "layer23": np.empty((n_samples, 64, 128), dtype=np.float32),
        #     "layer24": np.empty((n_samples, 32, 64), dtype=np.float32),
        #     "layer25": np.empty((n_samples, 32, 64), dtype=np.float32),
        #     "layer26": np.empty((n_samples, 32, 64), dtype=np.float32),
        #     "layer27": np.empty((n_samples, 16, 32), dtype=np.float32),
        #     "layer28": np.empty((n_samples, 16, 32), dtype=np.float32),
        #     "layer29": np.empty((n_samples, 16, 32), dtype=np.float32),
        #     "layer30": np.empty((n_samples, 8, 16), dtype=np.float32),
        #     "layer31": np.empty((n_samples, 8, 16), dtype=np.float32),
        #     "layer32": np.empty((n_samples, 8, 16), dtype=np.float32),
        #     "layer33a": np.empty((n_samples, 16, 32), dtype=np.float32),
        #     "layer34a": np.empty((n_samples, 32, 64), dtype=np.float32),
        #     "layer35a": np.empty((n_samples, 64, 128), dtype=np.float32),
        #     "layer36a": np.empty((n_samples, 128, 256), dtype=np.float32),
        #     "layer37": np.empty((n_samples, 256, 512), dtype=np.float32),
        # }

        # pre-allocate target disparity label
        n_samples = 2 * self.n_rds_each_disp
        disp_labels = np.empty(n_samples, dtype=np.int8)

        # iterate through the data and compute activations
        features = {}
        tepoch = tqdm(rds_loader, desc=f"RDS dotMatch= {dotMatch}, dotDens= {dotDens}")
        for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):
            # (inputs_left, inputs_right, disps) = next(iter(rds_loader))

            id_start = i * self.batch_size_rds
            id_end = id_start + self.batch_size_rds

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

            # compute activation for whole layers
            captured = self.compute_layer_activations(
                input_data, self.target_list, pooled=True, include_prediction=True
            )
            # with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            #     module_outputs = self.compute_layer_activations(
            #         input_data, self.target_list
            #     )

            # fetching each layer activation
            # for i, layer in enumerate(self.target_list):
            for name, module in zip(self.layer_name, self.target_list, strict=True):
                act = captured[module].numpy()
                if not np.isfinite(act).all():
                    raise ValueError(f"Nonfinite activations at {name}")
                if name not in features:
                    features[name] = np.empty(
                        (n_samples, act.shape[1]), dtype=np.float32
                    )
                features[name][id_start:id_end] = act

            # save target disparity label
            disp_labels[id_start:id_end] = disps.cpu().numpy()

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

            features["_metadata"] = self._metadata()

        # save file
        # for layer in self.layer_name:
        suffix = f"dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy"
        np.save(
            f"{self.layer_act_dir}/act_rds_{suffix}",
            features,
        )

        np.save(
            f"{self.layer_act_dir}/targetDisp_rds_{suffix}",
            disp_labels,
        )

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
    #     n_samples = (
    #         2 * self.n_rds_each_disp
    #     )  # number of rds in total, the "2" comes from near and far disp
    #     n_train = int(split_train * n_samples)  # number of training dataset

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
    #         crds_avg = layer_act_crds[layer].mean(axis=-2)  # [n_samples, w]
    #         ards_avg = layer_act_ards[layer].mean(axis=-2)  # [n_samples, w]
    #         hmrds_avg = layer_act_hmrds[layer].mean(axis=-2)  # [n_samples, w]

    #         # compute global mean and std for normalization across batch with crds
    #         x_mean = crds_avg.mean()  # axis=0, keepdims=True)  # [1, w]
    #         x_std = crds_avg.std()  # axis=0, keepdims=True)  # [1, w]
    #         # prevent division by zero
    #         # x_std[x_std == 0] = 1e-6

    #         # standardize/normalize activations across samples
    #         crds_norm = (crds_avg - x_mean) / x_std  # [n_samples, w]
    #         ards_norm = (ards_avg - x_mean) / x_std  # [n_samples, w]
    #         hmrds_norm = (hmrds_avg - x_mean) / x_std  # [n_samples, w]

    #         # bootstrap loop
    #         for i_bootstrap in range(n_bootstrap):

    #             print(
    #                 f"Cross-decoding for rds dotDens {dotDens:.2f}, "
    #                 + f"{layer}, bootstrap: {i_bootstrap}/{n_bootstrap}"
    #             )

    #             # generate random numbers for splitting train and test dataset
    #             idx = np.random.permutation(n_samples)
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

    def compute_cosine_similarity(
        self,
        dotDens: float,
        split_train: float,
        n_bootstrap: int,
        split_seed: int = 3407,
    ):

        def delta_disparity(features, labels):
            near = features[labels > 0]  # .mean(axis=0)
            far = features[labels < 0]  # .mean(axis=0)
            return near - far

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
            "crds_ards": np.empty((n_bootstrap, len(self.layer_name)), np.float32),
            "crds_hmrds": np.empty((n_bootstrap, len(self.layer_name)), np.float32),
        }

        for layer_idx, layer_name in enumerate(
            tqdm(self.layer_name, desc="Cosine similarity")
        ):

            x = {condition: values[layer_name] for condition, values in data.items()}
            if any(
                value.ndim != 2
                or value.shape[0] != len(y)
                or not np.isfinite(value).all()
                for value in x.values()
            ):
                raise ValueError(f"Invalid feature arrays for {layer_name}")

            for repeat, (train, test) in enumerate(splits):

                feat_crds = x["crds"][train]
                feat_hmrds = x["hmrds"][train]
                feat_ards = x["ards"][train]

                delta_crds = delta_disparity(feat_crds, y[train])
                delta_ards = delta_disparity(feat_ards, y[train])
                delta_hmrds = delta_disparity(feat_hmrds, y[train])

                scores["crds_ards"][repeat, layer_idx] = cosine_similarity(
                    delta_crds, delta_ards
                ).mean()
                scores["crds_hmrds"][repeat, layer_idx] = cosine_similarity(
                    delta_crds, delta_hmrds
                ).mean()

        for condition, score in scores.items():
            np.save(
                Path(self.layer_act_dir)
                / f"cosineSim_score_{condition}_dotDens_{dotDens:.2f}_bootstrap.npy",
                score,
            )

        return cosine_sim_crds_ards, cosine_sim_crds_hmrds

    def _split_groups(self, n_samples):
        """Keep shared generation trials AND inference batches in one fold.

        BatchNorm uses current batch statistics in this repository, even in eval.
        Splitting images from one inference batch between folds leaks context.
        """
        parent = np.arange(n_samples)

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def join(i, j):
            parent[root(i)] = root(j)

        for i in range(n_samples):
            join(i, (i // self.batch_size_rds) * self.batch_size_rds)
            join(i, i % self.n_rds_each_disp)
        return np.array([root(i) for i in range(n_samples)])

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
            fig.savefig(folder / filename, bbox_inches="tight")

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
