# %% load necessary modules
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch._dynamo
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor

import sys

# sys.path.append("engine")
from engine.engine_base import Engine
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS

# from GC_Net_v2 import *

from utilities.utils import *
from utilities.output_hook import ModuleOutputsHook
from utilities.misc import NestedTensor

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn import svm

from jaxtyping import Float
from config.config import GCNetconfig

# reproducibility
import random

seed_number = 3407  # 12321
torch.manual_seed(seed_number)
np.random.seed(seed_number)


# initialize random seed number for dataloader
def seed_worker(worker_id):
    worker_seed = seed_number  # torch.initial_seed()  % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # print out seed number for each worker
    # np_seed = np.random.get_state()[1][0]
    # py_seed = random.getstate()[1][0]

    # print(f"{worker_id} seed pytorch: {worker_seed}\n")
    # print(f"{worker_id} seed numpy: {np_seed}\n")
    # print(f"{worker_id} seed python: {py_seed}\n")


g = torch.Generator()
g.manual_seed(seed_number)


# %%
class RDS_LayerAct(Engine):

    def __init__(self, config: GCNetconfig, params_rds: dict) -> None:

        super().__init__(config)

        # rds parameters
        self.target_disp = params_rds[
            "target_disp"
        ]  # RDS target disparity (pix) to be analyzed
        self.n_rds_each_disp = params_rds[
            "n_rds_each_disp"
        ]  # n_rds for each disparity magnitude in disp_ct_pix
        self.dotDens_list = params_rds["dotDens_list"]  # dot density
        self.rds_type = params_rds["rds_type"]  # ards: 0, crds: 1, hmrds: 0.5, urds: -1
        self.batch_size = params_rds["batch_size_rds"]
        self.dotMatch_list = params_rds["dotMatch_list"]  # dot match
        self.disp_ct_pix_list = [
            self.target_disp,
            -self.target_disp,
        ]  # disparity magnitude: GC-Net (+ near, - far)
        self.background_flag = params_rds["background_flag"]
        self.pedestal_flag = params_rds[
            "pedestal_flag"
        ]  # 1: use pedestal to ensure rds disparity > 0
        self.n_bootstrap = params_rds["n_bootstrap"]

        # transform rds to tensor and in range [0, 1]
        # self.transform_data = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Lambda(lambda t: (t + 1.0) / 2.0)]
        # )
        # mean = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0)
        # std = (0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # mean = np.array([0.5, 0.5, 0.5])
        # std = np.array([0.5, 0.5, 0.5])
        self.transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t + 1.0) / 2.0),
                transforms.Normalize(mean, std),
            ]
        )

        # folders for rds analysis
        self.rds_dir = (
            f"{self.save_dir}/"
            + f"epoch_{self.config.epoch_to_load}"
            + f"_iter_{self.config.iter_to_load}"
            + f"/rds_analysis/target_disp_{self.target_disp}px"
        )
        if not os.path.exists(self.rds_dir):
            os.makedirs(self.rds_dir)

        # create folders for rds layer activation
        if self.pedestal_flag:
            self.layer_act_dir = (
                f"{self.rds_dir}/layer_activation_analysis_with_pedestal"
            )
        else:
            self.layer_act_dir = f"{self.rds_dir}/layer_activation_analysis_wo_pedestal"
        if not os.path.exists(self.layer_act_dir):
            os.mkdir(self.layer_act_dir)

        self.target_list = [
            self.model.decoder.layer19[0],
            self.model.decoder.layer20[0],
            self.model.decoder.layer21[0],
            self.model.decoder.layer22[0],
            self.model.decoder.layer23[0],
            self.model.decoder.layer24[0],
            self.model.decoder.layer25[0],
            self.model.decoder.layer26[0],
            self.model.decoder.layer27[0],
            self.model.decoder.layer28[0],
            self.model.decoder.layer29[0],
            self.model.decoder.layer30[0],
            self.model.decoder.layer31[0],
            self.model.decoder.layer32[0],
            self.model.decoder.layer33a[0],
            self.model.decoder.layer34a[0],
            self.model.decoder.layer35a[0],
            self.model.decoder.layer36a[0],
            self.model.decoder.layer37,
        ]

        self.layer_name = [
            "layer19",
            "layer20",
            "layer21",
            "layer22",
            "layer23",
            "layer24",
            "layer25",
            "layer26",
            "layer27",
            "layer28",
            "layer29",
            "layer30",
            "layer31",
            "layer32",
            "layer33a",
            "layer34a",
            "layer35a",
            "layer36a",
            "layer37",
        ]

    def create_disp_indices(
        self, n_disp_channel: int
    ) -> Float[Tensor, "batch disp_channel h w"]:

        # disp_indices = (
        #     torch.arange(-n_disp_channel // 2, n_disp_channel // 2)
        #     .view(1, -1, 1, 1)
        #     .pin_memory()
        #     .to(self.config.device, non_blocking=True)
        # )
        disp_indices = (
            torch.linspace(-1, 1, n_disp_channel)
            .view(1, -1, 1, 1)
            .pin_memory()
            .to(self.config.device, non_blocking=True)
        )

        return disp_indices

    def compute_layer_activations(
        self,
        input_data: NestedTensor,
        target: nn.Module,
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

        # get layer activation to get layer dimension
        # define hook
        if isinstance(target, list):
            hook = ModuleOutputsHook(target)
        else:
            hook = ModuleOutputsHook([target])
            # hook = ModuleOutputsHook([model.layer36a])

        # compute model's output.
        logits = self.model(input_data)

        # consume_outputs return the captured values and resets the hook's state
        # compute module output
        module_outputs = hook.consume_outputs()
        # activations = module_outputs[target]
        # activations = module_outputs[model.layer36a]

        hook.remove_hooks()

        return module_outputs

    def _generate_rds_loader(
        self,
        dotMatch: float,
        dotDens: float,
        background_flag: bool,
        pedestal_flag: bool,
    ) -> DataLoader:
        """
        generate RDS dataloader for a given dot match and dot density.
        """
        # dotMatch = 1.0
        # dotDens = 0.4
        # background_flag = 1

        # [len(disp_ct_pix) * batch_size, h, w, n_channels]
        rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
            dotMatch,
            dotDens,
            self.disp_ct_pix_list,
            self.n_rds_each_disp,
            background_flag,
            pedestal_flag,
        )

        rds_data = DatasetRDS(
            rds_left, rds_right, rds_label, transform=self.transform_data
        )

        rds_loader = DataLoader(
            rds_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=1,
            worker_init_fn=seed_worker,
            generator=g,
        )

        return rds_loader

    @torch.no_grad()
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

        n_samples = 2 * self.n_rds_each_disp
        # average across feature and disparity channels
        layer_act_dict = {
            "layer19": np.empty((n_samples, 128, 256), dtype=np.float32),
            "layer20": np.empty((n_samples, 128, 256), dtype=np.float32),
            "layer21": np.empty((n_samples, 64, 128), dtype=np.float32),
            "layer22": np.empty((n_samples, 64, 128), dtype=np.float32),
            "layer23": np.empty((n_samples, 64, 128), dtype=np.float32),
            "layer24": np.empty((n_samples, 32, 64), dtype=np.float32),
            "layer25": np.empty((n_samples, 32, 64), dtype=np.float32),
            "layer26": np.empty((n_samples, 32, 64), dtype=np.float32),
            "layer27": np.empty((n_samples, 16, 32), dtype=np.float32),
            "layer28": np.empty((n_samples, 16, 32), dtype=np.float32),
            "layer29": np.empty((n_samples, 16, 32), dtype=np.float32),
            "layer30": np.empty((n_samples, 8, 16), dtype=np.float32),
            "layer31": np.empty((n_samples, 8, 16), dtype=np.float32),
            "layer32": np.empty((n_samples, 8, 16), dtype=np.float32),
            "layer33a": np.empty((n_samples, 16, 32), dtype=np.float32),
            "layer34a": np.empty((n_samples, 32, 64), dtype=np.float32),
            "layer35a": np.empty((n_samples, 64, 128), dtype=np.float32),
            "layer36a": np.empty((n_samples, 128, 256), dtype=np.float32),
            "layer37": np.empty((n_samples, 256, 512), dtype=np.float32),
        }

        # pre-allocate target disparity label
        act_targetDisp = np.empty(n_samples, dtype=np.int8)

        # iterate through the data and compute activations
        tepoch = tqdm(rds_loader)
        for i, (inputs_left, inputs_right, disps) in enumerate(tepoch):
            # (inputs_left, inputs_right, disps) = next(iter(rds_loader))

            id_start = i * self.batch_size
            id_end = id_start + self.batch_size

            # generate disparity direction
            ref = disps / 10.0

            # build nested tensor
            input_data = NestedTensor(
                left=inputs_left.pin_memory().to(self.config.device, non_blocking=True),
                right=inputs_right.pin_memory().to(
                    self.config.device, non_blocking=True
                ),
                ref=ref.pin_memory().to(self.config.device, non_blocking=True),
            )

            # swap left and right inputs if ref < 0
            # if ref.mean() > 0:
            #     input_data = NestedTensor(
            #         left=inputs_right.pin_memory().to(
            #             self.config.device, non_blocking=True
            #         ),
            #         right=inputs_left.pin_memory().to(
            #             self.config.device, non_blocking=True
            #         ),
            #         ref=ref.pin_memory().to(self.config.device, non_blocking=True),
            #     )
            # else:
            #     input_data = NestedTensor(
            #         left=inputs_left.pin_memory().to(
            #             self.config.device, non_blocking=True
            #         ),
            #         right=inputs_right.pin_memory().to(
            #             self.config.device, non_blocking=True
            #         ),
            #         ref=ref.pin_memory().to(self.config.device, non_blocking=True),
            #     )

            # save target disparity label
            act_targetDisp[id_start:id_end] = disps.numpy()

            # compute activation for whole layers
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                module_outputs = self.compute_layer_activations(
                    input_data, self.target_list
                )

            # fetching each layer activation
            for i, layer in enumerate(self.target_list):

                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    # get layer activation and average across feature channels
                    # [batch, feat_channel, disp_channel, h, w] => [batch, disp_channel, h, w]
                    out = module_outputs[layer].mean(dim=1)

                    # compute the prob (normalized across disp_channel)
                    out = F.softmax(-out, dim=1)  # [batch, disp_channel, h, w]

                    # create disparity indices tensor
                    n_disp_channel = out.shape[1]
                    disp_indices = self.create_disp_indices(n_disp_channel)

                    # compute the expected activation across disp channels
                    if i == len(self.target_list) - 1:
                        layer_act = torch.sum(
                            out * disp_indices * input_data.ref.view(-1, 1, 1, 1), dim=1
                        )  # [batch, h, w]
                    else:
                        layer_act = torch.sum(
                            out * disp_indices, dim=1
                        )  # [batch, h, w]

                # store in the dict
                # layer_act_dict[self.layer_name[l]][i] = layer_act.cpu().detach().numnpy()
                layer_act_dict[self.layer_name[i]][id_start:id_end] = (
                    layer_act.cpu().detach().numpy()
                )

        # save file
        # for layer in self.layer_name:
        np.save(
            f"{self.layer_act_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}"
            + f"_dotMatch_{dotMatch:.2f}.npy",
            layer_act_dict,
        )

        np.save(
            f"{self.layer_act_dir}/targetDisp_rds"
            + f"_dotDens_{dotDens:.2f}"
            + f"_dotMatch_{dotMatch:.2f}.npy",
            act_targetDisp,
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

    def xDecode(
        self,
        dotDens: float,
        split_train: float,
        n_bootstrap: int,
    ) -> None:
        """
        single bootstrap cross-decoding for a given dot density

        Args:
            dotDens (float): dot density
            split_train (float): proportion of training dataset
            n_bootstrap (int): the number of bootstrap iterations

        Returns:
            score_ards, score_hmrds, score_crds
                (n_bootstrap, len(self.layer_name))
                float32: cross-decoding score for each rds type
        """

        # dotDens = 0.4
        n_samples = (
            2 * self.n_rds_each_disp
        )  # number of rds in total, the "2" comes from near and far disp
        n_train = int(split_train * n_samples)  # number of training dataset

        # load layer activation data and target disparity label
        # ards
        dotMatch = 0.0
        layer_act_ards = np.load(
            f"{self.layer_act_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        ).item()
        targetDisp_ards = np.load(
            f"{self.layer_act_dir}/targetDisp_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        )

        # hmrds
        dotMatch = 0.5
        layer_act_hmrds = np.load(
            f"{self.layer_act_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        ).item()
        targetDisp_hmrds = np.load(
            f"{self.layer_act_dir}/targetDisp_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        )

        # crds
        dotMatch = 1.0
        layer_act_crds = np.load(
            f"{self.layer_act_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        ).item()
        # layer_act_crds = np.load(
        #     f"{rdsl.layer_act_dir}/act_rds"
        #     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
        # allow_pickle=True,
        # ).item()
        targetDisp_crds = np.load(
            f"{self.layer_act_dir}/targetDisp_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        )
        # targetDisp_crds = np.load(
        #     f"{rdsl.layer_act_dir}/targetDisp_rds"
        #     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
        # allow_pickle=True,
        # )

        score_ards = np.empty(
            (n_bootstrap, len(self.layer_name)),
            dtype=np.float32,
        )
        score_hmrds = np.empty(
            (n_bootstrap, len(self.layer_name)),
            dtype=np.float32,
        )
        score_crds = np.empty(
            (n_bootstrap, len(self.layer_name)),
            dtype=np.float32,
        )

        # loop over each DNN layer
        for i, layer in enumerate(self.layer_name):

            # Pre-compute row-averaged activations for each sample
            crds_avg = layer_act_crds[layer].mean(axis=-2)  # [n_samples, w]
            ards_avg = layer_act_ards[layer].mean(axis=-2)  # [n_samples, w]
            hmrds_avg = layer_act_hmrds[layer].mean(axis=-2)  # [n_samples, w]

            # compute global mean and std for normalization across batch with crds
            x_mean = crds_avg.mean()  # axis=0, keepdims=True)  # [1, w]
            x_std = crds_avg.std()  # axis=0, keepdims=True)  # [1, w]
            # prevent division by zero
            # x_std[x_std == 0] = 1e-6

            # standardize/normalize activations across samples
            crds_norm = (crds_avg - x_mean) / x_std  # [n_samples, w]
            ards_norm = (ards_avg - x_mean) / x_std  # [n_samples, w]
            hmrds_norm = (hmrds_avg - x_mean) / x_std  # [n_samples, w]

            # bootstrap loop
            for i_bootstrap in range(n_bootstrap):

                print(
                    f"Cross-decoding for rds dotDens {dotDens:.2f}, "
                    + f"{layer}, bootstrap: {i_bootstrap}/{n_bootstrap}"
                )

                # generate random numbers for splitting train and test dataset
                idx = np.random.permutation(n_samples)
                idx_train = idx[:n_train]
                idx_test = idx[n_train:]

                # set up cRDS training dataset
                x_train = crds_norm[idx_train]  # [n_train, w]
                y_train = targetDisp_crds[idx_train]  # [n_train]

                # train classifier
                clf = svm.SVC(kernel="linear", cache_size=1000)
                clf.fit(x_train, y_train)

                # evaluate on ards
                score_ards[i_bootstrap, i] = clf.score(ards_norm, targetDisp_ards)

                # evaluate on hmrds
                score_hmrds[i_bootstrap, i] = clf.score(hmrds_norm, targetDisp_hmrds)

                # evaluate on crds test subset
                x_test = crds_norm[idx_test]  # [n_test, w]
                y_test = targetDisp_crds[idx_test]  # [n_test]
                # fit
                score_crds[i_bootstrap, i] = clf.score(x_test, y_test)

                # clean up svm classifier
                del clf
            # gc.collect()

        # save xDecode
        np.save(
            f"{self.layer_act_dir}/xDecode_score_ards_dotDens_{dotDens:.2f}_bootstrap",
            score_ards,
        )
        np.save(
            f"{self.layer_act_dir}/xDecode_score_hmrds_dotDens_{dotDens:.2f}_bootstrap",
            score_hmrds,
        )
        np.save(
            f"{self.layer_act_dir}/xDecode_score_crds_dotDens_{dotDens:.2f}_bootstrap",
            score_crds,
        )

    # def xDecode(
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
    #         crds_avg = (
    #             layer_act_crds[layer].mean(axis=(-1, -2)).reshape(-1, 1)
    #         )  # [n_samples, w]
    #         ards_avg = (
    #             layer_act_ards[layer].mean(axis=(-1, -2)).reshape(-1, 1)
    #         )  # [n_samples, w]
    #         hmrds_avg = (
    #             layer_act_hmrds[layer].mean(axis=(-1, -2)).reshape(-1, 1)
    #         )  # [n_samples, w]

    #         # compute global mean and std for normalization across batch with crds
    #         x_mean = crds_avg.mean(axis=0, keepdims=True)  # [1, w]
    #         x_std = crds_avg.std(axis=0, keepdims=True)  # [1, w]
    #         # prevent division by zero
    #         x_std[x_std == 0] = 1e-6

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
    #             clf = svm.SVC(kernel="linear", cache_size=1000)
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

    def plotLine_xDecode_across_layers_at_dotDens(
        self,
        dotDens,
        save_flag,
    ):

        # load xdecode score data at dotDens
        # [n_bootstrap, len(layer_name)]
        xDecode_ards_bootstrap = np.load(
            f"{self.layer_act_dir}/xDecode_score_ards_dotDens_{dotDens:.2f}_bootstrap.npy"
        )
        xDecode_hmrds_bootstrap = np.load(
            f"{self.layer_act_dir}/xDecode_score_hmrds_dotDens_{dotDens:.2f}_bootstrap.npy"
        )
        xDecode_crds_bootstrap = np.load(
            f"{self.layer_act_dir}/xDecode_score_crds_dotDens_{dotDens:.2f}_bootstrap.npy"
        )

        # average across bootstrap
        xDecode_ards_avg = xDecode_ards_bootstrap.mean(axis=0)  # [len(layer_name)]
        xDecode_hmrds_avg = xDecode_hmrds_bootstrap.mean(axis=0)
        xDecode_crds_avg = xDecode_crds_bootstrap.mean(axis=0)

        xDecode_ards_std = xDecode_ards_bootstrap.std(axis=0)
        xDecode_hmrds_std = xDecode_hmrds_bootstrap.std(axis=0)
        xDecode_crds_std = xDecode_crds_bootstrap.std(axis=0)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (12, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            f"Cross-decoding performance across GC-Net layers (dotDens: {dotDens:.2f})",
            ha="center",
        )
        fig.text(-0.03, 0.5, "Prediction acc.", va="center", rotation=90)
        fig.text(0.5, -0.04, "Layer", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        # upper and lower boxplot y-axis
        x_low = 0
        x_up = len(self.layer_name)
        x_step = 2
        y_low = 0.1
        y_up = 1.1
        y_step = 0.1

        # layer id
        layer_name = [
            "19",
            "21",
            "23",
            "25",
            "27",
            "29",
            "31",
            "33a",
            "35a",
            "37",
        ]

        x = np.arange(len(self.layer_name))

        # chance level
        axes.axhline(0.5, color="k", linestyle="--", linewidth=3)

        # ards
        y = xDecode_ards_avg
        y_err = xDecode_ards_std
        axes.errorbar(x, y, yerr=y_err, color="r", linewidth=3)

        # hmrds
        y = xDecode_hmrds_avg
        y_err = xDecode_hmrds_std
        axes.errorbar(x, y, yerr=y_err, color="g", linewidth=3)

        # crds
        y = xDecode_crds_avg
        y_err = xDecode_crds_std
        axes.errorbar(x, y, yerr=y_err, color="b", linewidth=3)

        # axes limit
        axes.set_xlim(x_low - 0.5, x_up + 0.5)
        axes.set_xticks(np.arange(x_low, x_up, x_step))
        axes.set_xticklabels(layer_name, rotation=45)
        axes.set_ylim(y_low, y_up)
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        plt.legend(
            [
                "Chance",
                "cRDS vs. aRDS",
                "cRDS vs. hmRDS",
                "cRDS",
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            fancybox=True,
            shadow=True,
            ncol=4,
        )

        # save
        if save_flag == 1:
            if not os.path.exists(f"{self.layer_act_dir}/Plots"):
                os.mkdir(f"{self.layer_act_dir}/Plots")
            fig.savefig(
                f"{self.layer_act_dir}/Plots/plotLine_xDecode_dotDens_{dotDens:.2f}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_xDecode_across_layers(
        self,
        save_flag,
    ):

        # load xdecode score data for all dotDens
        xDecode_ards_bootstrap = np.empty(
            (len(self.dotDens_list), self.n_bootstrap, len(self.target_list)),
            dtype=np.float32,
        )
        xDecode_hmrds_bootstrap = np.empty(
            (len(self.dotDens_list), self.n_bootstrap, len(self.target_list)),
            dtype=np.float32,
        )
        xDecode_crds_bootstrap = np.empty(
            (len(self.dotDens_list), self.n_bootstrap, len(self.target_list)),
            dtype=np.float32,
        )
        for dd, dotDens in enumerate(self.dotDens_list):

            xDecode_ards_bootstrap[dd] = np.load(
                f"{self.layer_act_dir}/xDecode_score_ards_dotDens_{dotDens:.2f}_bootstrap.npy"
            )
            xDecode_hmrds_bootstrap[dd] = np.load(
                f"{self.layer_act_dir}/xDecode_score_hmrds_dotDens_{dotDens:.2f}_bootstrap.npy"
            )
            xDecode_crds_bootstrap[dd] = np.load(
                f"{self.layer_act_dir}/xDecode_score_crds_dotDens_{dotDens:.2f}_bootstrap.npy"
            )

        # average across bootstrap
        xDecode_ards_avg = xDecode_ards_bootstrap.mean(axis=1)
        xDecode_hmrds_avg = xDecode_hmrds_bootstrap.mean(axis=1)
        xDecode_crds_avg = xDecode_crds_bootstrap.mean(axis=1)

        xDecode_ards_std = xDecode_ards_bootstrap.std(axis=1)
        xDecode_hmrds_std = xDecode_hmrds_bootstrap.std(axis=1)
        xDecode_crds_std = xDecode_crds_bootstrap.std(axis=1)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (25, 15)
        n_row = 3
        n_col = 3

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.02,
            "Cross-decoding performance across GC-Net layers",
            ha="center",
        )
        fig.text(-0.03, 0.5, "Prediction acc.", va="center", rotation=90)
        fig.text(0.5, -0.03, "Layer", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        # upper and lower boxplot y-axis
        x_low = 0
        x_up = len(self.layer_name)
        x_step = 2
        y_low = 0.0
        y_up = 1.1
        y_step = 0.25

        # layer id
        layer_name = [
            "19",
            "21",
            "23",
            "25",
            "27",
            "29",
            "31",
            "33a",
            "35a",
            "37",
        ]

        x = np.arange(len(self.layer_name))
        for i in range(n_row):
            for j in range(n_col):

                idx = i * n_row + j

                # chance level
                axes[i, j].axhline(0.5, color="k", linestyle="--", linewidth=3)

                # ards
                y = xDecode_ards_avg[idx]
                y_err = xDecode_ards_std[idx]
                axes[i, j].errorbar(x, y, yerr=y_err, color="r", linewidth=3)

                # hmrds
                y = xDecode_hmrds_avg[idx]
                y_err = xDecode_hmrds_std[idx]
                axes[i, j].errorbar(x, y, yerr=y_err, color="g", linewidth=3)

                # crds
                y = xDecode_crds_avg[idx]
                y_err = xDecode_crds_std[idx]
                axes[i, j].errorbar(x, y, yerr=y_err, color="b", linewidth=3)

                # title
                axes[i, j].set_title(f"Dot density {self.dotDens_list[idx]:.1f}")

                # axes limit
                axes[i, j].set_xlim(x_low - 0.5, x_up + 0.5)
                axes[i, j].set_xticks(np.arange(x_low, x_up, x_step))
                axes[i, j].set_xticklabels(layer_name, rotation=45)
                axes[i, j].set_ylim(y_low, y_up)
                axes[i, j].set_yticks(np.arange(y_low, y_up, y_step))
                axes[i, j].set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))

                # remove top and right frame
                axes[i, j].spines["top"].set_visible(False)
                axes[i, j].spines["right"].set_visible(False)

                # show ticks on the left and bottom axis
                axes[i, j].xaxis.set_ticks_position("bottom")
                axes[i, j].yaxis.set_ticks_position("left")

        plt.legend(
            [
                "Chance",
                "cRDS vs. aRDS",
                "cRDS vs. hmRDS",
                "cRDS",
            ],
            loc="upper center",
            bbox_to_anchor=(-0.75, -0.5),
            fancybox=True,
            shadow=True,
            ncol=4,
        )

        # save
        if save_flag == 1:
            if not os.path.exists(f"{self.layer_act_dir}/Plots"):
                os.mkdir(f"{self.layer_act_dir}/Plots")
            fig.savefig(
                f"{self.layer_act_dir}/Plots/plotLine_xDecode.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotHeat_xDecode(
        self,
        save_flag,
    ):
        # load xdecode score data for all dotDens
        xDecode_ards_bootstrap = np.empty(
            (len(self.dotDens_list), self.n_bootstrap, len(self.target_list)),
            dtype=np.float32,
        )
        xDecode_hmrds_bootstrap = np.empty(
            (len(self.dotDens_list), self.n_bootstrap, len(self.target_list)),
            dtype=np.float32,
        )
        xDecode_crds_bootstrap = np.empty(
            (len(self.dotDens_list), self.n_bootstrap, len(self.target_list)),
            dtype=np.float32,
        )
        for dd, dotDens in enumerate(self.dotDens_list):

            xDecode_ards_bootstrap[dd] = np.load(
                f"{self.layer_act_dir}/xDecode_score_ards_dotDens_{dotDens:.2f}_bootstrap.npy"
            )
            xDecode_hmrds_bootstrap[dd] = np.load(
                f"{self.layer_act_dir}/xDecode_score_hmrds_dotDens_{dotDens:.2f}_bootstrap.npy"
            )
            xDecode_crds_bootstrap[dd] = np.load(
                f"{self.layer_act_dir}/xDecode_score_crds_dotDens_{dotDens:.2f}_bootstrap.npy"
            )

        # average across bootstrap
        xDecode_ards_avg = np.flipud(xDecode_ards_bootstrap.mean(axis=1).T)
        xDecode_hmrds_avg = np.flipud(xDecode_hmrds_bootstrap.mean(axis=1).T)
        xDecode_crds_avg = np.flipud(xDecode_crds_bootstrap.mean(axis=1).T)

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (20, 15)
        n_row = 1
        n_col = 3

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            0.9,
            "Cross-decoding performance across GC-Net layers",
            ha="center",
        )
        fig.text(-0.02, 0.5, "Layer", va="center", rotation=90)
        fig.text(0.5, 0.1, "Dot density", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # heatmap params
        layer_id = [
            "37",
            "36a",
            "35a",
            "34a",
            "33a",
            "32",
            "31",
            "30",
            "29",
            "28",
            "27",
            "26",
            "25",
            "24",
            "23",
            "22",
            "21",
            "20",
            "19",
        ]
        cmap = "coolwarm"
        v_min = 0.0
        v_max = 1.0
        x_low = 0.1
        x_up = 0.91
        x_step = 0.2
        y_low = 0
        y_up = len(layer_id)
        y_step = 1

        # ards
        im = axes[0].imshow(
            xDecode_ards_avg, cmap=cmap, vmin=v_min, vmax=v_max, interpolation="nearest"
        )
        # title
        axes[0].set_title("cRDS vs. aRDS")

        # axis
        axes[0].set_xticks(np.arange(len(self.dotDens_list), step=2))
        axes[0].set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes[0].set_yticks(np.arange(y_low, y_up, y_step))
        axes[0].set_yticklabels(layer_id)
        plt.colorbar(im, fraction=0.08, pad=0.08)

        # hmrds
        im = axes[1].imshow(
            xDecode_hmrds_avg,
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
            interpolation="nearest",
        )
        axes[1].set_title("cRDS vs. hmRDS")
        plt.colorbar(im, fraction=0.08, pad=0.08)

        # crds
        im = axes[2].imshow(
            xDecode_crds_avg, cmap=cmap, vmin=v_min, vmax=v_max, interpolation="nearest"
        )
        axes[2].set_title("cRDS")
        plt.colorbar(im, fraction=0.08, pad=0.08)

        # save
        if save_flag == 1:
            if not os.path.exists(f"{self.layer_act_dir}/Plots"):
                os.mkdir(f"{self.layer_act_dir}/Plots")
            fig.savefig(
                f"{self.layer_act_dir}/Plots/plotHeatmap_xDecode.pdf",
                dpi=600,
                bbox_inches="tight",
            )


# %%
