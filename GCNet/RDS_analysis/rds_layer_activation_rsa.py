# %% load necessary modules
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor

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
from numba import njit, prange
import os
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


@njit
def pearson_corr_vector(x, y):
    """
    compute pearson correlation between two vectors

    x: [w]
    y: [w]
    """
    tol = 1e-4
    # compute covariance
    cov_xy = np.sum((x - x.mean()) * (y - y.mean())) / len(x)

    # compute std
    std_x = np.std(x)
    std_y = np.std(y)
    # prevent division by zero
    if std_x < tol:
        std_x = tol
    if std_y < tol:
        std_y = tol

    if std_x <= 0.0 or std_y <= 0.0:
        return 0.0

    # compute correlation
    corr = cov_xy / (std_x * std_y)

    if corr > 1.0:
        corr = 1.0
    elif corr < -1.0:
        corr = -1.0

    return corr


@njit(parallel=True)
def pearson_corr_matrices(x, y):
    """
    compute pearson correlation between two matrices

    x: [n_samples, w]
    y: [n_samples, w]
    """

    n_samples = x.shape[0]

    score = np.empty(n_samples, dtype=np.float32)
    for i in prange(n_samples):
        score[i] = pearson_corr_vector(x[i], y[i])

    return score


@njit
def rdm_score(
    rds_act,
):
    """
    compute rdm score

    rds_act: Float[np.array, "6 n_samples w"]

        each row is a vector of activations for each rds type:
            ards_near, # [n_samples, w]
            ards_far,
            hmrds_near,
            hmrds_far,
            crds_near,
            crds_far,

    """

    rdm = np.empty((6, 6), dtype=np.float32)

    for i in range(6):
        for j in range(i, 6):

            temp = pearson_corr_matrices(rds_act[i], rds_act[j]).mean()
            rdm[i, j] = temp
            rdm[j, i] = temp
    return rdm


# %%
class RDS_LayerAct_RSA(Engine):

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
            # input_data = NestedTensor(
            #     left=inputs_left.pin_memory().to(self.config.device, non_blocking=True),
            #     right=inputs_right.pin_memory().to(
            #         self.config.device, non_blocking=True
            #     ),
            #     ref=ref.pin_memory().to(self.config.device, non_blocking=True),
            # )

            # swap left and right inputs if ref < 0
            if ref.mean() > 0:
                input_data = NestedTensor(
                    left=inputs_right.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    right=inputs_left.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )
            else:
                input_data = NestedTensor(
                    left=inputs_left.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    right=inputs_right.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )

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
                    # if i == len(self.target_list) - 1:
                    #     layer_act = torch.sum(
                    #         out * disp_indices * input_data.ref.view(-1, 1, 1, 1), dim=1
                    #     )  # [batch, h, w]
                    # else:
                    layer_act = torch.sum(out * disp_indices, dim=1)  # [batch, h, w]

                # store in the dict
                # layer_act_dict[self.layer_name[l]][i] = layer_act.cpu().detach().numpy()
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

    def compute_rdm(self, dotDens: float):
        """
        compute rsa for each layer activation
        """

        n_samples = rdsl.n_rds_each_disp

        # load layer activation data and target_disp label
        # ards
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
        # targetDisp_ards = np.load(
        #     f"{rdsl.layer_act_dir}/targetDisp_rds"
        #     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
        #     allow_pickle=True,
        # )

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
        # targetDisp_hmrds = np.load(
        #     f"{rdsl.layer_act_dir}/targetDisp_rds"
        #     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
        #     allow_pickle=True,
        # )

        # crds
        dotMatch = 1.0
        layer_act_crds = np.load(
            f"{self.layer_act_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        ).item()
        targetDisp_crds = np.load(
            f"{self.layer_act_dir}/targetDisp_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        )
        # targetDisp_crds = np.load(
        #     f"{rdsl.layer_act_dir}/targetDisp_rds"
        #     + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
        #     allow_pickle=True,
        # )

        # allocate memory for distance score
        rdm = np.empty((len(rdsl.layer_name), rdsl.n_bootstrap, 6, 6), dtype=np.float32)

        # loop through each layer
        for i, layer in enumerate(rdsl.layer_name):

            layer = rdsl.layer_name[i]

            # row-averaged activations
            ards_avg = layer_act_ards[layer].mean(
                axis=-2
            )  # [n_samples, h, w] -> [n_samples, w]
            hmrds_avg = layer_act_hmrds[layer].mean(axis=-2)
            crds_avg = layer_act_crds[layer].mean(axis=-2)

            # get near/far layer activation
            ards_near = ards_avg[targetDisp_ards == -10]
            ards_far = ards_avg[targetDisp_ards == 10]
            hmrds_near = hmrds_avg[targetDisp_hmrds == -10]
            hmrds_far = hmrds_avg[targetDisp_hmrds == 10]
            crds_near = crds_avg[targetDisp_crds == -10]
            crds_far = crds_avg[targetDisp_crds == 10]

            # # compute global mean and std for normalization across batch with crds
            x_mean = crds_avg.mean()  # axis=0, keepdims=True)  # [1, w]
            x_std = crds_avg.std()  # axis=0, keepdims=True)  # [1, w]
            # # prevent division by zero
            # # x_std[x_std == 0] = 1e-6

            # # standardize/normalize activations across samples
            # x_mean = crds_avg.mean()  # axis=0, keepdims=True)  # [1, w]
            # x_std = crds_avg.std()  # axis=0, keepdims=True)  # [1, w]
            crds_near_norm = (crds_near - x_mean) / x_std  # [n_samples, w]
            crds_far_norm = (crds_far - x_mean) / x_std  # [n_samples, w]

            # x_mean = ards_avg.mean()  # axis=0, keepdims=True)  # [1, w]
            # x_std = ards_avg.std()  # axis=0, keepdims=True)  # [1, w]
            ards_near_norm = (ards_near - x_mean) / x_std  # [n_samples, w]
            ards_far_norm = (ards_far - x_mean) / x_std  # [n_samples, w]

            # x_mean = hmrds_avg.mean()  # axis=0, keepdims=True)  # [1, w]
            # x_std = hmrds_avg.std()  # axis=0, keepdims=True)  # [1, w]
            hmrds_near_norm = (hmrds_near - x_mean) / x_std  # [n_samples, w]
            hmrds_far_norm = (hmrds_far - x_mean) / x_std  # [n_samples, w]

            # bootstrap loop
            for i_bootstrap in range(rdsl.n_bootstrap):

                # generate random numbers for splitting train and test dataset
                idx = np.random.choice(np.arange(n_samples), n_samples, replace=True)

                ards_near_i = ards_near_norm[idx]  # [n_samples, w]
                ards_far_i = ards_far_norm[idx]
                hmrds_near_i = hmrds_near_norm[idx]
                hmrds_far_i = hmrds_far_norm[idx]
                crds_near_i = crds_near_norm[idx]
                crds_far_i = crds_far_norm[idx]

                # concatenate
                rds_act = np.stack(
                    [
                        ards_near_i,
                        ards_far_i,
                        hmrds_near_i,
                        hmrds_far_i,
                        crds_near_i,
                        crds_far_i,
                    ],
                    axis=0,
                )

                rdm[i, i_bootstrap] = rdm_score(rds_act)

        return rdm

    def compute_corr_activation(self, dotDens: float):
        """
        compute rsa for each layer activation
        """

        n_samples = 2 * self.n_rds_each_disp

        # load layer activation data and target_disp label
        # ards
        # ards
        dotMatch = 0.0
        layer_act_ards = np.load(
            f"{self.layer_act_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        ).item()

        # hmrds
        dotMatch = 0.5
        layer_act_hmrds = np.load(
            f"{self.layer_act_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        ).item()

        # crds
        dotMatch = 1.0
        layer_act_crds = np.load(
            f"{self.layer_act_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}_dotMatch_{dotMatch:.2f}.npy",
            allow_pickle=True,
        ).item()

        # allocate memory for distance score
        corr_crds_ards = np.empty(
            (len(self.layer_name), self.n_bootstrap), dtype=np.float32
        )
        corr_crds_hmrds = np.empty(
            (len(self.layer_name), self.n_bootstrap), dtype=np.float32
        )
        corr_crds_crds = np.empty(
            (len(self.layer_name), self.n_bootstrap), dtype=np.float32
        )

        # loop through each layer
        for i, layer in enumerate(self.layer_name):

            layer = self.layer_name[i]

            # row-averaged activations
            ards_avg = layer_act_ards[layer].mean(
                axis=-2
            )  # [n_samples, h, w] -> [n_samples, w]
            hmrds_avg = layer_act_hmrds[layer].mean(axis=-2)
            crds_avg = layer_act_crds[layer].mean(axis=-2)

            # # compute global mean and std for normalization across batch with crds
            x_mean = crds_avg.mean()  # axis=0, keepdims=True)  # [1, w]
            x_std = crds_avg.std()  # axis=0, keepdims=True)  # [1, w]
            # # prevent division by zero
            # # x_std[x_std == 0] = 1e-6

            # # standardize/normalize activations across samples
            # x_mean = crds_avg.mean()  # axis=0, keepdims=True)  # [1, w]
            # x_std = crds_avg.std()  # axis=0, keepdims=True)  # [1, w]
            crds_norm = (crds_avg - x_mean) / x_std  # [n_samples, w]

            # x_mean = ards_avg.mean()  # axis=0, keepdims=True)  # [1, w]
            # x_std = ards_avg.std()  # axis=0, keepdims=True)  # [1, w]
            ards_norm = (ards_avg - x_mean) / x_std  # [n_samples, w]

            # x_mean = hmrds_avg.mean()  # axis=0, keepdims=True)  # [1, w]
            # x_std = hmrds_avg.std()  # axis=0, keepdims=True)  # [1, w]
            hmrds_norm = (hmrds_avg - x_mean) / x_std  # [n_samples, w]

            # bootstrap loop
            for i_bootstrap in range(self.n_bootstrap):

                # generate random numbers for splitting train and test dataset
                idx = np.random.choice(np.arange(n_samples), n_samples, replace=True)

                # crds vs ards
                x = crds_norm[idx]
                y = ards_norm[idx]
                corr_crds_ards[i, i_bootstrap] = pearson_corr_matrices(x, y).mean()

                # crds vs hmrds
                x = crds_norm[idx]
                y = hmrds_norm[idx]
                corr_crds_hmrds[i, i_bootstrap] = pearson_corr_matrices(x, y).mean()

                # crds vs crds
                x = crds_norm[idx[: n_samples // 2]]
                y = crds_norm[idx[n_samples // 2 :]]
                corr_crds_crds[i, i_bootstrap] = pearson_corr_matrices(x, y).mean()

        return corr_crds_ards, corr_crds_hmrds, corr_crds_crds

    def plotLine_rsa(
        self,
        score_crds_ards: Float[Tensor, "n_layers n_bootstrap"],
        score_crds_hmrds: Float[Tensor, "n_layers n_bootstrap"],
        score_crds_crds: Float[Tensor, "n_layers n_bootstrap"],
        dotDens: float,
        save_flag: bool,
    ):
        """
        plot rsa scores across layers
        """

        # average across bootstrap
        crds_ards_mean = score_crds_ards.mean(axis=1)
        crds_hmrds_mean = score_crds_hmrds.mean(axis=1)
        crds_crds_mean = score_crds_crds.mean(axis=1)
        crds_ards_std = score_crds_ards.std(axis=1)
        crds_hmrds_std = score_crds_hmrds.std(axis=1)
        crds_crds_std = score_crds_crds.std(axis=1)

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
            f"Pearson correlation between RDS activation across GC-Net layers\n(dotDens: {dotDens:.2f})",
            ha="center",
        )
        fig.text(-0.03, 0.5, "Pearson corr.", va="center", rotation=90)
        fig.text(0.5, -0.04, "Layer", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        # upper and lower boxplot y-axis
        x_low = 0
        x_up = len(self.layer_name)
        x_step = 2
        y_low = -1.0
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

        # chance level
        axes.axhline(0.0, color="k", linestyle="--", linewidth=3)

        # ards
        y = crds_ards_mean
        y_err = crds_ards_std
        axes.errorbar(x, y, yerr=y_err, color="r", linewidth=3)

        # hmrds
        y = crds_hmrds_mean
        y_err = crds_hmrds_std
        axes.errorbar(x, y, yerr=y_err, color="g", linewidth=3)

        # crds
        y = crds_crds_mean
        y_err = crds_crds_std
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
                f"{self.layer_act_dir}/Plots/plotLine_layer_act_pearson_dotDens_{dotDens:.2f}.pdf",
                dpi=600,
                bbox_inches="tight",
            )
