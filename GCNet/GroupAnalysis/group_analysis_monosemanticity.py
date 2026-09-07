# %% load necessary modules

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from joblib import Parallel, delayed
from timeit import default_timer as timer

from scipy.stats import sem
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf

from GroupAnalysis.group_analysis import GA

# from modules.bnn import build_bnn
from config.config import GCNetconfig
from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS
from utilities.misc import NestedTensor

from jaxtyping import Float
from typing import Dict
import os


# %%
class GA_Monosemanticity(GA):

    def __init__(self, config: GCNetconfig, params_rds: dict):

        super().__init__(config, params_rds)

        # create folders for monosemanticity analysis
        self.monosemanticity_dir = (
            f"{self.experiment_dir}/"
            + f"monosemanticity_epoch_{self.epoch_to_load}"
            + f"_iter_{self.iter_to_load}"
        )
        if not os.path.exists(self.monosemanticity_dir):
            os.makedirs(self.monosemanticity_dir)

        # save folder
        self.data_dir = os.path.join(self.group_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        # print out network configuration
        self.__getconfig__()

    def __getconfig__(self) -> None:
        """
        print out the network configuration
        """

        print(
            "Network config\n"
            + f"binocular interaction: {self.config.binocular_interaction}\n"
            + f"seed: {self.config.seed}\n"
            + f"epoch: {self.config.epoch_to_load}\n"
            + f"iter: {self.config.iter_to_load}\n"
            + f"experiment_dir: {self.experiment_dir}\n"
            + f"monosemanticity_dir: {self.monosemanticity_dir}\n"
        )

    def update_network_config(
        self, interaction: str, seed: int, epoch_to_load: int, iter_to_load: int
    ) -> None:
        """
        Update the binocular interaction in the config and experiment
        directory.

        Args:
            interaction (str): The new binocular interaction to set.
        """

        # old config, for printing purposes
        interaction_old = self.binocular_interaction
        seed_old = self.seed
        epoch_old = self.epoch_to_load
        iter_old = self.iter_to_load

        # update binocular_interaction, seed, epoch, iter in
        # the class and config
        self.binocular_interaction = interaction
        self.config.binocular_interaction = interaction
        self.seed = seed
        self.config.seed = seed
        self.epoch_to_load = epoch_to_load
        self.config.epoch_to_load = epoch_to_load
        self.iter_to_load = iter_to_load
        self.config.iter_to_load = iter_to_load

        # update the experiment directories based on the new interaction
        self.experiment_dir = (
            f"run/{self.dataset}/"
            + f"bino_interaction_{self.binocular_interaction}/"
            + f"{self.seed}"
        )

        # update directory for storing plots of a given interaction
        # (average across seeds)
        self.plot_dir = f"{self.experiment_dir}/../plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # update monosemanticity_dir
        self.monosemanticity_dir = (
            f"{self.experiment_dir}/"
            + f"monosemanticity_epoch_{self.epoch_to_load}"
            + f"_iter_{self.iter_to_load}"
        )
        if not os.path.exists(self.monosemanticity_dir):
            os.makedirs(self.monosemanticity_dir)

        print(
            "Updating network config\n"
            + f"binocular interaction: {interaction_old} => {self.config.binocular_interaction}\n"
            + f"seed: {seed_old} => {self.config.seed}\n"
            + f"epoch: {epoch_old} => {self.config.epoch_to_load}\n"
            + f"iter: {iter_old} => {self.config.iter_to_load}\n"
            + f"experiment_dir: {self.experiment_dir}\n"
            + f"monosemanticity_dir: {self.monosemanticity_dir}\n"
        )

    @torch.inference_mode()
    def compute_layer_act_rds(self, dotMatch: float, dotDens: float):
        """
        compute the RDS activation for each layer for a given dotMatch
        and dotDens
        """

        # set up seed
        seed_number = self.seed
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        random.seed(seed_number)
        np.random.seed(seed_number)
        os.environ["PYTHONHASHSEED"] = str(seed_number)

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

        # set the model to evaluation mode
        self.model.eval()  # inference mode (turns off dropout, etc.)

        # Create a dict to store the output
        activations = {}

        # Define a forward hook to capture the output of the target layer
        def get_activation(name):
            def hook(module, input, output):
                # output is still on GPU if the model is; detach & move to CPU if needed
                activations[name] = output  # .detach()

            return hook

        # Register the forward hook
        hook_handles = []
        for layer in self.target_layer:
            hook_handle = layer.register_forward_hook(get_activation(layer))
            hook_handles.append(hook_handle)

        # create dataloader
        # dotMatch = 1.0
        # dotDens = 0.3
        rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
            dotMatch,
            dotDens,
            self.disp_ct_pix_list,
            self.n_rds_each_disp,
            self.background_flag,
            self.pedestal_flag,
        )

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t + 1.0) / 2.0),
                transforms.Normalize(mean, std),
            ]
        )
        rds_data = DatasetRDS(rds_left, rds_right, rds_label, transform=transform_data)
        batch_size = self.batch_size_rds
        rds_loader = DataLoader(
            rds_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=2,
            persistent_workers=True,  # re-use workers across epochs
            prefetch_factor=4,  # overlap load/transfer
            worker_init_fn=seed_worker,
            generator=g,
        )

        n_samples = len(rds_loader.dataset)
        layer_act_dict = {
            "layer19": torch.empty((n_samples, 32 * 96), dtype=torch.float32),
            "layer20": torch.empty((n_samples, 32 * 96), dtype=torch.float32),
            "layer21": torch.empty((n_samples, 64 * 48), dtype=torch.float32),
            "layer22": torch.empty((n_samples, 64 * 48), dtype=torch.float32),
            "layer23": torch.empty((n_samples, 64 * 48), dtype=torch.float32),
            "layer24": torch.empty((n_samples, 64 * 24), dtype=torch.float32),
            "layer25": torch.empty((n_samples, 64 * 24), dtype=torch.float32),
            "layer26": torch.empty((n_samples, 64 * 24), dtype=torch.float32),
            "layer27": torch.empty((n_samples, 64 * 12), dtype=torch.float32),
            "layer28": torch.empty((n_samples, 64 * 12), dtype=torch.float32),
            "layer29": torch.empty((n_samples, 64 * 12), dtype=torch.float32),
            "layer30": torch.empty((n_samples, 128 * 6), dtype=torch.float32),
            "layer31": torch.empty((n_samples, 128 * 6), dtype=torch.float32),
            "layer32": torch.empty((n_samples, 128 * 6), dtype=torch.float32),
            "layer33a": torch.empty((n_samples, 64 * 12), dtype=torch.float32),
            "layer34a": torch.empty((n_samples, 64 * 24), dtype=torch.float32),
            "layer35a": torch.empty((n_samples, 64 * 48), dtype=torch.float32),
            "layer36a": torch.empty((n_samples, 32 * 96), dtype=torch.float32),
            "layer37": torch.empty((n_samples, 192), dtype=torch.float32),
        }

        disp_record = torch.empty(n_samples, dtype=torch.int8)
        for i, (inputs_left, inputs_right, disps) in enumerate(rds_loader):

            id_start = i * batch_size
            id_end = id_start + batch_size

            # record the disparity
            disp_record[id_start:id_end] = disps.cpu()

            # build nested tensor
            # (inputs_left, inputs_right, disps) = next(iter(rds_loader))
            ref = disps / 10.0  # disparity direction
            if ref.mean() > 0:
                input_data = NestedTensor(
                    left=inputs_left.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    right=inputs_right.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )
            else:
                input_data = NestedTensor(
                    left=inputs_right.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    right=inputs_left.pin_memory().to(
                        self.config.device, non_blocking=True
                    ),
                    ref=ref.pin_memory().to(self.config.device, non_blocking=True),
                )
            # input_data = NestedTensor(
            #     left=inputs_left.pin_memory().to(ga_mono.device, non_blocking=True),
            #     right=inputs_right.pin_memory().to(ga_mono.device, non_blocking=True),
            #     disp=disps.pin_memory().to(ga_mono.device, non_blocking=True),
            #     ref=ref.pin_memory().to(ga_mono.device, non_blocking=True),
            # )

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                _ = self.model(input_data)

            # retrieve the activations
            for j, layer in enumerate(self.target_layer):
                out = activations[layer]  # shape: (B, C_out, H_out, W_out)
                # print(f"{target_layer[i]} output:", out.size())

                # average the output across 2D spatial dimensions
                # out_avg = out.mean(dim=(-2, -1)).cpu().detach()
                out_avg = out.mean(dim=(-2, -1)).detach().to("cpu", non_blocking=True)
                if out.ndim == 4:  # len(out.shape) == 4:
                    layer_act_dict[self.layer_name[j]][id_start:id_end] = out_avg
                else:
                    out_avg = out_avg.view(out_avg.size(0), -1)
                    layer_act_dict[self.layer_name[j]][id_start:id_end] = out_avg

        # clean up
        for h in hook_handles:
            h.remove()

        # save file
        # for layer in self.layer_name:
        torch.save(
            layer_act_dict,
            f"{self.monosemanticity_dir}/act_rds"
            + f"_dotDens_{dotDens:.2f}"
            + f"_dotMatch_{dotMatch:.2f}.pt",
        )

        torch.save(
            disp_record,
            f"{self.monosemanticity_dir}/disp_record_rds"
            + f"_dotDens_{dotDens:.2f}"
            + f"_dotMatch_{dotMatch:.2f}.pt",
        )

    def compute_layer_act_all_rds(self):
        """
        compute the RDS activation for each layer for all dotMatch and dotDens
        """

        for dotDens in self.dotDens_list:
            for dotMatch in self.dotMatch_list:

                print(
                    "Computing RDS activation: "
                    + f"dotDens={dotDens:.2f}, "
                    + f"dotMatch={dotMatch:.2f}"
                )

                # compute layer activation for RDS
                self.compute_layer_act_rds(dotMatch, dotDens)

    def compute_layer_act_rds_all_seeds(self, interaction: str):

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

            # update model
            self.load_model()

            # compute layer activation for all RDSs
            self.compute_layer_act_all_rds()

    def compute_monosemanticity(
        self, layer_name: str
    ) -> tuple[Float[Tensor, "n_neuron"], Float[Tensor, "n_neuron"]]:

        # load the activation only for getting the shape
        dotDens = 0.3
        dotMatch = 1.0
        temp = torch.load(
            f"{self.monosemanticity_dir}/"
            + f"act_rds_dotDens_{dotDens:.2f}_"
            + f"dotMatch_{dotMatch:.2f}.pt"
        )[layer_name]
        n_rds_each_disp = temp.size(0) // 2
        # n_feat = len(self.dotMatch_list) * len(self.dotDens_list) * 2
        n_neuron = temp.size(1)
        act_all = torch.empty(
            (
                n_rds_each_disp,
                len(self.dotMatch_list),
                len(self.dotDens_list),
                2,  # near/far
                n_neuron,
            ),
            dtype=torch.float32,
        )

        # gather the activation for all features (RDSs)
        for dm, dotMatch in enumerate(self.dotMatch_list):
            for dd, dotDens in enumerate(self.dotDens_list):

                # dotDens = 0.3
                # dotMatch = 1.0
                act_rds = torch.load(
                    f"{self.monosemanticity_dir}/"
                    + f"act_rds_dotDens_{dotDens:.2f}_"
                    + f"dotMatch_{dotMatch:.2f}.pt"
                )[
                    layer_name
                ]  # [n_sample, n_neuron]

                disp_record = torch.load(
                    f"{self.monosemanticity_dir}/"
                    + f"disp_record_rds_dotDens_{dotDens:.2f}_"
                    + f"dotMatch_{dotMatch:.2f}.pt"
                )

                # separate near/far disparity activation
                # count = (dm * len(self.dotDens_list) * 2) + (dd * 2)
                # act_all[:, count] = act_rds[disp_record == self.disp_ct_pix_list[0]]
                # act_all[:, count + 1] = act_rds[disp_record == self.disp_ct_pix_list[1]]
                act_all[:, dm, dd, 0] = act_rds[disp_record == self.disp_ct_pix_list[0]]
                act_all[:, dm, dd, 1] = act_rds[disp_record == self.disp_ct_pix_list[1]]

        tol = 1e-6
        # average across trials
        act_all_avg = act_all.mean(dim=0)  # [dotMatch, dotDens, near/far, n_neuron]
        num = torch.amax(F.relu(act_all_avg), dim=(0, 1, 2))  # [n_neuron]
        den = torch.sum(F.relu(act_all_avg), dim=(0, 1, 2))  # [n_neuron]
        monosemanticity = num / (den + tol)

        ## compute monosemanticity for each rds_type
        monosemanticity_rds = torch.empty((3, n_neuron), dtype=torch.float32)

        # ards
        num = torch.amax(F.relu(act_all_avg[0]), dim=(0, 1))  # [n_neuron]
        monosemanticity_rds[0] = num / (den + tol)

        # hmrds
        num = torch.amax(F.relu(act_all_avg[1]), dim=(0, 1))  # [n_neuron]
        monosemanticity_rds[1] = num / (den + tol)

        # crds
        num = torch.amax(F.relu(act_all_avg[2]), dim=(0, 1))  # [n_neuron]
        monosemanticity_rds[2] = num / (den + tol)

        return monosemanticity, monosemanticity_rds

    def compute_monosemanticity_rds(
        self, rds_type: str
    ) -> Dict[str, Float[Tensor, "n_neuron"]]:

        n_feat = len(self.dotDens_list) * 2  # number of rds for a given rds_type
        dotMatch = self.rds_types[rds_type]

        # load the activation only for getting the shape
        dotDens = 0.3
        act = torch.load(
            f"{self.monosemanticity_dir}/"
            + f"act_rds_dotDens_{dotDens:.2f}_"
            + f"dotMatch_{dotMatch:.2f}.pt"
        )  # {layer_name: [n_sample, n_neuron]}
        # n_sample = n_rds_each_disp * 2 (near and far activations)

        # setup dict
        act_all_layers = {}
        monosemanticity_rds = {}
        for layer in self.layer_name:

            act_layer = act[layer]  # [n_sample, n_neuron]

            n_neuron = act_layer.size(1)
            act_all_layers[layer] = torch.empty((n_feat, n_neuron), dtype=torch.float32)
            monosemanticity_rds[layer] = torch.empty(n_neuron, dtype=torch.float32)

        # gather the activation for a given rds_type
        for dd, dotDens in enumerate(self.dotDens_list):
            act = torch.load(
                f"{self.monosemanticity_dir}/"
                + f"act_rds_dotDens_{dotDens:.2f}_"
                + f"dotMatch_{dotMatch:.2f}.pt"
            )  # {layer_name: [n_sample, n_neuron]}

            disp_record = torch.load(
                f"{self.monosemanticity_dir}/"
                + f"disp_record_rds_dotDens_{dotDens:.2f}_"
                + f"dotMatch_{dotMatch:.2f}.pt"
            )  # a record containing disparity magnitudes

            count = dd * 2
            for layer in self.layer_name:

                # get layer activation
                act_layer = act[layer]  # [n_sample, n_neuron]

                # separate near/far disparity activation
                # average across batch (n_rds_each_disp)
                act_all_layers[layer][count] = act_layer[
                    disp_record == self.disp_ct_pix_list[0]
                ].mean(
                    dim=0
                )  # [n_neuron]
                act_all_layers[layer][count + 1] = act_layer[
                    disp_record == self.disp_ct_pix_list[1]
                ].mean(
                    dim=0
                )  # [n_neuron]

        # compute monosemanticity
        tol = 1e-6

        for layer in self.layer_name:

            act = act_all_layers[layer]  # [n_feat, n_neuron]
            num = torch.max(F.relu(act), dim=0)[0]  # [n_neuron]
            den = torch.sum(F.relu(act), dim=0)  # [n_neuron]
            monosemanticity_rds[layer] = num / (den + tol)  # [n_neuron]

        return monosemanticity_rds

    def compute_mono_score_layer(
        self, layer_name: str, threshold: float, n_feat_global: int, n_feat_rds: int
    ) -> tuple[Float[Tensor, "n_neuron"], Float[Tensor, "rds_type n_neuron"]]:

        monosemanticity, monosemanticity_rds = self.compute_monosemanticity(
            layer_name
        )  # [n_neuron], [rds_type, n_neuron]

        # compute monosemantic global
        mono_thresh = torch.where(monosemanticity > threshold, 1.0, 0.0)
        mono_score = mono_thresh.sum() / n_feat_global

        # compute monosemantic score for each rds_type
        mono_score_rds = torch.empty(3)
        # ards
        mono_thresh = torch.where(monosemanticity_rds[0] > threshold, 1.0, 0.0)
        mono_per_feat = mono_thresh.sum() / n_feat_rds
        mono_score_rds[0] = mono_per_feat

        # hmrds
        mono_thresh = torch.where(monosemanticity_rds[1] > threshold, 1.0, 0.0)
        mono_per_feat = mono_thresh.sum() / n_feat_rds
        mono_score_rds[1] = mono_per_feat

        # crds
        mono_thresh = torch.where(monosemanticity_rds[2] > threshold, 1.0, 0.0)
        mono_per_feat = mono_thresh.sum() / n_feat_rds
        mono_score_rds[2] = mono_per_feat

        return mono_score, mono_score_rds

    def compute_monosemantic_score(
        self, threshold: float = 0.9, n_feat_global: int = 27, n_feat_rds: int = 9
    ) -> tuple[
        Float[Tensor, "interaction seed layer"],
        Float[Tensor, "interaction seed layer rds_type"],
    ]:
        """
        compute monosemantic score: the number of monosemantic neurons
            per unit feature

        input args:
            interaction: str, binocular interaction to analyze
                (e.g., "default", "bem", "cmm", "sum_diff")

            threshold: float, threshold for monosemanticity
                (default: 0.9)

            n_feat_global: int, number of features per layer for
                monosemantic global
                (default: 27 -> n_rds_types * n_dotDens
                    where n_rds_types = 3 (ards, hmrds, crds)
                          n_dotDens = 9 (0.1, 0.2, ..., 0.9)

            n_feat_rds: int, number of features per layer for
                monosemantic rds
                (default: 9 -> n_dotDens
                    where n_dotDens = 9 (0.1, 0.2, ..., 0.9)

        Returns
        -------
        tuple
            A pair (mono_score, mono_score_rds):

            mono_score : torch.FloatTensor, shape (n_interactions, n_seeds, n_layers)
                Global monosemantic score per interaction / seed / layer. .

            mono_score_rds : torch.FloatTensor, shape (n_interactions, n_seeds, n_layers, 3)
                Per-RDS-type monosemantic scores (ards, hmrds, crds) per interaction /
                seed / layer.
        """
        mono_score = torch.empty(
            (
                len(self.interactions),
                len(self.config.seed_to_analyse),
                len(self.layer_name),
            ),
            dtype=torch.float32,
        )
        mono_score_rds = torch.empty(
            (
                len(self.interactions),
                len(self.config.seed_to_analyse),
                len(self.layer_name),
                3,
            ),
            dtype=torch.float32,
        )

        for i, interaction in enumerate(self.interactions):
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

                # update model
                self.load_model()

                tik = timer()
                results = Parallel(n_jobs=8)(
                    delayed(self.compute_mono_score_layer)(
                        layer_name, threshold, n_feat_global, n_feat_rds
                    )
                    for layer_name in self.layer_name
                )
                tok = timer()
                print(tok - tik)

                # stitching results
                mono_score_layers = torch.empty(
                    len(self.layer_name), dtype=torch.float32
                )
                mono_score_rds_layers = torch.empty(
                    (len(self.layer_name), 3), dtype=torch.float32
                )
                for j in range(len(self.layer_name)):
                    mono_score_layers[j] = results[j][0]
                    mono_score_rds_layers[j] = results[j][1]

                mono_score[i, s] = mono_score_layers
                mono_score_rds[i, s] = mono_score_rds_layers

        # save
        torch.save(mono_score, os.path.join(self.data_dir, "mono_score.pt"))
        torch.save(mono_score_rds, os.path.join(self.data_dir, "mono_score_rds"))

        return mono_score, mono_score_rds

    def stat_test(
        self,
        mono_score: Float[Tensor, "interaction seed layer"],
        mono_score_rds: Float[Tensor, "interaction seed layer rds_type"],
    ) -> None:

        # create dataframe
        n_interaction, n_seed, n_layer = mono_score.shape
        seeds = np.arange(n_seed)
        layers = np.arange(19, 38)

        # covnert tensor to numpy
        arr = mono_score.cpu().numpy()  # [interacton, seed, layer]

        records = []
        for i, inter in enumerate(self.interactions):
            for s, seed in enumerate(seeds):
                for l, layer in enumerate(layers):
                    records.append(
                        {
                            "interaction": inter,
                            "seed": seed,
                            "layer_id": l + 1,  # layer index for modelling
                            "layer_name": layer,  # layer name
                            "score": arr[i, s, l],
                        }
                    )
        df = pd.DataFrame(records)

        # repeated mesures ANOVA
        anova = AnovaRM(df, "score", "seed", within=["interaction", "layer_name"]).fit()
        print(anova)

        # save ANOVA table to CSV
        anova_table = getattr(anova, "anova_table")
        anova_table.to_csv(
            os.path.join(self.group_stat_dir, "monosemanticity_anova.csv")
        )

        # mixed linear model
        model = smf.mixedlm("score ~ layer_id * C(interaction)", df, groups=df["seed"])
        result = model.fit()
        print(result.summary())

        # save numeric results (coef, std err, t, p, conf int) to CSV
        params = result.params
        bse = getattr(result, "bse", None)
        tvalues = getattr(result, "tvalues", None)
        if tvalues is None and bse is not None:
            tvalues = params / bse
        pvalues = getattr(result, "pvalues", None)
        ci = result.conf_int() if hasattr(result, "conf_int") else None

        df_result = pd.DataFrame({"coef": params})
        if bse is not None:
            df_result["std_err"] = bse
        if tvalues is not None:
            df_result["t"] = tvalues
        if pvalues is not None:
            df_result["pvalue"] = pvalues
        if ci is not None:
            df_result[["ci_lower", "ci_upper"]] = ci

        df_result.to_csv(
            os.path.join(self.group_stat_dir, "monosemanticity_mixedlm.csv")
        )

        ##################################
        ## stat test for each RDS types ##
        ##################################

        # create dataframe
        # convert tensor to numpy
        arr = mono_score_rds.cpu().numpy()  # [interacton, seed, layer]

        records = []
        for i, inter in enumerate(self.interactions):
            for s, seed in enumerate(seeds):
                for l, layer in enumerate(layers):
                    for r, rds_type in enumerate(self.rds_type):
                        records.append(
                            {
                                "interaction": inter,
                                "seed": seed,
                                "layer_index": l + 1,  # layer index for modelling
                                "layer_id": layer,  # layer name
                                "rds_type": rds_type,
                                "score": arr[i, s, l, r],
                            }
                        )
        df = pd.DataFrame(records)

        # repeated mesures ANOVA
        anova = AnovaRM(
            df, "score", "seed", within=["interaction", "layer_id", "rds_type"]
        ).fit()
        print(anova)

        # save ANOVA table to CSV
        anova_table = getattr(anova, "anova_table")
        anova_table.to_csv(
            os.path.join(self.group_stat_dir, "monosemanticity_rds_anova.csv")
        )

        # mixed linear model, the modeling here only concerns with the main effect only,
        # not the interaction. Therefore we use "+"
        model = smf.mixedlm(
            "score ~ C(rds_type) + layer_index + C(interaction)", df, groups=df["seed"]
        )
        result = model.fit()
        print(result.summary())

        # save numeric results (coef, std err, t, p, conf int) to CSV
        params = result.params
        bse = getattr(result, "bse", None)
        tvalues = getattr(result, "tvalues", None)
        if tvalues is None and bse is not None:
            tvalues = params / bse
        pvalues = getattr(result, "pvalues", None)
        ci = result.conf_int() if hasattr(result, "conf_int") else None

        df_result = pd.DataFrame({"mean-diff": params})
        if bse is not None:
            df_result["std_err"] = bse
        if tvalues is not None:
            df_result["t"] = tvalues
        if pvalues is not None:
            df_result["pvalue"] = pvalues
        if ci is not None:
            df_result[["ci_lower", "ci_upper"]] = ci

        df_result.to_csv(
            os.path.join(self.group_stat_dir, "monosemanticity_rds_mixedlm.csv")
        )

    def plot_monosemanticity_spectrum_in_layer(
        self,
        monosemanticity: Float[Tensor, "n_neuron"],
        layer_name: str,
        save_flag: bool = False,
    ):
        """
        Plot the monosemanticity spectrum for a given layer.

        input args:
            monosemanticity: Float[Tensor, "n_neuron"], monosemanticity values
                for each neuron in the layer

            layer_name: str, name of the layer to plot

            save_flag: bool, whether to save the figure (default: False)
        """

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 5)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Monosemanticity spectrum ({layer_name})",
            ha="center",
        )
        # fig.text(-0.01, 0.5, "Monosemanticity", va="center", rotation=90)
        # fig.text(0.5, -0.01, "Neuron", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        axes.stem(monosemanticity, linefmt="black", basefmt="k-")

        y_up = 1.1
        y_low = 0.0
        y_step = 0.2
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlabel("Neuron_index")
        axes.set_ylabel("Monosemanticity")

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag:
            plt.savefig(
                f"{self.monosemanticity_dir}/monosemanticity_{layer_name}.pdf",
                dpi=600,
                bbox_inches="tight",
            )

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()

    def plotLine_n_mono_vs_layer(
        self, threshold: float = 0.5, n_features: float = 54, save_flag: bool = True
    ):
        """
        Plot n monosemantic neurons per feature as a function of layer.

        input args:
            threshold: float, threshold for monosemanticity
                (default: 0.5)

            n_features: int, number of features per layer
                (default: 27 -> n_rds_types * n_dotDens
                    where n_rds_types = 3 (ards, hmrds, crds)
                          n_dotDens = 9 (0.1, 0.2, ..., 0.9)

                          the "2" comes from crossed and uncrossed disparities

            save_flag: bool, whether to save the figure (default: False)
        """

        mono_per_feat = torch.empty(len(self.layer_name))
        for i in range(len(self.layer_name)):
            layer_name = self.layer_name[i]
            monosemanticity = self.compute_monosemanticity(layer_name)

            # threshold the monosemanticity
            mono_thresh = torch.where(monosemanticity > threshold, 1.0, 0.0)
            mono_per_feat[i] = mono_thresh.sum() / n_features

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (12, 5)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Monosemantic neurons per feature across Layer\n"
            + f"({self.binocular_interaction}, seed={self.seed})",
            ha="center",
        )
        # fig.text(-0.01, 0.5, "Monosemanticity", va="center", rotation=90)
        # fig.text(0.5, -0.01, "Neuron", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        axes.plot(mono_per_feat, linewidth=2, marker="o", color="black")

        y_up = 1.1
        y_low = 0.0
        y_step = 0.2
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlabel("Layer")
        axes.set_ylabel("n_mono/n_feat")
        axes.set_xticks(np.arange(len(self.layer_name)))
        axes.set_xticklabels(self.layer_name, rotation=45, ha="right")
        axes.set_xlim(-0.5, len(self.layer_name) - 0.5)

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag:
            plt.savefig(
                f"{self.monosemanticity_dir}/monosemanticity_vs_layer.pdf",
                dpi=600,
                bbox_inches="tight",
            )

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")
        plt.close(fig)
        gc.collect()

    def plotLine_n_mono_vs_layer_all_interactions(
        self,
        mono_score_global: Float[Tensor, "n_interactions n_seeds n_layers"],
        save_flag: bool = True,
    ):
        """
        Plot n monosemantic neurons per feature as a function of layer,
        for all binocular interactions.

        input args:
            interaction: str, binocular interaction to analyze
                (e.g., "default", "bem", "cmm", "sum_diff")

            threshold: float, threshold for monosemanticity
                (default: 0.9)

            n_features: int, number of features per layer
                (default: 27 -> n_rds_types * n_dotDens
                    where n_rds_types = 3 (ards, hmrds, crds)
                          n_dotDens = 9 (0.1, 0.2, ..., 0.9)

            save_flag: bool, whether to save the figure (default: False)
        """

        # average across seeds
        mono_avg = mono_score_global.mean(dim=1)  # [interaction, n_layers]
        mono_sem = sem(mono_score_global, axis=1)  # [interaction, n_layers]

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (12, 8)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Monosemantic neurons per feature across Layer\navg across seeds",
            ha="center",
        )
        # fig.text(-0.01, 0.5, "Monosemanticity", va="center", rotation=90)
        # fig.text(0.5, -0.01, "Neuron", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]
        for i in range(len(self.interactions)):
            axes.plot(
                mono_avg[i], linewidth=2, marker="o", color=colors[i], label=labels[i]
            )
            axes.fill_between(
                np.arange(len(mono_avg[i])),
                mono_avg[i] - mono_sem[i],
                mono_avg[i] + mono_sem[i],
                color=colors[i],
                alpha=0.2,
            )

        y_up = 0.41
        y_low = 0.0
        y_step = 0.1
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlabel("Layer")
        axes.set_ylabel("n_mono/n_feat")
        axes.set_xticks(np.arange(len(self.layer_name)))
        axes.set_xticklabels(self.layer_name, rotation=45, ha="right")
        axes.set_xlim(-0.5, len(self.layer_name) - 0.5)

        plt.legend(
            loc="upper left",
            fontsize=20,
            frameon=False,
        )

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag:
            plt.savefig(
                f"{self.group_dir}/plots/monosemanticity_vs_layer_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_n_mono_vs_layer_all_interactions_rds(
        self,
        mono_score_rds: Float[Tensor, "n_interactions n_seeds n_layers rds_type"],
        rds_type: str,
        save_flag: bool = True,
    ):
        """
        Plot n monosemantic neurons per feature as a function of layer,
        for all rds_types and all binocular interactions.

        input args:
            rds_type: str, type of rds (ards, hmrds, or crds)

            interaction: str, binocular interaction to analyze
                (e.g., "default", "bem", "cmm", "sum_diff")

            threshold: float, threshold for monosemanticity
                (default: 0.9)

            n_features: int, number of features per layer
                (default: 18 -> n_dotDens * 2
                    where n_dotDens = 9 (0.1, 0.2, ..., 0.9)

                        the "2" comes from crossed and uncrossed disparities

            save_flag: bool, whether to save the figure (default: False)
        """
        # get rds index
        if rds_type == "ards":
            rds_index = 0
        elif rds_type == "hmrds":
            rds_index = 1
        elif rds_type == "crds":
            rds_index = 2

        # average across seeds
        mono_avg = mono_score_rds[:, :, :, rds_index].mean(
            dim=1
        )  # [interaction, n_layers]
        mono_sem = sem(
            mono_score_rds[:, :, :, rds_index], axis=1
        )  # [interaction, n_layers]

        # start plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (12, 8)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Monosemantic neurons per feature across Layer\navg across seeds ({rds_type})",
            ha="center",
        )
        # fig.text(-0.01, 0.5, "Monosemanticity", va="center", rotation=90)
        # fig.text(0.5, -0.01, "Neuron", ha="center")

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.6)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]
        for i in range(len(self.interactions)):
            axes.plot(
                mono_avg[i], linewidth=2, marker="o", color=colors[i], label=labels[i]
            )
            axes.fill_between(
                np.arange(len(mono_avg[i])),
                mono_avg[i] - mono_sem[i],
                mono_avg[i] + mono_sem[i],
                color=colors[i],
                alpha=0.2,
            )

        y_up = 0.41
        y_low = 0.0
        y_step = 0.1
        axes.set_yticks(np.arange(y_low, y_up, y_step))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlabel("Layer")
        axes.set_ylabel("n_mono/n_feat")
        axes.set_xticks(np.arange(len(self.layer_name)))
        axes.set_xticklabels(self.layer_name, rotation=45, ha="right")
        axes.set_xlim(-0.5, len(self.layer_name) - 0.5)

        plt.legend(
            loc="upper left",
            fontsize=20,
            frameon=False,
        )

        # remove top and right frame
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # show ticks on the left and bottom axis
        axes.xaxis.set_ticks_position("bottom")
        axes.yaxis.set_ticks_position("left")

        if save_flag:
            plt.savefig(
                f"{self.group_dir}/plots/monosemanticity_vs_layer_all_interactions_{rds_type}.pdf",
                dpi=600,
                bbox_inches="tight",
            )
