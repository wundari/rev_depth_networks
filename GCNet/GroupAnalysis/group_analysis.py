# %% load necessary modules
import torch
from torch import Tensor
from torch import nn

from modules.gcnet import build_gcnet

from config.config import GCNetconfig
import os


# %%
class GA:
    def __init__(self, config: GCNetconfig, params_rds: dict):

        self.model_name = config.model_name
        self.config = config
        self.dataset = config.dataset
        self.binocular_interaction = config.binocular_interaction
        self.interactions = ["default", "bem", "cmm", "sum_diff"]
        self.interactions = config.interactions
        self.seed = config.seed
        self.epoch_to_load = config.epoch_to_load
        self.iter_to_load = config.iter_to_load
        self.device = config.device

        # rds parameters
        self.target_disp = params_rds[
            "target_disp"
        ]  # RDS target disparity (pix) to be analyzed
        self.n_rds_each_disp = params_rds[
            "n_rds_each_disp"
        ]  # n_rds for each disparity magnitude in disp_ct_pix
        self.dotDens_list = params_rds["dotDens_list"]  # dot density
        self.rds_type = params_rds[
            "rds_type"
        ]  # ards: 0.0, crds: 1.0, hmrds: 0.5, urds: -1.0
        self.rds_types = {
            "ards": 0.0,
            "hmrds": 0.5,
            "crds": 1.0,
        }  # {rds_type: dotMatch}
        self.dotMatch_list = params_rds["dotMatch_list"]  # [0.0, 0.5, 1.0] dot match
        self.background_flag = params_rds["background_flag"]  # 1: with cRDS background
        self.pedestal_flag = params_rds[
            "pedestal_flag"
        ]  # 1: use pedestal to ensure rds disparity > 0
        self.batch_size_rds = params_rds[
            "batch_size_rds"
        ]  # batch size for RDS generation
        self.disp_ct_pix_list = [self.target_disp, -self.target_disp]

        # set up experiment directory
        self.experiment_dir = (
            f"run/{self.dataset}/"
            + f"bino_interaction_{self.binocular_interaction}/"
            + f"{self.seed}"
        )

        # create folder for saving plots of a given interaction
        # (all seeds are combined)
        self.plot_dir = f"{self.experiment_dir}/../plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # create folder for saving group analysis
        # (all interactions are combined)
        self.group_dir = f"run/{self.dataset}/bino_interaction_group"
        if not os.path.exists(self.group_dir):
            os.makedirs(self.group_dir)

        # create folder for saving group analysis plots
        # (all interactions are combined)
        self.group_plot_dir = f"{self.group_dir}/plots"
        if not os.path.exists(self.group_plot_dir):
            os.makedirs(self.group_plot_dir)

        # create folder for saving group analysis statistical tests
        # (all interactions are combined)
        self.group_stat_dir = f"{self.group_dir}/stats"
        if not os.path.exists(self.group_stat_dir):
            os.makedirs(self.group_stat_dir)

        # load model
        if config.load_state:
            self.load_model()
        else:  # build from scratch
            self.model = build_gcnet(config)
            self.model.to(self.device)
            self.model.eval()
            print(
                f"{self.model_name} was built from scratch and successfully loaded to {self.device}.\n"
                + f"{self.model_name} is in eval mode"
            )

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

        # # target_layer, only convolutional layers
        # self.target_layer = [
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
        #     self.model.decoder.layer33a[0],  # convT3d
        #     self.model.decoder.layer34a[0],  # convT3d
        #     self.model.decoder.layer35a[0],  # convT3d
        #     self.model.decoder.layer36a[0],  # convT3d
        #     self.model.decoder.layer37,  # convT3d
        # ]

        # target_layer, only convolutional layers
        self.target_layer = [
            self.model.encoder.layer1[0],
            self.model.encoder.res_block[0].conv2d_block1[0],
            self.model.encoder.res_block[0].conv2d_block2[0],
            self.model.encoder.res_block[1].conv2d_block1[0],
            self.model.encoder.res_block[1].conv2d_block2[0],
            self.model.encoder.res_block[2].conv2d_block1[0],
            self.model.encoder.res_block[2].conv2d_block2[0],
            self.model.encoder.res_block[3].conv2d_block1[0],
            self.model.encoder.res_block[3].conv2d_block2[0],
            self.model.encoder.res_block[4].conv2d_block1[0],
            self.model.encoder.res_block[4].conv2d_block2[0],
            self.model.encoder.res_block[5].conv2d_block1[0],
            self.model.encoder.res_block[5].conv2d_block2[0],
            self.model.encoder.res_block[6].conv2d_block1[0],
            self.model.encoder.res_block[6].conv2d_block2[0],
            self.model.encoder.res_block[7].conv2d_block1[0],
            self.model.encoder.res_block[7].conv2d_block2[0],
            self.model.encoder.layer18,
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
            self.model.decoder.layer33a[0],  # convT3d
            self.model.decoder.layer34a[0],  # convT3d
            self.model.decoder.layer35a[0],  # convT3d
            self.model.decoder.layer36a[0],  # convT3d
            self.model.decoder.layer37,  # convT3d
        ]

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

        print(
            "Updating network config\n"
            + f"binocular interaction: {interaction_old} => {self.config.binocular_interaction}\n"
            + f"seed: {seed_old} => {self.config.seed}\n"
            + f"epoch: {epoch_old} => {self.config.epoch_to_load}\n"
            + f"iter: {iter_old} => {self.config.iter_to_load}\n"
            + f"experiment_dir: {self.experiment_dir}\n"
        )

    def load_model(self) -> None:
        """
        Load the model state from a checkpoint (pre-trained model).
        """

        # build model
        self.model = build_gcnet(self.config)

        # load model state from checkpoint
        resume = (
            f"epoch_{self.epoch_to_load}_iter_{self.iter_to_load}_model_best.pth.tar"
        )
        resume_path = os.path.join(self.experiment_dir, resume)
        checkpoint = torch.load(f"{resume_path}", map_location=self.device)
        pretrained_dict = checkpoint["state_dict"]

        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(pretrained_dict.items()):
            if k.startswith(unwanted_prefix):
                pretrained_dict[k[len(unwanted_prefix) :]] = pretrained_dict.pop(k)

        self.model.load_state_dict(pretrained_dict)
        self.model.to(self.device)
        self.model.eval()

        # reset target layer names, important for hooking.
        # every time a new model is loaded, the new target layers
        # seem to have different memory addresses from the old ones,
        # although they shared the same names. Therefore, they
        # need to be reset.
        # This is important for hooking, as the hooks are attached
        # to the target layers possibly by their memory addresses.
        self.target_layer = [
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
            self.model.decoder.layer33a[0],  # convT3d
            self.model.decoder.layer34a[0],  # convT3d
            self.model.decoder.layer35a[0],  # convT3d
            self.model.decoder.layer36a[0],  # convT3d
            self.model.decoder.layer37,  # convT3d
        ]

        print(
            f"{self.model_name} was successfully loaded to {self.device}. \n"
            + f"{self.model_name} model: {resume_path}\n"
            + f"binocular interaction: {self.binocular_interaction}\n"
            + f"compile mode: {self.config.compile_mode}\n"
            + f"experiment dir: {self.experiment_dir}\n"
        )

    def get_conv_names_and_weights(self) -> tuple[list[Tensor], list[Tensor]]:
        """
        Get the names and weights of convolutional layers in the model.

        the original weight matrix is of shape [C_out, C_in, h, w]
        (for 2D convolution) or [C_out, C_in, d, h, w] for
        3D convolution. See:
        https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

        HOWEVER, in the case of convTranspose3D such
        as in decoder.layer4, the weight matrix is of shape
        [C_in, C_out, d, h, w] (please see this documentation:
        https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html)]

        the input dimension (C_in) is associated with the number
        of features (n_feat),
        and the layer dimension (C_out) is associated with the number
        of neurons (n_neuron).
        Thus, the dimension becomes [n_neuron, n_feat].

        Returns:
            conv_layer_names (list): List of names of convolutional layers.
            conv_layer_weights (list): List of weights of convolutional layers.
        """

        # names = []
        # params = []
        # for name, param in ga_mono.model.named_parameters():
        #     names.append(name)
        #     params.append(param)

        # # get index for convolutional layers, only decoder
        # layer_idx = []
        # for i in range(len(names)):  # gather the first 3 conv layers in [names]
        #     if (
        #         "decoder" in names[i] and ".0.weight" in names[i]
        #     ) or "layer37.weight" in names[i]:
        #         layer_idx.append(i)

        # get convolutional layer names and weights, only decoder
        conv_layer_names = []
        conv_layer_weights = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv3d):
                conv_layer_names.append(name)
                conv_layer_weights.append(module.weight)
            elif isinstance(module, nn.ConvTranspose3d):
                conv_layer_names.append(name)
                conv_layer_weights.append(module.weight.transpose(0, 1))

        # for i in layer_idx:
        #     conv_layer_names.append(names[i])

        #     if isinstance(temp[1])

        #     # if layer4, transpose the weights because it is ConvTranspose3D
        #     # whose original shape is [C_in, C_out, d, h, w]
        #     # see: https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
        #     if "layer4" in names[i]:
        #         conv_layer_weights.append(params[i].transpose(0, 1))
        #     else:
        #         conv_layer_weights.append(params[i])

        return (conv_layer_names, conv_layer_weights)
