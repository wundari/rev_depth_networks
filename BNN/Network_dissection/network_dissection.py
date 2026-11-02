# %% Network dissection:

# https://www.pnas.org/doi/epdf/10.1073/pnas.1907375117
# how many neurons activate a given disparity?
# 1. make a scatter plot neuron density vs disparity (possibly
# for neurons whose activation is above a certain threshold)
# 2. look for the effect of removing important units
# 3. disparity tuning of each neuron in each layer. observe which neurons
#    have cross-correlation / cross-matching tuning profiles,

# possible questions:
# 1. What are the roles of each unit in NN?
# 2. what units in the network contribute to reversed depth?
# 3. what semantic that constribute to reversed depth?
# 4. does disparity map depends on semantic (specifically, scenery concept)?
# 5. Lesion analysis: which neurons/channels/layers that change the
# behavior of reversed depth?
# 6. feature channel attribution: how much did each feature detector
# contribute to the final output?
# 7. disparity channel attribution: how much did each disparity detector
# contribute to the final output?
# 8. which dataset that contribute to reversed depth the most?

# %% load necessary modules
import torch
from torch import nn
import torchvision
import torch._dynamo

# settings for pytorch 2.0 compile
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from typing import Optional, assert_type
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append("captum")
import captum.optim as optimviz

from data_handler.DataHandler_SceneFlow_v2 import *
from RDS.DataHandler_RDS import *
from engine.model import GCNet
from Optimization.loss import (
    NeuronActivation,
    ChannelActivation,
    FeatureChannelActivation,
    DisparityChannelActivation,
    LayerActivation,
)
from utils.utils import *
from utils.output_hook import ModuleOutputsHook
from utils.typing import LossFunction
from Common.network_correlation import *

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
class NetworkDissection:
    """
    NetworkDissection is a class designed for dissecting and analyzing
    convolutional neural networks,
    particularly in the context of scene flow estimation.
    This class offers methods for loading a model,
    setting up image transformations, and optimizing neuron activations
    to understand model behavior.

    Attributes:
        w_bg (int): Width of the background.

        h_bg (int): Height of the background.

        n_bootstrap (int): Number of bootstrapping iterations.

        batch_size (int): Batch size for processing.

        maxdisp (int): Maximum disparity, should be a multiple of 32.

        loss_mul (list): List for storing loss multipliers.

        epoch_to_load (int): The epoch number at which the model is loaded.

        iter_to_load (int): The iteration number at which the model is loaded.

        disp_mag (int): Magnitude of the disparity.

        c_disp_shift (float): Multiplier for disparity shift.

        dataset_to_process (str): Type of the dataset to process, e.g.,
        "driving", "flying", "monkaa".

        transform (torch.nn.Sequential): Image transformation pipeline.

        load_model (function): Method for loading the model.

        save_dir (str): Directory for saving results.

        network_diss_dir (str): Directory for saving network dissection results.

    Methods:
        dataset_to_process: Getter and setter for `dataset_to_process` attribute.
        loss_mul: Getter and setter for `loss_mul` attribute.
        load_model: Getter and setter for loading and managing the model.
        get_activations_shape(target: nn.Module):
            Gets the shape of activations for a given model layer.
        default_loss_summarize(loss_value: torch.Tensor):
            Summarizes loss values for optimization.
        optimize_neuron_activation:
            Optimizes activation of a specific neuron in a layer.
        optimize:
            Generic optimization function for optimizing neuron activations.
        optimize_all_neurons_in_single_layer:
            Optimizes all neurons in a single layer.
        optimize_channel_activation:
            Optimizes activation of a specific channel.
        optimize_channel_activation_all:
            Optimizes all channels in a layer.
        optimize_feature_channel_activation:
            Optimizes activation of a specific feature channel.
        optimize_feature_channel_activation_all:
            Optimizes all feature channels in a layer.
        optimize_disparity_channel_activation:
            Optimizes activation of a specific disparity channel.
        optimize_disparity_channel_activation_all:
            Optimizes all disparity channels in a layer.
        optimize_layer_activation:
            Optimizes activation of an entire layer.
        optimize_layer_activation_all: Optimizes all layers in the model.

    Args:
        sceneflow_type (str): Type of scene flow dataset to use.
        epoch_to_load (int): Epoch number for loading the model.
        iter_to_load (int): Iteration number for loading the model.
        disp_mag (int): Magnitude of disparity.
        c_disp_shift (float): Disparity shift multiplier.
    """

    def __init__(self, params_network, params_train) -> None:
        # network parameter
        self.w_bg = params_network["w_bg"]
        self.h_bg = params_network["h_bg"]
        self.maxdisp = params_network["maxdisp"]

        # training parameter
        self.n_bootstrap = 1000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

        # construct disparity indices for normalizing the probability of
        # cost volume across disparity axis (see eq. 1 in the Kendall 2017 paper)
        self.disp_indices = []

        # parameters for reading the model file
        # sceneflow_type: "driving", "flying", "monkaa"
        # for ex:
        # dataset names: "driving", "flying", "monkaa"
        # epoch_to_load (earlystop): driving: 4; flying: 4; monkaa: 5
        # iter_to_load (earlystop): driving: 6801; flying: 29201; monkaa: 17201
        self.epoch_to_load = params_train["epoch_to_load"]
        self.iter_to_load = params_train["iter_to_load"]
        self.batch_size = params_train["batch_size"]  # training batch size
        self.c_disp_shift = params_train["c_disp_shift"]  # disparity shift multiplier
        self.dataset_to_process = params_train["sceneflow_type"]
        self.flip_input = params_train["flip_input"]
        self.target_disp = params_train["target_disp"]

        # prepare saving directory
        if self.flip_input:
            self.save_dir = (
                "results/sceneflow/"
                + self.dataset_to_process
                + f"/shift_{self.c_disp_shift}_median_wrt_right"
            )
        else:
            self.save_dir = (
                "results/sceneflow/"
                + self.dataset_to_process
                + f"/shift_{self.c_disp_shift}_median_wrt_left"
            )

        self.network_diss_dir = (
            f"{self.save_dir}/"
            + f"epoch_{self.epoch_to_load}"
            + f"_iter_{self.iter_to_load}/"
            + "network_dissection/"
            + f"target_disp_{self.target_disp}px"
        )
        if not os.path.exists(self.network_diss_dir):
            os.makedirs(self.network_diss_dir)

        self.neuron_dir = self.network_diss_dir + "/neuron_optimization"
        if not os.path.exists(self.neuron_dir):
            os.makedirs(self.neuron_dir)

        self.channel_dir = self.network_diss_dir + "/channel_optimization"
        if not os.path.exists(self.channel_dir):
            os.makedirs(self.channel_dir)

        self.layer_dir = self.network_diss_dir + "/deep_dream_optimization"
        if not os.path.exists(self.layer_dir):
            os.makedirs(self.layer_dir)

        # load model
        self.load_model = None

        # define image transformation for input optimization
        self.transform = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(16),
            optimviz.transforms.RandomSpatialJitter(16),
            optimviz.transforms.RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05)),
            torchvision.transforms.RandomRotation(degrees=(-5, 5)),
            optimviz.transforms.RandomSpatialJitter(8),
            optimviz.transforms.CenterCrop((self.h_bg, self.w_bg)),
        )

        self.target_name = [
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

        self.target_list = [
            self.model.layer19[0],  # only convolutional layer
            self.model.layer20[0],
            self.model.layer21[0],
            self.model.layer22[0],
            self.model.layer23[0],
            self.model.layer24[0],
            self.model.layer25[0],
            self.model.layer26[0],
            self.model.layer27[0],
            self.model.layer28[0],
            self.model.layer29[0],
            self.model.layer30[0],
            self.model.layer31[0],
            self.model.layer32[0],
            self.model.layer33a[0],
            self.model.layer34a[0],
            self.model.layer35a[0],
            self.model.layer36a[0],
            self.model.layer37,
        ]

    @property
    def disp_indices(self):
        return self._disp_indices

    @disp_indices.setter
    def disp_indices(self, disp_indices_list):
        """
        create disparity indices used for calculating the expected value
        of model's output to regress disparity (see eq. 1 in Kendall 2017 paper)
        """
        disp_indices_list = [
            d * torch.ones((1, self.h_bg, self.w_bg))
            for d in range(-self.maxdisp // 2, self.maxdisp // 2, 1)
        ]
        self._disp_indices = (
            torch.cat(disp_indices_list, 0)
            .pin_memory()
            .to(self.device, non_blocking=True)
        )

    @property
    def load_model(self):
        return self.model

    @load_model.setter
    def load_model(self, model):
        # load pre-trained gcnet
        self.model = GCNet(self.h_bg, self.w_bg, self.maxdisp)
        checkpoint = torch.load(
            f"{self.save_dir}/checkpoint/"
            + f"gcnet_state_earlystop_epoch_{self.epoch_to_load}"
            + f"_iter_{self.iter_to_load}.pth"
        )
        self.checkpoint = checkpoint
        # fix the keys of the state dictionary
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        # self.model = torch.compile(self.model)  # don't use for layer analysis
        self.model.to(self.device)
        self.model.eval()  # evaluation mode

        print(f"load GC-Net successfully to {self.device} and in eval mode")

    def get_activations_shape(self, target: nn.Module):
        # get layer activation to get layer dimension
        # define hook
        if isinstance(target, list):
            hook = ModuleOutputsHook(target)
        else:
            hook = ModuleOutputsHook([target])
            # hook = ModuleOutputsHook([model.layer36a])

        # define noise image for optimization
        image_left_t = optimviz.images.NaturalImage((self.h_bg, self.w_bg)).to(
            self.device
        )
        image_right_t = optimviz.images.NaturalImage((self.h_bg, self.w_bg)).to(
            self.device
        )

        # compute model's output.
        logits = self.model(image_left_t(), image_right_t())

        # consume_outputs return the captured values and resets the hook's state
        # compute module output
        module_outputs = hook.consume_outputs()
        activations = module_outputs[target]
        # activations = module_outputs[model.layer36a]

        hook.remove_hooks()

        # get dimension
        n_batch, n_feature_channels, n_disp_channels, h, w = activations.shape

        return n_feature_channels, n_disp_channels, h, w

    @staticmethod
    def default_loss_summarize(loss_value: torch.Tensor) -> torch.Tensor:
        """
        Helper function to summarize tensor outputs from loss functions.

        default_loss_summarize applies `mean` to the loss tensor
        and negates it so that optimizing it maximizes the activations we
        are interested in.
        """
        lambda_reg = 0.0
        return -1 * (loss_value.mean() + lambda_reg * (loss_value**2).sum())

    def optimize_neuron_activation(
        self,
        target: nn.Module,
        feature_channel_index: int,
        disp_channel_index: int,
        neuron_row_index: int,
        neuron_col_index: int,
        batch_index: Optional[int] = None,
        n_iters: int = 128,
        lr: float = 0.025,
    ):
        # define loss function
        loss_fn = NeuronActivation(
            target,
            feature_channel_index,
            disp_channel_index,
            neuron_row_index,
            neuron_col_index,
        )

        # define noise image for optimization
        input_param_left = optimviz.images.NaturalImage((self.h_bg, self.w_bg))
        input_param_right = optimviz.images.NaturalImage((self.h_bg, self.w_bg))

        # move to gpu
        self.input_param_left = input_param_left.to(self.device)
        self.input_param_right = input_param_right.to(self.device)

        # define objective function
        # obj = optimviz.InputOptimization(
        #     self.model, loss_fn, image_left, image_right, transforms
        # )

        # define optimizer
        # lr = 0.025
        # optimizer = torch.optim.Adam(obj.parameters(), lr=lr)
        params = list(self.input_param_left.parameters()) + list(
            self.input_param_right.parameters()
        )
        optimizer = torch.optim.Adam(params, lr=lr)

        # start optimizing
        # n_iters = 128
        loss_history = self.optimize(loss_fn, optimizer, n_iters)

        # fetch the optimized inputs and swap their axis to [h, w, rgb_channels]
        img_left_optimized = self.input_param_left()[0].permute(1, 2, 0).detach()
        img_right_optimized = self.input_param_right()[0].permute(1, 2, 0).detach()

        return img_left_optimized, img_right_optimized, loss_history

    def optimize(
        self,
        loss_fn: LossFunction,
        optimizer: Optional[torch.optim.Optimizer],
        n_iters: int,
    ):
        # define hook
        if isinstance(loss_fn.target, list):
            hook = ModuleOutputsHook(loss_fn.target)
        else:
            hook = ModuleOutputsHook([loss_fn.target])

        pbar = tqdm(total=n_iters, unit="step")
        history = []
        step = 0
        try:
            # while step < n_iters:
            for step in range(n_iters):
                optimizer.zero_grad()

                # get and transform input images
                image_left_t = self.transform(self.input_param_left())
                image_right_t = self.transform(self.input_param_right())
                # image_left_t = nd.transform(image_left_t)
                # image_right_t = nd.transform(image_right_t)

                # compute model's output.
                logits = self.model(image_left_t, image_right_t)
                # logits = nd.model(image_left_t, image_right_t)

                # consume_outputs return the captured values and resets the hook's state
                # compute module output
                module_outputs = hook.consume_outputs()
                # activations = module_outputs[loss_fn.target]

                # compute loss value.
                # essentially this is the activation of the target neuron
                loss_value = self.default_loss_summarize(loss_fn(module_outputs))
                # loss_value = self.default_loss_summarize(
                #     activations[
                #         0,  # batch index
                #         channel_index,
                #         disp_channel_index,
                #         neuron_row_idx,
                #         neuron_col_idx,
                #     ]
                # )
                history.append(loss_value.clone().detach())
                # history.append(loss_value.item())

                # update parameters
                loss_value.backward()
                optimizer.step()

                # update progress bar
                # pbar.set_postfix(
                #         {"Objective": f"{history[-1].mean():.1f}"}, refresh=False
                #     )
                # if step < n_iters:
                #     pbar.update()
                # else:
                #     pbar.close()

                if step % 10 == 0 or step == n_iters - 1:
                    pbar.set_postfix(
                        {"Objective": f"{history[-1].mean():.1f}"}, refresh=False
                    )
                    pbar.update(min(10, n_iters - step))

                # step += 1
            pbar.close()
        finally:
            hook.remove_hooks()

        return torch.stack(history)

    def optimize_all_neurons_in_single_layer(
        self,
        target: nn.Module,
        n_iters: int = 128,
    ):
        # get layer activation to get layer dimension
        n_feature_channels, n_disp_channels, h, w = self.get_activations_shape(target)

        # coordinate of target neurons
        neuron_coordinates = [h // 2, w // 2]
        img_left_layer = torch.empty(
            (
                n_feature_channels,
                n_disp_channels,
                self.h_bg,
                self.w_bg,
                3,
            ),
            dtype=torch.float32,
        )
        img_right_layer = torch.empty(
            (
                n_feature_channels,
                n_disp_channels,
                self.h_bg,
                self.w_bg,
                3,
            ),
            dtype=torch.float32,
        )
        loss_history_layer = torch.empty(
            (
                n_feature_channels,
                n_disp_channels,
                n_iters,
            ),
            dtype=torch.float32,
        )

        neuron_row_index = neuron_coordinates[0]
        neuron_col_index = neuron_coordinates[1]
        for c in range(n_feature_channels):
            for d in range(n_disp_channels):
                print(
                    f"optimize neuron activation at "
                    + f"feature_channel_idx: {c}/{n_feature_channels}, "
                    + f"disp_channel_idx: {d}/{n_disp_channels}, "
                    + f"neuron_coordinate: ({neuron_row_index}, {neuron_col_index})"
                )
                (
                    img_left_optimized,
                    img_right_optimized,
                    loss,
                ) = self.optimize_neuron_activation(
                    target,
                    c,  # feature_channel_index,
                    d,  # disp_channel_index,
                    neuron_row_index,  # neuron_row_index
                    neuron_col_index,  # neuron_col_index
                    n_iters,
                )

                img_left_layer[c, d] = img_left_optimized
                img_right_layer[c, d] = img_right_optimized
                loss_history_layer[c, d] = loss

        return img_left_layer, img_right_layer, loss_history_layer

    def optimize_channel_activation(
        self,
        target: nn.Module,
        feature_channel_index: int,
        disp_channel_index: int,
        n_iters: int = 128,
        lr: float = 0.025,
    ):
        # define loss function
        loss_fn = ChannelActivation(target, feature_channel_index, disp_channel_index)

        # define noise image for optimization
        input_param_left = optimviz.images.NaturalImage((self.h_bg, self.w_bg))
        input_param_right = optimviz.images.NaturalImage((self.h_bg, self.w_bg))

        # move to gpu
        self.input_param_left = input_param_left.to(self.device)
        self.input_param_right = input_param_right.to(self.device)

        # define optimizer
        # lr = 0.025
        # optimizer = torch.optim.Adam(obj.parameters(), lr=lr)
        params = list(self.input_param_left.parameters()) + list(
            self.input_param_right.parameters()
        )
        optimizer = torch.optim.Adam(params, lr=lr)

        # start optimizing
        loss_history = self.optimize(loss_fn, optimizer, n_iters)

        # fetch the optimized inputs and swap their axis to [h, w, rgb_channels]
        img_left_optimized = self.input_param_left()[0].permute(1, 2, 0).detach()
        img_right_optimized = self.input_param_right()[0].permute(1, 2, 0).detach()

        return img_left_optimized, img_right_optimized, loss_history

    def optimize_channel_activation_all(
        self,
        target: nn.Module,
        layer_name: str,
        n_iters: int = 128,
        lr: float = 0.025,
    ):
        n_feature_channels, n_disp_channels, _, _ = self.get_activations_shape(target)

        img_left_layer = torch.empty(
            (n_feature_channels, n_disp_channels, self.h_bg, self.w_bg, 3),
            dtype=torch.float32,
        )
        img_right_layer = torch.empty(
            (n_feature_channels, n_disp_channels, self.h_bg, self.w_bg, 3),
            dtype=torch.float32,
        )
        loss_history_layer = torch.empty(
            (n_feature_channels, n_disp_channels, n_iters), dtype=torch.float32
        )
        for c in range(n_feature_channels):
            for d in range(n_disp_channels):
                print(
                    f"optimize layer activation in {layer_name},"
                    + f"feature_layer: {c}/{n_feature_channels}, "
                    + f"disp_layer: {d}/{n_disp_channels}"
                )
                (
                    img_left_layer[c, d],
                    img_right_layer[c, d],
                    loss_history_layer[c, d],
                ) = self.optimize_channel_activation(target, c, d, n_iters, lr)

        return img_left_layer, img_right_layer, loss_history_layer

    def optimize_feature_channel_activation(
        self,
        target: nn.Module,
        feature_channel_index: int,
        n_iters: int = 128,
        lr: float = 0.025,
    ):
        # define loss function
        loss_fn = FeatureChannelActivation(target, feature_channel_index)

        # define noise image for optimization
        input_param_left = optimviz.images.NaturalImage((self.h_bg, self.w_bg))
        input_param_right = optimviz.images.NaturalImage((self.h_bg, self.w_bg))

        # move to gpu
        self.input_param_left = input_param_left.to(self.device)
        self.input_param_right = input_param_right.to(self.device)

        # define optimizer
        # lr = 0.025
        # optimizer = torch.optim.Adam(obj.parameters(), lr=lr)
        params = list(self.input_param_left.parameters()) + list(
            self.input_param_right.parameters()
        )
        optimizer = torch.optim.Adam(params, lr=lr)

        # start optimizing
        loss_history = self.optimize(loss_fn, optimizer, n_iters)

        # fetch the optimized inputs and swap their axis to [h, w, rgb_channels]
        img_left_optimized = self.input_param_left()[0].permute(1, 2, 0).detach()
        img_right_optimized = self.input_param_right()[0].permute(1, 2, 0).detach()

        return img_left_optimized, img_right_optimized, loss_history

    def optimize_feature_channel_activation_all(
        self,
        target: nn.Module,
        layer_name: str,
        n_iters: int = 128,
        lr: float = 0.025,
    ):
        n_feature_channels, _, _, _ = self.get_activations_shape(target)

        img_left_layer = torch.empty(
            (n_feature_channels, self.h_bg, self.w_bg, 3), dtype=torch.float32
        )
        img_right_layer = torch.empty(
            (n_feature_channels, self.h_bg, self.w_bg, 3), dtype=torch.float32
        )
        loss_history_layer = torch.empty(
            (n_feature_channels, n_iters), dtype=torch.float32
        )
        for c in range(n_feature_channels):
            print(
                f"optimize feature layer activation in {layer_name},"
                + f"feature_layer: {c}/{n_feature_channels}"
            )
            (
                img_left_layer[c],
                img_right_layer[c],
                loss_history_layer[c],
            ) = self.optimize_feature_channel_activation(target, c, n_iters, lr)

        return img_left_layer, img_right_layer, loss_history_layer

    def optimize_disparity_channel_activation(
        self,
        target: nn.Module,
        disp_channel_index: int,
        n_iters: int = 128,
        lr: float = 0.025,
    ):
        # define loss function
        loss_fn = DisparityChannelActivation(target, disp_channel_index)

        # define noise image for optimization
        input_param_left = optimviz.images.NaturalImage((self.h_bg, self.w_bg))
        input_param_right = optimviz.images.NaturalImage((self.h_bg, self.w_bg))

        # move to gpu
        self.input_param_left = input_param_left.to(self.device)
        self.input_param_right = input_param_right.to(self.device)

        # define optimizer
        # lr = 0.025
        # optimizer = torch.optim.Adam(obj.parameters(), lr=lr)
        params = list(self.input_param_left.parameters()) + list(
            self.input_param_right.parameters()
        )
        optimizer = torch.optim.Adam(params, lr=lr)

        # start optimizing
        loss_history = self.optimize(loss_fn, optimizer, n_iters)

        # fetch the optimized inputs and swap their axis to [h, w, rgb_channels]
        img_left_optimized = self.input_param_left()[0].permute(1, 2, 0).detach()
        img_right_optimized = self.input_param_right()[0].permute(1, 2, 0).detach()

        return img_left_optimized, img_right_optimized, loss_history

    def optimize_disparity_channel_activation_all(
        self,
        target: nn.Module,
        n_iters: int = 128,
        lr: float = 0.025,
    ):
        _, n_disp_channels, _, _ = self.get_activations_shape(target)

        img_left_layer = torch.empty(
            (n_disp_channels, self.h_bg, self.w_bg, 3), dtype=torch.float32
        )
        img_right_layer = torch.empty(
            (n_disp_channels, self.h_bg, self.w_bg, 3), dtype=torch.float32
        )
        loss_history_layer = torch.empty(
            (n_disp_channels, n_iters), dtype=torch.float32
        )
        for d in range(n_disp_channels):
            print(
                f"optimize disparity layer activation, disp_layer: "
                + f"{d}/{n_disp_channels}"
            )
            (
                img_left_layer[d],
                img_right_layer[d],
                loss_history_layer[d],
            ) = self.optimize_disparity_channel_activation(target, d, n_iters, lr)

        return img_left_layer, img_right_layer, loss_history_layer

    def optimize_layer_activation(
        self,
        target: nn.Module,
        n_iters: int,
        lr: float = 0.025,
    ):
        # define loss function
        loss_fn = LayerActivation(target)

        # define noise image for optimization
        input_param_left = optimviz.images.NaturalImage((self.h_bg, self.w_bg))
        input_param_right = optimviz.images.NaturalImage((self.h_bg, self.w_bg))

        # move to gpu
        self.input_param_left = input_param_left.to(self.device)
        self.input_param_right = input_param_right.to(self.device)

        # define optimizer
        # lr = 0.025
        # optimizer = torch.optim.Adam(obj.parameters(), lr=lr)
        params = list(self.input_param_left.parameters()) + list(
            self.input_param_right.parameters()
        )
        optimizer = torch.optim.Adam(params, lr=lr)

        # start optimizing
        loss_history = self.optimize(loss_fn, optimizer, n_iters)

        # fetch the optimized inputs and swap their axis to [h, w, rgb_channels]
        img_left_optimized = self.input_param_left()[0].permute(1, 2, 0).detach()
        img_right_optimized = self.input_param_right()[0].permute(1, 2, 0).detach()

        return img_left_optimized, img_right_optimized, loss_history

    def optimize_layer_activation_all(self, n_iters: int, lr: float = 0.025):
        target_list = [
            self.model.layer19[0],
            self.model.layer20[0],
            self.model.layer21[0],
            self.model.layer22[0],
            self.model.layer23[0],
            self.model.layer24[0],
            self.model.layer25[0],
            self.model.layer26[0],
            self.model.layer27[0],
            self.model.layer28[0],
            self.model.layer29[0],
            self.model.layer30[0],
            self.model.layer31[0],
            self.model.layer32[0],
            self.model.layer33a[0],
            self.model.layer34a[0],
            self.model.layer35a[0],
            self.model.layer36a[0],
            self.model.layer37,
        ]

        img_left_layer = torch.empty(
            (len(target_list), self.h_bg, self.w_bg, 3), dtype=torch.float32
        )
        img_right_layer = torch.empty(
            (len(target_list), self.h_bg, self.w_bg, 3), dtype=torch.float32
        )
        loss_history_layer = torch.empty(
            (len(target_list), n_iters), dtype=torch.float32
        )

        for t in range(len(target_list)):
            target = target_list[t]
            (
                img_left,
                img_right,
                loss_history,
            ) = self.optimize_layer_activation(target, n_iters)

            img_left_layer[t] = img_left
            img_right_layer[t] = img_right
            loss_history_layer[t] = loss_history

        return img_left_layer, img_right_layer, loss_history_layer

    def neuron_pref_input_corr(self, filter_mean, filter_size):
        # compute binocular correlation of preferred inputs for
        # center neuron in each feature and disparity channel
        filt = gauss2d(filter_mean, filter_size)

        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]
            print(f"compute binocular correlation {target_layer_name}")
            img_left_layer = np.load(
                f"{self.neuron_dir}/img_left_{target_layer_name}_conv.npy"
            )  # [feature_channels, disp_channels, h, w, rgb]
            img_right_layer = np.load(
                f"{self.neuron_dir}/img_right_{target_layer_name}_conv.npy"
            )

            # convert to grayscale
            img_left_layer = img_left_layer.mean(axis=-1)
            img_right_layer = img_right_layer.mean(axis=-1)

            corr_neuron = compute_corr_neuron_pref_input(
                img_left_layer, img_right_layer, filt
            )

            # save
            np.save(
                f"{self.neuron_dir}/neuron_corr_{target_layer_name}_conv.npy",
                corr_neuron,
            )

    def channel_correlation(self, filter_mean, filter_size):
        # compute binocular correlation of preferred inputs for
        # each feature and disparity channel
        filt = gauss2d(filter_mean, filter_size)
        # filt = 1.0 * np.ones((filter_size, filter_size), dtype=np.float32)

        for t in range(len(self.target_name)):
            target_layer_name = self.target_name[t]
            print(f"compute binocular correlation {target_layer_name}")
            img_left_layer = np.load(
                f"{self.channel_dir}/img_left_channel_{target_layer_name}.npy"
            )
            img_right_layer = np.load(
                f"{self.channel_dir}/img_right_channel_{target_layer_name}.npy"
            )

            # convert to grayscale
            img_left_layer = img_left_layer.mean(axis=-1)
            img_right_layer = img_right_layer.mean(axis=-1)

            corr_channel = compute_channel_correlation_pearson(
                img_left_layer, img_right_layer, filt
            )

            # save
            np.save(
                f"{self.channel_dir}/channel_corr_{target_layer_name}.npy",
                corr_channel,
            )

    def layer_correlation(self, filter_mean, filter_size):
        # %% compute binocular correlation of preferred inputs for each layer
        filt = gauss2d(filter_mean, filter_size)

        img_left_layers = np.load(f"{self.layer_dir}/img_left_deep_dream.npy")
        img_right_layers = np.load(f"{self.layer_dir}/img_right_deep_dream.npy")

        # convert to grayscale
        img_left_layer = img_left_layers.mean(axis=-1)
        img_right_layer = img_right_layers.mean(axis=-1)

        corr_layer = compute_layer_correlation(img_left_layer, img_right_layer, filt)

        return corr_layer
        # save
        # np.save(f"{self.layer_dir}/layer_corr.npy", corr_layer)
