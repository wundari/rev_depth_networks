#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import random

import numpy as np
import torch
from albumentations import Compose

from dataset.stereo_albumentation import Normalize, ToTensor

__imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]

normalization = Compose(
    [Normalize(p=1.0), ToTensor(p=1.0)], p=1.0
)


def seed_stereo_worker(worker_id):
    """Seed global crop randomness and Albumentations' per-worker generators."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker = torch.utils.data.get_worker_info()
    if worker is not None:
        dataset = worker.dataset
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        transformation = getattr(dataset, "transformation", None)
        if transformation is not None:
            transformation.set_random_seed(worker_seed)

###
from torchvision import transforms

DATA_MEANS = np.array([0.5, 0.5, 0.5])
DATA_STD = np.array([0.5, 0.5, 0.5])

transform_data = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)]
)

###


def denormalize(img):
    """
    De-normalize a tensor and return img

    :param img: normalized image, [C,H,W]
    :return: original image, [H,W,C]
    """

    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0)  # H,W,C
        img *= torch.tensor(__imagenet_stats["std"])
        img += torch.tensor(__imagenet_stats["mean"])
        return img.numpy()
    else:
        img = img.transpose(1, 2, 0)  # H,W,C
        img *= np.array(__imagenet_stats["std"])
        img += np.array(__imagenet_stats["mean"])
        return img


def compute_left_occ_region(w, disp):
    """
    Compute occluded region on the left image border

    :param w: image width
    :param disp: left disparity
    :return: occ mask
    """

    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    shifted_coord = coord - disp
    occ_mask = shifted_coord < 0  # occlusion mask, 1 indicates occ

    return occ_mask


def compute_right_occ_region(w, disp):
    """
    Compute occluded region on the right image border

    :param w: image width
    :param disp: right disparity
    :return: occ mask
    """
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    shifted_coord = coord + disp
    occ_mask = shifted_coord > w  # occlusion mask, 1 indicates occ

    return occ_mask


def augment(input_data, transformation):
    """
    apply augmentation and find occluded pixels
    """

    if transformation is not None:
        #     # perform augmentation first
        input_data = transformation(**input_data)

    # w = input_data["disp"].shape[-1]
    # set large/small values to be 0
    # input_data["disp"][input_data["disp"] > w] = 0
    # input_data['disp'][input_data['disp'] < 0] = 0

    ###
    # input_data["disp"][input_data["disp"] > 127] = 127
    # input_data["disp"][input_data["disp"] < -128] = -128
    ###

    # # manually compute occ area (this is necessary after cropping)
    # occ_mask = compute_left_occ_region(w, input_data["disp"])
    # input_data["occ_mask"][occ_mask] = True  # update
    # input_data["occ_mask"] = np.ascontiguousarray(input_data["occ_mask"])

    # # manually compute occ area for right image
    # try:
    #     occ_mask = compute_right_occ_region(w, input_data["disp_right"])
    #     input_data["occ_mask_right"][occ_mask] = 1
    #     input_data["occ_mask_right"] = np.ascontiguousarray(
    #         input_data["occ_mask_right"]
    #     )
    # except KeyError:
    #     # print('No disp mask right, check if dataset is KITTI')
    #     input_data["occ_mask_right"] = np.zeros_like(occ_mask).astype(np.bool)
    # input_data.pop("disp_right", None)  # remove disp right after finish

    # set occlusion area to 0
    # occ_mask = input_data["occ_mask"]
    # input_data["disp"][occ_mask] = 0
    # input_data["disp"] = np.ascontiguousarray(input_data["disp"], dtype=np.float32)

    # return normalized image
    result = normalization(**input_data)
    # Compose 1.3.1 made ALL numpy outputs contiguous; 2.x only handles
    # registered targets. PFM disparity arrays may have negative strides.
    return {
        key: np.ascontiguousarray(value) if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }

    ###
    # input_data["left"] = transform_data(input_data["left"])
    # input_data["right"] = transform_data(input_data["right"])
    # input_data["disp"] = torch.tensor(input_data["disp"], dtype=torch.float32)

    # return input_data
    ###
