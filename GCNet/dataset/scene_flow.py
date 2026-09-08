#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

# %%
import os
import random

import numpy as np
import torch.utils.data as data
from PIL import Image
from albumentations import Compose, OneOf
from natsort import natsorted

from dataset.preprocess import augment
from dataset.stereo_albumentation import (
    RandomShiftRotate,
    GaussNoiseStereo,
    RGBShiftStereo,
    RandomBrightnessContrastStereo,
    random_crop,
    horizontal_flip,
    ToTensor,
    Normalize,
)
from utilities.python_pfm import readPFM


# %%
class SceneFlowSamplePackDataset(data.Dataset):
    def __init__(self, datadir, split="train"):
        super(SceneFlowSamplePackDataset, self).__init__()

        self.datadir = datadir
        self.left_fold = "RGB_cleanpass/left/"
        self.right_fold = "RGB_cleanpass/right/"
        self.disp = "disparity/left"
        self.disp_right = "disparity/right"
        self.occ_fold = "occlusion/left"
        self.occ_fold_right = "occlusion/right"

        self.data = os.listdir(os.path.join(self.datadir, self.left_fold))
        # left_fold = "frames_cleanpass/left/"
        # data = os.listdir(os.path.join(datadir, left_fold))

        self._augmentation()

    def _augmentation(self):
        self.transformation = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = {}

        path = self.datadir

        left = np.array(
            Image.open(os.path.join(path, self.left_fold, self.data[idx]))
        ).astype(np.uint8)[..., :3]
        input_data["left"] = left

        right = np.array(
            Image.open(os.path.join(path, self.right_fold, self.data[idx]))
        ).astype(np.uint8)[..., :3]
        input_data["right"] = right

        occ = np.array(
            Image.open(os.path.join(path, self.occ_fold, self.data[idx]))
        ).astype(np.bool)
        input_data["occ_mask"] = occ

        occ_right = np.array(
            Image.open(os.path.join(path, self.occ_fold_right, self.data[idx]))
        ).astype(np.bool)
        input_data["occ_mask_right"] = occ_right

        disp, _ = readPFM(
            os.path.join(path, self.disp, self.data[idx].replace("png", "pfm"))
        )
        input_data["disp"] = disp

        disp_right, _ = readPFM(
            os.path.join(path, self.disp_right, self.data[idx].replace("png", "pfm"))
        )
        input_data["disp_right"] = disp_right

        input_data = augment(input_data, self.transformation)

        return input_data


class SceneFlowFlyingThingsDataset(data.Dataset):
    def __init__(self, datadir, config, split="train"):
        super(SceneFlowFlyingThingsDataset, self).__init__()

        self.datadir = datadir
        self.split = split
        if self.split == "train":
            self.split_folder = "TRAIN"
        else:
            self.split_folder = "TEST"

        ####
        self.h_crop = config.img_height
        self.w_crop = config.img_width
        self.c_disp_shift = config.c_disp_shift
        self.augmentation_seed = config.seed
        ####

        self._read_data()
        self._augmentation()

    def _read_data(self):
        directory = os.path.join(self.datadir, "frames_cleanpass", self.split_folder)
        # directory = os.path.join(datadir, "frames_cleanpass", split_folder)
        sub_folders = [
            os.path.join(directory, subset)
            for subset in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, subset))
        ]

        seq_folders = []
        for sub_folder in sub_folders:
            seq_folders += [
                os.path.join(sub_folder, seq)
                for seq in os.listdir(sub_folder)
                if os.path.isdir(os.path.join(sub_folder, seq))
            ]

        self.left_data = []
        for seq_folder in seq_folders:
            self.left_data += [
                os.path.join(seq_folder, "left", img)
                for img in os.listdir(os.path.join(seq_folder, "left"))
            ]
        # left_data = []
        # for seq_folder in seq_folders:
        #     left_data += [
        #         os.path.join(seq_folder, "left", img)
        #         for img in os.listdir(os.path.join(seq_folder, "left"))
        #     ]

        self.left_data = natsorted(self.left_data)
        # left_data = natsorted(left_data)

        # directory = os.path.join(self.datadir, "occlusion", self.split_folder, "left")
        # self.occ_data = [os.path.join(directory, occ) for occ in os.listdir(directory)]
        # self.occ_data = natsorted(self.occ_data)
        # directory = os.path.join(datadir, "occlusion", split_folder, "left")
        # occ_data = [os.path.join(directory, occ) for occ in os.listdir(directory)]
        # occ_data = natsorted(occ_data)

    def _augmentation(self):
        if self.split == "train":
            self.transformation = Compose(
                [
                    RandomShiftRotate(p=1.0),
                    RGBShiftStereo(p=1.0, p_asym=0.3),
                    OneOf(
                        [
                            GaussNoiseStereo(p=1.0, p_asym=1.0),
                            RandomBrightnessContrastStereo(
                                p=1.0, p_asym=0.5
                            ),
                        ],
                        p=1.0,
                    ),
                ],
                seed=self.augmentation_seed,
            )
        else:
            self.transformation = None

        # self.transformation = None

    def __len__(self):
        return len(self.left_data)

    def __getitem__(self, idx):
        return _load_stereo_sample(self, idx)


class SceneFlowMonkaaDataset(data.Dataset):
    def __init__(self, datadir, config, split="train"):
        super(SceneFlowMonkaaDataset, self).__init__()

        self.datadir = datadir
        self.split = split
        ####
        self.h_crop = config.img_height
        self.w_crop = config.img_width
        self.c_disp_shift = config.c_disp_shift
        self.augmentation_seed = config.seed
        ####

        self._read_data()
        self._augmentation()

    def _read_data(self):
        directory = os.path.join(self.datadir, "frames_cleanpass")
        sub_folders = [
            os.path.join(directory, subset)
            for subset in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, subset))
        ]

        self.left_data = []
        for sub_folder in sub_folders:
            self.left_data += [
                os.path.join(sub_folder, "left", img)
                for img in os.listdir(os.path.join(sub_folder, "left"))
            ]

        self.left_data = natsorted(self.left_data)

    def _split_data(self):
        return

    def _augmentation(self):
        ###
        if self.split == "train":
            self.transformation = Compose(
                [
                    RandomShiftRotate(p=1.0),
                    RGBShiftStereo(p=1.0, p_asym=0.3),
                    OneOf(
                        [
                            GaussNoiseStereo(p=1.0, p_asym=1.0),
                            RandomBrightnessContrastStereo(
                                p=1.0, p_asym=0.5
                            ),
                        ],
                        p=1.0,
                    ),
                ],
                seed=self.augmentation_seed,
            )
        else:
            self.transformation = None
        ###
        # self.transformation = None

    def __len__(self):
        return len(self.left_data)

    def __getitem__(self, idx):
        return _load_stereo_sample(self, idx)


def _load_stereo_sample(dataset, idx):
    left_path = dataset.left_data[idx]
    right_path = left_path.replace("left", "right")
    with Image.open(left_path) as image:
        left = np.asarray(image.convert("RGB"))
    with Image.open(right_path) as image:
        right = np.asarray(image.convert("RGB"))
    reference = 1 if random.random() <= 0.5 else -1
    path = left_path if reference == 1 else right_path
    path = path.replace("frames_cleanpass", "disparity").replace(".png", ".pfm")
    disparity, _ = readPFM(path)
    result = {"left": left, "right": right,
              "disp" if reference == 1 else "disp_right": disparity}
    result = random_crop(dataset.h_crop, dataset.w_crop, result,
                         dataset.c_disp_shift, dataset.split, reference=reference)
    result.pop("disp_right", None)
    return augment(result, dataset.transformation)
