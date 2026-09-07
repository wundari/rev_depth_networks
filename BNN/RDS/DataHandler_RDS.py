# %% load necessary modules
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from RDS.RDS_v3 import RDS

# reproducibility
import random
from config.config import BNNconfig

cfg = BNNconfig()
seed_number = cfg.seed  # 12321
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

# %% define global variables
W_BG = 512  # input width
H_BG = 256  # input height
W_CT = W_BG // 2  # width center RDS
H_CT = H_BG // 2  # height center RDS
R_DOT = 5  # rds dot radius in pixel


# %% define dataset class for dataloader
class DatasetRDS(Dataset):
    def __init__(
        self,
        rds_left,
        rds_right,
        rds_label,
        transform=None,
    ):
        self.rds_left = rds_left
        self.rds_right = rds_right
        self.rds_label = rds_label
        self.transform = transform

    def __len__(self):
        return len(self.rds_left)

    def __getitem__(self, idx):
        # load image
        img_left = self.rds_left[idx]
        img_right = self.rds_right[idx]
        img_disp = self.rds_label[idx]

        # transform image
        # transform_data = transforms.Compose(
        #         [transforms.ToTensor(), transforms.Lambda(lambda t: (t + 1.0) / 2.0)]
        #     )
        if self.transform is not None:
            # convert to tensor and in range [0, 1]
            img_left = self.transform(img_left)
            img_right = self.transform(img_right)

        return img_left, img_right, img_disp


# %%
class RDS_Handler:
    def __init__(
        self,
        disp_ct_pix_list,
        n_rds_each_disp,
        dotDens_list,
        background_flag,
        overlap_flag=1,
    ):
        """_summary_

        Args:
            disp_ct_pix_list (list or np.array, int): a list of disparity magnitudes
                crossed disparity (near): disp > 0 (+)
                uncrossed disparity (far): disp < 0 (-)

                for ex: disp_mag = 10
                        disp_ct_pix = np.arange(-disp_mag, disp_mag + 2, 2)

            n_rds_each_disp (int): the number of rds images for each disparity
                magnitude in disp_ct_pix

            dotDens_list (list or np.array): a list of dot densities
                for ex: dotDens_list = 0.1 * np.arange(1, 10)

            background_flag (int): a flag indicating rds using cRDS background
                0: without cRDS background
                1: with cRDS background

            overlap_flag (int, optional): a flag indicating rds dots can
                overlap each other or not.
                0: dots dot not overlap
                1: dots overlap
                Defaults to 1.
        """
        # rds params
        self.w_bg = W_BG  # rds width
        self.h_bg = H_BG  # rds height
        self.w_ct = W_CT  # width center RDS
        self.h_ct = H_CT  # height center RDS
        self.rds_type = ["ards", "hmrds", "crds"]
        # dot match for ards: 0, hmrds: 0.5, crds: 1, urds: -1
        self.dotMatch_list = [0.0, 0.5, 1.0]

        self.n_rds_each_disp = n_rds_each_disp
        self.disp_ct_pix_list = disp_ct_pix_list  # a list of disparity magnitudes
        self.dotDens_list = dotDens_list
        self.rDot = R_DOT  # rds dot radius in pixel. Defaults to R_DOT = 5.
        self.background_flag = background_flag
        self.overlap_flag = overlap_flag

    @staticmethod
    def generate_rds(
        dotMatch_ct,
        dotDens,
        disp_ct_pix_list,
        n_rds_each_disp,
        background_flag,
        pedestal_flag,
    ):
        """
        generate RDSs with the following parameters

        Args:
            dotMatch_ct (float): a dot match value for the center RDS

            dotDens (float): a dot density value for the RDS.
                min: 0, max: 1.0

            n_rds_each_disp (int): the number of rds images for each disparity
                magnitude in disp_ct_pix

            disp_ct_pix_list (list or np.array, int): a list of disparity magnitudes
                crossed disparity (near): disp > 0 (+)
                uncrossed disparity (far): disp < 0 (-)

                for ex: disp_mag = 10
                        disp_ct_pix = np.arange(-disp_mag, disp_mag + 2, 2)

            background_flag (binary): a flag indicating with or without cRDS background
                0: without cRDS background
                1: with cRDS background

            pedestal_flag (binary): a flag indicating with or without pedestal.
                pedestal here means that the whole RDSs are shifted such that
                the smallest disparity = 0.
                0: without pedestal
                1: with pedestal

        Returns:
            rds_left = np.zeros((n_rds, rds.h_bg, rds.w_bg, n_channels), dtype=np.float32)
            rds_right = np.zeros((n_rds, rds.h_bg, rds.w_bg, n_channels), dtype=np.float32)
            rds_disp = np.zeros(n_rds, dtype=np.int8)
        """
        overlap_flag = 1  # 0: dots are not allowed to overlap; 1: otherwise

        n_rds = n_rds_each_disp * len(disp_ct_pix_list)

        rds = RDS(n_rds_each_disp, W_BG, H_BG, W_CT, H_CT, dotDens, R_DOT, overlap_flag)
        if background_flag:  # generate RDSs with cRDS background
            rds_batch_left, rds_batch_right = rds.create_rds_batch(
                disp_ct_pix_list, dotMatch_ct
            )
            bg_message = "with cRDS background"
        else:  # without cRDS background
            rds_batch_left, rds_batch_right = rds.create_rds_without_bg_batch(
                disp_ct_pix_list, dotMatch_ct
            )
            bg_message = "without cRDS background"
        # rds_batch_right : [batch_size, len(disp_ct_pix), h, w]

        # remapping rds into [len(disp_ct_pix) * batch_size, h, w, n_channels]
        n_channels = 3  # rgb channels
        rds_left = np.zeros((n_rds, rds.h_bg, rds.w_bg, n_channels), dtype=np.float32)
        rds_right = np.zeros((n_rds, rds.h_bg, rds.w_bg, n_channels), dtype=np.float32)
        rds_disp = np.zeros(n_rds, dtype=np.int8)
        count = 0
        for d in range(len(disp_ct_pix_list)):
            print(
                f"generating rds: {bg_message}, "
                + f"dot match: {dotMatch_ct:.2f}, "
                + f"disparity: {disp_ct_pix_list[d]}"
            )
            # if disp_ct_pix_list[d] < 0:
            #     depth_label = -1  # uncrossed-sisparity (far)
            # elif disp_ct_pix_list[d] == 0:
            #     depth_label = 0  # disparity 0
            # else:
            #     depth_label = 1  # crossed-disparity (near)

            for t in range(n_rds_each_disp):
                temp = rds_batch_left[t, d]

                # using pedestal
                if pedestal_flag:
                    # shift the whole rds to set near disp at 0 disp
                    temp = np.roll(temp, disp_ct_pix_list[0], axis=1)
                rds_left[count, :, :, 0] = temp
                rds_left[count, :, :, 1] = temp
                rds_left[count, :, :, 2] = temp

                temp = rds_batch_right[t, d]
                # temp = np.roll(temp, disp_ct_pix_list[1], axis=1)
                rds_right[count, :, :, 0] = temp
                rds_right[count, :, :, 1] = temp
                rds_right[count, :, :, 2] = temp

                # rds_label[count] = depth_label
                rds_disp[count] = disp_ct_pix_list[d]

                count += 1

        return rds_left, rds_right, rds_disp

    @staticmethod
    def generate_rds_v2(
        dotMatch_ct, dotDens, disp_ct_pix_list, n_rds_each_disp, background_flag
    ):
        """
        generate RDSs with the following parameters

        Args:
            dotMatch_ct (float): a dot match value for the center RDS

            dotDens (float): a dot density value for the RDS.
                min: 0, max: 1.0

            n_rds_each_disp (int): the number of rds images for each disparity
                magnitude in disp_ct_pix

            disp_ct_pix_list (list or np.array, int): a list of disparity magnitudes
                crossed disparity (near): disp > 0 (+)
                uncrossed disparity (far): disp < 0 (-)

                for ex: disp_mag = 10
                        disp_ct_pix = np.arange(-disp_mag, disp_mag + 2, 2)

            background_flag (binary): a flag indicating with or without cRDS background
                0: without cRDS background
                1: with cRDS background

        Returns:
            rds_left = np.zeros((n_rds, rds.h_bg, rds.w_bg, n_channels), dtype=np.float32)
            rds_right = np.zeros((n_rds, rds.h_bg, rds.w_bg, n_channels), dtype=np.float32)
            rds_disp = np.zeros((n_rds, 2), dtype=np.int8)  # colnames: [rds_type, disp]
        """
        overlap_flag = 1  # 0: dots are not allowed to overlap; 1: otherwise

        n_rds = n_rds_each_disp * len(disp_ct_pix_list)

        rds = RDS(n_rds_each_disp, W_BG, H_BG, W_CT, H_CT, dotDens, R_DOT, overlap_flag)
        if background_flag:  # generate RDSs with cRDS background
            rds_batch_left, rds_batch_right = rds.create_rds_batch(
                disp_ct_pix_list, dotMatch_ct
            )
            bg_message = "with cRDS background"
        else:  # without cRDS background
            rds_batch_left, rds_batch_right = rds.create_rds_without_bg_batch(
                disp_ct_pix_list, dotMatch_ct
            )
            bg_message = "without cRDS background"
        # rds_batch_right : [batch_size, len(disp_ct_pix), h, w]

        # remapping rds into [len(disp_ct_pix) * batch_size, h, w, n_channels]
        n_channels = 3  # rgb channels
        rds_left = np.zeros((n_rds, rds.h_bg, rds.w_bg, n_channels), dtype=np.float32)
        rds_right = np.zeros((n_rds, rds.h_bg, rds.w_bg, n_channels), dtype=np.float32)
        rds_disp = np.zeros((n_rds, 2), dtype=np.int8)  # colnames: [rds_type, disp]
        # rds_type: 0 = ards; 1= hmrds; 2 = crds
        if dotMatch_ct == 0:
            rds_type = 0  # ards
        elif dotMatch_ct == 0.5:
            rds_type = 1  # hmrds
        elif dotMatch_ct == 1:
            rds_type = 2  # crds
        count = 0
        for d in range(len(disp_ct_pix_list)):
            print(
                f"generating rds: {bg_message}, "
                + f"dot match: {dotMatch_ct}, "
                + f"disparity: {disp_ct_pix_list[d]}"
            )
            # if disp_ct_pix_list[d] < 0:
            #     depth_label = -1  # uncrossed-sisparity (far)
            # elif disp_ct_pix_list[d] == 0:
            #     depth_label = 0  # disparity 0
            # else:
            #     depth_label = 1  # crossed-disparity (near)

            for t in range(n_rds_each_disp):
                temp = rds_batch_left[t, d]
                # shift the whole rds to set near disp at 0 disp
                # temp = np.roll(temp, disp_ct_pix_list[0], axis=1)
                rds_left[count, :, :, 0] = temp
                rds_left[count, :, :, 1] = temp
                rds_left[count, :, :, 2] = temp

                temp = rds_batch_right[t, d]
                # temp = np.roll(temp, disp_ct_pix_list[1], axis=1)
                rds_right[count, :, :, 0] = temp
                rds_right[count, :, :, 1] = temp
                rds_right[count, :, :, 2] = temp

                rds_disp[count, 0] = rds_type
                rds_disp[count, 1] = disp_ct_pix_list[d]

                count += 1

        return rds_left, rds_right, rds_disp

    def _rds_dataloader(self, rds_left, rds_right, rds_disp, batch_size):
        # transform rds to tensor and in range [0, 1]
        transform_data = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda t: (t + 1.0) / 2.0)]
        )
        rds_data = DatasetRDS(rds_left, rds_right, rds_disp, transform=transform_data)
        rds_loader = DataLoader(
            rds_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=g,
        )

        return rds_loader

    def generate_rds_dataloader(self, dotDens, batch_size):
        # generate ards
        dotMatch_ct = 0
        # rds_left, rds_right: [len(disp_ct_pix_list) * batch_size, h, w, n_channels]
        # rds_disp : [len(disp_ct_pix) * batch_size]
        ards_left, ards_right, ards_disp = self.generate_rds_v2(
            dotMatch_ct,
            dotDens,
            self.disp_ct_pix_list,
            self.n_rds_each_disp,
            self.background_flag,
        )

        # generate hmrds
        dotMatch_ct = 0.5
        # rds_left, rds_right: [len(disp_ct_pix_list) * batch_size, h, w, n_channels]
        # rds_disp : [len(disp_ct_pix) * batch_size]
        hmrds_left, hmrds_right, hmrds_disp = self.generate_rds_v2(
            dotMatch_ct,
            dotDens,
            self.disp_ct_pix_list,
            self.n_rds_each_disp,
            self.background_flag,
        )

        # generate crds
        dotMatch_ct = 1
        # rds_left, rds_right: [len(disp_ct_pix_list) * batch_size, h, w, n_channels]
        # rds_disp : [len(disp_ct_pix) * batch_size]
        crds_left, crds_right, crds_disp = self.generate_rds_v2(
            dotMatch_ct,
            dotDens,
            self.disp_ct_pix_list,
            self.n_rds_each_disp,
            self.background_flag,
        )

        # concatenate all rds
        rds_left = np.concatenate([ards_left, hmrds_left, crds_left])
        rds_right = np.concatenate([ards_right, hmrds_right, crds_right])
        rds_disp = np.concatenate([ards_disp, hmrds_disp, crds_disp])

        # create dataloader for ards
        rds_loader = self._rds_dataloader(rds_left, rds_right, rds_disp, batch_size)

        return rds_loader

    @staticmethod
    def create_ground_truth(disp_ct_pix, n_rds, background_flag):
        if background_flag:  # rds with crds background
            rds_bg = np.zeros((n_rds, H_BG, W_BG), dtype=np.int32)

            # calculate center position in pixel
            center = (H_BG // 2, W_BG // 2)

            # calculate the starting and the ending coordinate of the rds center
            row_ct_start = center[0] - H_CT // 2
            row_ct_end = row_ct_start + H_CT + 1
            col_ct_start = center[1] - W_CT // 2
            col_ct_end = col_ct_start + W_CT + 2

            rds_bg[:, row_ct_start:row_ct_end, col_ct_start:col_ct_end] = disp_ct_pix

        else:
            rds_bg = disp_ct_pix * np.ones((n_rds, H_BG, W_BG), dtype=np.int32)

        return rds_bg

    # def set_sceneflow_val_loader(self):
    #     train_list = [self._dataset_to_process]
    #     val_list = [self._dataset_to_process]
    #     # train_list = ["driving", "flying", "monkaa"]
    #     # val_list = ["driving", "flying", "monkaa"]
    #     (
    #         _,
    #         _,
    #         file_train_disp,
    #         file_val_left,
    #         file_val_right,
    #         file_val_disp,
    #     ) = generate_image_paths(train_list, val_list, flip_input=1)

    #     ## data normalization notes: https://cs231n.github.io/neural-networks-2/
    #     # a = np.zeros((len(imgs_left), 540, 960, 3), dtype=np.float32)
    #     # for i in range(len(imgs_left)):
    #     #     a[i] = imgs_left[i]
    #     # DATA_MEANS = [a[:, :, :, 0].mean(), a[:, :, :, 1].mean(), a[:, :, :, 2].mean()]
    #     # DATA_STD = [a[:, :, :, 0].std(), a[:, :, :, 1].std(), a[:, :, :, 2].std()]
    #     # DATA_MEANS = np.array([0.32, 0.32, 0.28])
    #     # DATA_STD = np.array([0.28, 0.27, 0.25])
    #     DATA_MEANS = np.array([0.5, 0.5, 0.5])
    #     DATA_STD = np.array([0.5, 0.5, 0.5])

    #     transform_data = transforms.Compose(
    #         [transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)]
    #     )

    #     # get validation dataset
    #     n_train = int(len(file_train_disp) * 0.8)
    #     train_id = np.random.choice(
    #         np.arange(len(file_train_disp)), n_train, replace=False
    #     )
    #     val_id = np.setdiff1d(np.arange(len(file_train_disp)), train_id)
    #     patch_data = DatasetVal(
    #         [file_val_left[i] for i in val_id],
    #         [file_val_right[i] for i in val_id],
    #         [file_val_disp[i] for i in val_id],
    #         self.h_bg,
    #         self.w_bg,
    #         self.c_disp_shift,
    #         transform=transform_data,
    #         flip_input=1,
    #     )
    #     val_loader = DataLoader(
    #         patch_data,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         pin_memory=True,
    #         drop_last=True,
    #         num_workers=1,
    #         worker_init_fn=seed_worker,
    #         generator=g,
    #     )

    #     return val_loader
