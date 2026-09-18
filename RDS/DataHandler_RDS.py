# %% load necessary modules
import numpy as np
from jaxtyping import Float, Int

import torch
from torch.utils.data import Dataset

from RDS.RDS import RDS

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
        rds_left: Float[np.ndarray, "n_rds h w n_rgb_channels"],
        rds_right: Float[np.ndarray, "n_rds h w n_rgb_channels"],
        rds_label: Int[np.ndarray, "n_rds"],
        transform=None,
    ):
        self.rds_left = rds_left  # [n_rds, h, w, n_rgb_channels]
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
        disp_ct_pix_list: list,
        n_rds_each_disp: int,
        dotDens_list: list,
        background_flag: bool,
        overlap_flag: bool = True,
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

        rds = RDS(n_rds_each_disp, W_BG, H_BG, W_CT, H_CT, dotDens, R_DOT, overlap_flag)
        if background_flag:  # generate RDSs with cRDS background
            rds_batch_left, rds_batch_right = rds.create_rds_batch(
                disp_ct_pix_list, dotMatch_ct
            )  # [n_rds_each_disp, len(disp_ct_pix_list), h, w]
            bg_message = "with cRDS background"
        else:  # without cRDS background
            rds_batch_left, rds_batch_right = rds.create_rds_without_bg_batch(
                disp_ct_pix_list, dotMatch_ct
            )  # [n_rds_each_disp, len(disp_ct_pix), h, w]
            bg_message = "without cRDS background"

        for d in range(len(disp_ct_pix_list)):
            print(
                f"generating rds: {bg_message}, "
                + f"dot match: {dotMatch_ct:.2f}, "
                + f"disparity: {disp_ct_pix_list[d]}"
            )

        # [n_rds_each_disp, len(disp_ct_pix), h, w] =>
        # [len(disp_ct_pix), n_rds_each_disp, h, w] =>
        # [len(disp_ct_pix) * n_rds_each_disp, h, w] = [n_rds, h, w]
        n_rds = n_rds_each_disp * len(disp_ct_pix_list)
        left_planes = np.transpose(rds_batch_left, (1, 0, 2, 3)).reshape(
            n_rds, rds.h_bg, rds.w_bg
        )
        right_planes = np.transpose(rds_batch_right, (1, 0, 2, 3)).reshape(
            n_rds, rds.h_bg, rds.w_bg
        )

        if pedestal_flag:
            # shift the whole rds to set near disp at 0 disp
            left_planes = np.roll(left_planes, disp_ct_pix_list[0], axis=-1)

        # [n_rds, h, w] => [n_rds, h, w, n_rgb_channels]
        n_channels = 3  # rgb channels
        rds_left = np.repeat(left_planes[..., None], n_channels, axis=-1).astype(
            np.float32
        )
        rds_right = np.repeat(right_planes[..., None], n_channels, axis=-1).astype(
            np.float32
        )
        rds_disp = np.repeat(
            np.asarray(disp_ct_pix_list, dtype=np.int8), n_rds_each_disp
        )

        return rds_left, rds_right, rds_disp

    @staticmethod
    def generate_rds_v2(
        dotMatch_ct,
        dotDens,
        disp_ct_pix_list,
        n_rds_each_disp,
        background_flag,
        pedestal_flag,
    ):
        """
        V2 swaps the left and right images such that positive disparity = near (pixels
        on the left images are shifted to the left to get their
        corresponding pixels on right images),
        negative disparity = far (pixels on the left images are shifted to the right
        to get their corresponding pixels on the right images)
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
        # rds_batch_right : [n_rds_each_disp, len(disp_ct_pix), h, w]

        for d in range(len(disp_ct_pix_list)):
            print(
                f"generating rds: {bg_message}, "
                + f"dot match: {dotMatch_ct:.2f}, "
                + f"disparity: {disp_ct_pix_list[d]}"
            )

        # [n_rds_each_disp, len(disp_ct_pix), h, w] =>
        # [len(disp_ct_pix), n_rds_each_disp, h, w] =>
        # [len(disp_ct_pix) * n_rds_each_disp, h, w] = [n_rds, h, w]
        n_rds = n_rds_each_disp * len(disp_ct_pix_list)
        left_planes = np.transpose(rds_batch_right, (1, 0, 2, 3)).reshape(
            n_rds, rds.h_bg, rds.w_bg
        )
        right_planes = np.transpose(rds_batch_left, (1, 0, 2, 3)).reshape(
            n_rds, rds.h_bg, rds.w_bg
        )

        if pedestal_flag:
            # shift the whole rds to set near disp at 0 disp
            left_planes = np.roll(left_planes, disp_ct_pix_list[1], axis=-1)

        # [n_rds, h, w] => [n_rds, h, w, n_rgb_channels]
        n_channels = 3  # rgb channels
        rds_left = np.repeat(left_planes[..., None], n_channels, axis=-1).astype(
            np.float32
        )
        rds_right = np.repeat(right_planes[..., None], n_channels, axis=-1).astype(
            np.float32
        )
        rds_disp = np.repeat(
            np.asarray(disp_ct_pix_list, dtype=np.int8), n_rds_each_disp
        )

        return rds_left, rds_right, rds_disp
