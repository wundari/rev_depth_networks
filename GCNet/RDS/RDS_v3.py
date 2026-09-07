"""
File: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/RDS/RDS_v2.py
Project: /home/wundari/NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/RDS
Created Date: 2023-03-29 16:22:00
Author: Bayu G. Wundari
-----
Last Modified: 2023-03-29 16:22:20
Modified By: Bayu G. Wundari
-----
Copyright (c) 2023 National Institute of Information and Communications Technology (NICT)

-----
HISTORY:
Date    	By	Comments
----------	---	----------------------------------------------------------
dots can be not overlap
"""

import numpy as np
from joblib import Parallel, delayed

from skimage.draw import disk

from timeit import default_timer as timer
from datetime import datetime


class RDS:
    def __init__(self, n_rds, w_bg, h_bg, w_ct, h_ct, dotDens, rDot, overlap_flag=1):
        self.n_rds = n_rds  # number of rds
        self.w_bg = w_bg  # rds_bg width in pixel
        self.h_bg = h_bg  # rds_bg height in pixel
        self.w_ct = w_ct  # rds_center width in pixel
        self.h_ct = h_ct  # rds_center height in pixel
        self.dotDens = dotDens  # dot density
        self.rDot = rDot  # dot radius in pixel
        self.overlap_flag = overlap_flag  # 0 dots aren't overlap; 1 otherwise

    def generate_dot_position(self, overlap_flag):
        self.overlap_flag = overlap_flag

        # rDot = 5
        # w_bg = 512
        # h_bg = 256
        # dotDens = 0.25

        # calculate nDots:
        nDots = np.int32(
            self.dotDens * self.w_bg * self.h_bg / (np.pi * self.rDot**2)
        )
        # nDots = np.int32(dotDens * w_bg*h_bg/(np.pi*rDot**2))

        # random dot positions
        if self.overlap_flag == 1:  # if dots are overlap
            pos_x = np.arange(self.rDot, self.w_bg, self.rDot)
            pos_y = np.arange(self.rDot, self.h_bg, self.rDot)

        else:  # if dots aren't overlap
            pos_x = np.arange(self.rDot, self.w_bg, 2 * self.rDot)
            pos_y = np.arange(self.rDot, self.h_bg, 2 * self.rDot)

        xc = np.random.choice(pos_x, size=nDots)
        yc = np.random.choice(pos_y, size=nDots)

        return xc, yc

    def _set_dotMatch(self, rds_ct, dotMatch_ct, nDots, rDot_pix):
        """
        set dot match level betweem rds left and right

        Inputs:
            - rds_ct: <2D np.array> rds center matrix
            - rds_bg: <2D np.array> rds background matrix
            - dotMatch_ct: <scalar>, dot match level, between 0 and 1.
                            -1 mean uncorrelated RDS
                            0 means anticorrelated RDS
                            0.5 means half-matched RDS
                            1 means correlated RDS

        Outputs:
            rds_ct_left: <2D np.array>, rds for left
            rds_ct_right: <2D np.array>, rds for right
        """

        # find id_dot in rds_ct, excluding gray background
        dotID_ct = np.unique(rds_ct)[np.unique(rds_ct) > 0.5]

        if dotMatch_ct == -1:  # make urds
            nx, ny = np.shape(rds_ct)
            rDot_pix = self._compute_deg2pix(self.rDot)  # dot radius in pixel
            # nDots_ct = np.int32(self.dotDens*np.prod(self.size_rds_bg)/(np.pi*rDot_pix**2))
            nDots_ct = np.int32((nx * ny) / np.prod(self.size_rds_bg) * nDots)

            ## make rds left
            rds_ct_left = np.zeros((nx, ny), dtype=np.int8)
            rds_ct_right = rds_ct_left.copy()

            pos_x_left = np.random.randint(0, nx, nDots_ct).astype(np.int32)
            pos_y_left = np.random.randint(0, ny, nDots_ct).astype(np.int32)
            pos_x_right = np.random.randint(0, nx, nDots_ct).astype(np.int32)
            pos_y_right = np.random.randint(0, ny, nDots_ct).astype(np.int32)
            # distribute white dots
            for d in np.arange(0, np.int(nDots_ct / 2)):
                rr, cc = disk(
                    (pos_x_left[d], pos_y_left[d]), rDot_pix, shape=np.shape(rds_ct)
                )
                rds_ct_left[rr, cc] = 1

                rr, cc = disk(
                    (pos_x_right[d], pos_y_right[d]), rDot_pix, shape=np.shape(rds_ct)
                )
                rds_ct_right[rr, cc] = 1

            # distribute black dots
            for d in np.arange(np.int(nDots_ct / 2) + 1, nDots_ct):
                rr, cc = disk(
                    (pos_x_left[d], pos_y_left[d]), rDot_pix, shape=np.shape(rds_ct)
                )
                rds_ct_left[rr, cc] = -1

                rr, cc = disk(
                    (pos_x_right[d], pos_y_right[d]), rDot_pix, shape=np.shape(rds_ct)
                )
                rds_ct_right[rr, cc] = -1

        elif dotMatch_ct == 0:  # make ards
            rds_ct_left = rds_ct.copy()
            rds_ct_right = rds_ct.copy()

            id_start = 0
            id_end = np.int32(len(dotID_ct) / 2)

            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), -1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), 1, rds_ct_right
            )

            id_start = id_end + 1
            id_end = len(dotID_ct) - 1
            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), 1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), -1, rds_ct_right
            )

        elif (dotMatch_ct > 0) & (dotMatch_ct < 1):
            rds_ct_left = rds_ct.copy()
            rds_ct_right = rds_ct.copy()

            num_dot_to_match = np.int32(dotMatch_ct * len(dotID_ct))
            # always make even number
            if num_dot_to_match % 2 != 0:
                num_dot_to_match = num_dot_to_match - 1

            dotID_to_match = dotID_ct[0:num_dot_to_match]

            # distribute black dots
            id_start = 0
            id_end = np.int32(len(dotID_to_match) / 2)

            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), -1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), -1, rds_ct_right
            )

            # distribute white dots
            id_start = id_end + 1
            id_end = len(dotID_to_match) - 1
            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), 1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), 1, rds_ct_right
            )

            ## set other dots in rds_ct to be unmatched
            id_start = id_end + 1
            # id_end = id_start + np.int((len(dotID_ct)-len(dotID_to_match))/2)
            id_end = np.int32(len(dotID_ct) - 1)
            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), -1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), 1, rds_ct_right
            )

            # for i in range(id_start, id_end):
            #     x = dotID_ct[i]
            #     rds_ct_left[rds_ct_left==x] = 0
            #     rds_ct_right[rds_ct_right==x] = 1

            # id_start = id_end + 1
            # id_end = len(dotID_ct) - 1
            # x0 = dotID_ct[id_start]
            # x1 = dotID_ct[id_end]
            # rds_ct_left = np.where((rds_ct_left>=x0) & (rds_ct_left<=x1), 1,
            #                        rds_ct_left)
            # rds_ct_right = np.where((rds_ct_right>=x0) & (rds_ct_right<=x1), 0,
            #                         rds_ct_right)

            # check other dotID in rds_ct_left and rds_ct_right that hasn't been
            # converted to 0 or 1
            rds_ct_left[rds_ct_left > 1] = 1
            rds_ct_right[rds_ct_right > 1] = 1

        elif dotMatch_ct == 1:  # make crds
            rds_ct_left = rds_ct.copy()
            rds_ct_right = rds_ct.copy()

            # distribute black dots
            id_start = 0
            id_end = np.int32(len(dotID_ct) / 2)

            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), -1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), -1, rds_ct_right
            )

            # distribute white dots
            id_start = id_end + 1
            id_end = len(dotID_ct) - 1
            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), 1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), 1, rds_ct_right
            )

        return rds_ct_left, rds_ct_right

    def create_rds(self, disp_ct_pix, dotMatch_ct):
        """
        create RDSs with disparity in disp_ct_pix

        rds_bg and rds_ct are a matrix with size_bg and size_ct, respectively:
            0.0 = gray background
            -1.0 = black dot
            1.0 = white dot

        Args:
            disp_ct_pix ([list]): horizontal disparity
                # disp_ct_pix < 0 -> (crossed-disparity) near:
                                   put the dots in RDS_right to the left RDS_left
                # disp_ct_pix > 0 -> (uncrossed-disparity) far:
                                    put the dots in RDS_right to the right RDS_left

            dotMatch_ct ([type]): dot match level
                -1.0 = urds (uncorrelated rds)
                0.0 = ards (anticorrelated rds)
                0.5: hmrds (half-matched rds)
                1.0 = crds (correlated rds)

        Returns:
            [type]: [description]
        """

        # calculate center position in pixel
        center = (self.h_bg // 2, self.w_bg // 2)
        # center = (rds.h_bg // 2, rds.w_bg // 2)

        # calculate the starting and the ending coordinate of the rds center
        row_ct_start = center[0] - self.h_ct // 2
        row_ct_end = row_ct_start + self.h_ct + 1
        col_ct_start = center[1] - self.w_ct // 2
        col_ct_end = col_ct_start + self.w_ct + 2
        # row_ct_start = center[0] - rds.h_ct // 2
        # row_ct_end = row_ct_start + rds.h_ct + 1
        # col_ct_start = center[1] - rds.w_ct // 2
        # col_ct_end = col_ct_start + rds.w_ct + 2

        # allocate memory for rds that follows the format "NHWC" (batch_size, height, width, in_channels)
        rdsDisp_channels = len(disp_ct_pix)
        rds_left_set = np.zeros(
            (rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32
        )
        rds_right_set = np.zeros(
            (rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32
        )
        # rds_left_set = np.zeros((rdsDisp_channels, rds.h_bg, rds.w_bg), dtype=np.int32)
        # rds_right_set = np.zeros((rdsDisp_channels, rds.h_bg, rds.w_bg), dtype=np.int32)

        for i in range(rdsDisp_channels):
            # create rds matrix for rds background and rds center that has pixel value -1, 0, and 1
            ## make rds background
            rds_bg = np.zeros(
                (self.h_bg, self.w_bg), dtype=np.int32
            )  # for indexing dots
            # rds_bg = np.zeros((h_bg, w_bg), dtype=np.int32)
            rds_left = rds_bg.copy()  # for black and white rds_bg
            rds_right = rds_bg.copy()

            # generate dot position
            xc, yc = self.generate_dot_position(self.overlap_flag)
            # xc, yc = rds.generate_dot_position(rds.overlap_flag)
            nDots = len(xc)

            # generate a binary number indicating the sign of dot contrast, depending on
            # the dotMatch level
            dotContrast_sign = np.random.choice(
                [-1, 1], size=nDots, p=[1 - dotMatch_ct, dotMatch_ct]
            )

            for d in np.arange(nDots):
                rr, cc = disk((yc[d], xc[d]), self.rDot, shape=(self.h_bg, self.w_bg))
                # rr, cc = disk((yc[d], xc[d]), rDot, shape=(h_bg, w_bg))

                rds_bg[rr, cc] = d

                # distribute white dots
                if d <= nDots // 2:
                    rds_left[rr, cc] = 1
                else:  # distribute black dots
                    rds_left[rr, cc] = -1

                ## shift dots in RDS_right to the left if they are located in rds_ct
                ## Rule for setting the horizontal disparity using the left disparity map:
                # disp_ct_pix > 0 -> (crossed-disparity) near:
                #                    put the dots in RDS_right to the right so that the
                #                    RDS_right dots are on the right side of RDS_left dots
                # disp_ct_pix < 0 -> (uncrossed-disparity) far:
                #                    put the dots in RDS_right to the left so that the
                #                    RDS_right dots are on the left side of RDS_left dots
                if (
                    (rr.mean() > row_ct_start)
                    & (rr.mean() < row_ct_end)
                    & (cc.mean() > col_ct_start)
                    & (cc.mean() < col_ct_end)
                ):
                    # shift the dots and set the dotMatch
                    rds_right[rr, cc + disp_ct_pix[i]] = (
                        dotContrast_sign[d] * rds_left[rr, cc]
                    )

                else:  # if the dots are outside of rds_ct
                    rds_right[rr, cc] = rds_left[rr, cc]

                    # randomly assign a dot next to the shifted dot around the border of rds_ct
                    # to fill the gap due to the shifting
                    if (
                        (disp_ct_pix[i] < 0)
                        & (rr.mean() > row_ct_start)
                        & (rr.mean() < row_ct_end)
                        & (cc.mean() > col_ct_end)
                        & (cc.mean() < col_ct_end - disp_ct_pix[i])
                    ):
                        dot_fill = np.random.choice([-1, 1], size=1, p=[1 / 2, 1 / 2])[
                            0
                        ]
                        rds_right[rr, cc + disp_ct_pix[i]] = dot_fill * rds_left[rr, cc]
                    elif (
                        (disp_ct_pix[i] > 0)
                        & (rr.mean() > row_ct_start)
                        & (rr.mean() < row_ct_end)
                        & (cc.mean() > col_ct_start - disp_ct_pix[i])
                        & (cc.mean() < col_ct_start)
                    ):
                        dot_fill = np.random.choice([-1, 1], size=1, p=[1 / 2, 1 / 2])[
                            0
                        ]
                        rds_right[rr, cc + disp_ct_pix[i]] = dot_fill * rds_left[rr, cc]

            rds_left_set[i] = rds_left
            rds_right_set[i] = rds_right

            # shift all pixels in rds so disparity of each pixel >= 0
            # rds_right_shift = np.roll(rds_right, -np.abs(disp_ct_pix[0]), axis=1)
            # rds_right_set[i] = rds_right_shift

        rds_all = np.zeros((2, rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32)
        rds_all[0] = rds_left_set
        rds_all[1] = rds_right_set

        return rds_all

    def create_rds_batch(self, disp_ct_pix, dotMatch_ct):
        """
        Make nBatch of random dot stereogram obtained from create_rds

        rds_bg and rds_ct are a matrix with size_bg and size_ct, respectively:
            0.0 = gray background
            -1.0 = black dot
            1.0 = white dot

        This module creates a set of rds with disparity listed on disp_ct_pix

        Inputs:
            - size_rds_bg: <tuple>, size of rds background, ex: (501,501)
            - size_rds_ct: <tuple> size of rds center, ex: (251,251)
            - disp_ct_pix: <np.array>, a list of disparity magnitude of center
                                        rds (pixel)

                        This variable is a kind of disparity axis in disparity
                        tuning curve

                        ex:
                        disp_ct_deg = np.round(np.arange(-0.4,
                                                         (0.4 + deg_per_pix),
                                                         deg_per_pix),
                                               2)
                        disp_ct_pix = cm.fxCompute_deg2pix(disp_ct_deg)

                disp_ct_pix > 0 = far -> dots in the left shifted to the right, rds_right to the left
                disp_ct_pix < 0 = near -> dots in the left shifted to the left, rds_right to the right

            - dotMatch_ct: <scalar>, dot match level of center rds, between 0 and 1.
                            -1 means uncorrelated RDS
                            0 means anticorrelated RDS
                            0.5 means half-matched RDS
                            1 means correlated RDS

            - dotDens: <scalar> dot density

            - rDot: <scalar> dot radius in degree

            - nBatch: <scalar> number of batch size (ex: 1000)

            - n_workers: <scalar>: number of cpu

        Outputs:
            rds_left_unpack, rds_right_unpack: <[nBatch, rdsDisp_channels, height, width] np.array>,
                            nBatch pair of rds with which are a mixed of rds_bg and rds_ct
        """

        #    nBatch = 10
        rdsDisp_channels = len(disp_ct_pix)

        now = datetime.now()
        time_start = now.strftime("%H:%M:%S")
        t_start = timer()
        rds_batch = []
        rds_batch.append(
            Parallel(n_jobs=-1)(
                delayed(self.create_rds)(disp_ct_pix, dotMatch_ct)
                for i in range(self.n_rds)
            )
        )
        t_end = timer()
        now = datetime.now()
        time_end = now.strftime("%H:%M:%S")
        print(time_start, time_end, t_end - t_start)

        # unpack rds_batch
        rds_left_unpack = np.zeros(
            (self.n_rds, rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32
        )
        rds_right_unpack = np.zeros(
            (self.n_rds, rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32
        )
        for i in range(self.n_rds):
            rds_unpack = rds_batch[0][i]

            rds_left_unpack[i] = rds_unpack[0]
            rds_right_unpack[i] = rds_unpack[1]

        return rds_left_unpack, rds_right_unpack

    def create_rds_without_bg(self, disp_ct_pix, dotMatch_ct):
        """
        Make a single plane of random dot stereogram (without background RDS).
        it means that the whole dots in RDS are shifted to set the disparity.

        The pixel values are as follow:
            0 = gray background
            -1 = black dot
            1 = white dot

        Outputs:
            rds_all: <[2, len(disp_ct_pix), size_rds_bg, size_rds_bg] np.array>,
                    A pair of rds with which is a
                    mixed of rds_bg and rds_ct
        """

        # rdsDisp_channels: <scalar>, number of disparity points in disparity tuning function
        # rdsDisp_channels = len(disp_ct_pix)

        rdsDisp_channels = len(disp_ct_pix)

        # allocate memory for rds that follows the format "NHWC" (batch_size, height, width, in_channels)
        rds_left_set = np.zeros(
            (rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32
        )
        rds_right_set = np.zeros(
            (rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32
        )

        for d in range(rdsDisp_channels):  # iterate over crossed-uncrossed disparity
            ## create rds matrix for rds background and rds center that has pixel value -1, 0, and 1
            # make rds background
            rds_bg = np.zeros(
                (self.h_bg, self.w_bg), dtype=np.int32
            )  # for indexing dots
            rds_bg2 = rds_bg.copy()  # for black and white rds_bg

            # generate dot position
            xc, yc = self.generate_dot_position(self.overlap_flag)
            # xc, yc = rds.generate_dot_position(rds.overlap_flag)
            nDots = len(xc)

            for i_dot in np.arange(nDots):  # iterate over dot ID
                rr, cc = disk(
                    (yc[i_dot], xc[i_dot]), self.rDot, shape=(self.h_bg, self.w_bg)
                )
                # rr, cc = disk((pos_x[d], pos_y[d]), rDot_pix,
                #               shape=(48,48))
                rds_bg[rr, cc] = i_dot

                # distribute white dots
                if i_dot <= np.int32(nDots / 2):
                    rds_bg2[rr, cc] = 1
                else:  # distribute black dots
                    rds_bg2[rr, cc] = -1

            ## set dotMatch level
            rds_bg_left, rds_bg_right = self._set_dotMatch(
                rds_bg, dotMatch_ct, nDots, self.rDot
            )

            ## make rds_left:
            # disp_ct_pix < 0 -> (crossed-disparity) near:
            #                       put the dots in RDS_right to the left RDS_left
            # disp_ct_pix > 0 -> (uncrossed-disparity) far:
            #                       put the dots in RDS_right to the right RDS_left
            # rds_left_set[d, :, :] = rds_left
            rds_left_set[d, :, :] = rds_bg_left

            ## make rds_right: shift all pixels according to the given disparity
            # disp_ct_pix < 0 -> (crossed-disparity) near:
            #                       put the dots in RDS_right to the left RDS_left
            # disp_ct_pix > 0 -> (uncrossed-disparity) far:
            #                       put the dots in RDS_right to the right RDS_left
            rds_right = np.roll(
                rds_bg_right, disp_ct_pix[d], axis=1
            )  # set disparity magnitude
            rds_right_set[d, :, :] = rds_right

        ## alocate array to store the left and right rds images
        rds_all = np.zeros((2, rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32)

        rds_all[0] = rds_left_set
        rds_all[1] = rds_right_set

        return rds_all

    def create_rds_without_bg_batch(self, disp_ct_pix, dotMatch_ct):
        """
        Make nBatch of random dot stereogram obtained from fxCreate_rds

        rds_bg and rds_ct are a matrix with size_bg and size_ct, respectively:
            0.5 = gray background
            0 = black dot
            1 = white dot

        This module creates a set of rds with disparity listed on disp_ct_pix


        Inputs:
            - size_rds_bg: <tuple>, size of rds background, ex: (501,501)
            - size_rds_ct: <tuple> size of rds center, ex: (251,251)
            - disp_ct_pix: <np.array>, a list of disparity magnitude of center
                                        rds (pixel)

                        This variable is a kind of disparity axis in disparity
                        tuning curve

                        ex:
                        disp_ct_deg = np.round(np.arange(-0.4,
                                                         (0.4 + deg_per_pix),
                                                         deg_per_pix),
                                               2)
                        disp_ct_pix = cm.fxCompute_deg2pix(disp_ct_deg)

            - dotMatch_ct: <scalar>, dot match level of center rds, between 0 and 1.
                            -1 means uncorrelated RDS
                            0 means anticorrelated RDS
                            0.5 means half-matched RDS
                            1 means correlated RDS

            - dotDens: <scalar> dot density

            - rDot: <scalar> dot radius in degree

            - nBatch: <scalar> number of batch size (ex: 1000)

            - n_workers: <scalar>: number of cpu

        Outputs:
            rds_left_unpack: <[n_trials, len(disp_ct_pix),
                             size_rds_bg, size_rds_bg] np.array>,
                            n_trials pair of rds whose whole pixels are shifted

            rds_right_unpack: <[n_trials, len(disp_ct_pix),
                             size_rds_bg, size_rds_bg] np.array>,
                            n_trials pair of rds whose whole pixels are shifted
        """

        #    nBatch = 10
        rdsDisp_channels = len(disp_ct_pix)

        now = datetime.now()
        time_start = now.strftime("%H:%M:%S")
        t_start = timer()
        rds_batch = []
        rds_batch.append(
            Parallel(n_jobs=-1)(
                delayed(self.create_rds_without_bg)(disp_ct_pix, dotMatch_ct)
                for i in range(self.n_rds)
            )
        )

        t_end = timer()
        now = datetime.now()
        time_end = now.strftime("%H:%M:%S")
        print(time_start, time_end, t_end - t_start)

        # unpack rds_batch
        rds_left_unpack = np.zeros(
            (self.n_rds, rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32
        )
        rds_right_unpack = np.zeros(
            (self.n_rds, rdsDisp_channels, self.h_bg, self.w_bg), dtype=np.int32
        )
        for i in range(self.n_rds):
            rds_unpack = rds_batch[0][i]

            rds_left_unpack[i] = rds_unpack[0]
            rds_right_unpack[i] = rds_unpack[1]

        return rds_left_unpack, rds_right_unpack
