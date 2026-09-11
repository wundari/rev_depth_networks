#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import random

import cv2
import numpy as np
import torch
from albumentations.core.transforms_interface import BasicTransform

"""
functions that cannot fit in albumentation framework
"""


def get_random_crop_coords(height, width, crop_height, crop_width):
    """
    get coordinates for cropping

    :param height: image height, int
    :param width: image width, int
    :param crop_height: crop height, int
    :param crop_width: crop width, int
    :return: xy coordinates
    """
    y1 = random.randint(0, height - crop_height)
    y2 = y1 + crop_height
    x1 = random.randint(0, width - crop_width)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def crop(img, x1, y1, x2, y2):
    """
    crop image given coordinates

    :param img: input image, [H,W,3]
    :param x1: coordinate, int
    :param y1: coordinate, int
    :param x2: coordinate, int
    :param y2: coordinate, int
    :return: cropped image
    """
    img = img[y1:y2, x1:x2]
    return img


def horizontal_flip(
    img_left, img_right, occ_left, occ_right, disp_left, disp_right, split
):
    """
    horizontal flip left and right images, then disparity has to be swapped

    :param img_left: left image, [H,W,3]
    :param img_right: right image, [H,W,3]
    :param occ_left: left occlusion mask, [H,W]
    :param occ_right: right occlusion mask, [H,W]
    :param disp_left: left disparity, [H,W]
    :param disp_right: right disparity, [H,W]
    :param split: train/validation split, string
    :return: updated data
    """
    if split == "validation":
        p = 0.0
    else:
        p = random.random()

    # if hflip, we flip left/right, and read everything of right images
    if p >= 0.5:
        left_flipped = img_left[:, ::-1]
        right_flipped = img_right[:, ::-1]
        img_left = right_flipped
        img_right = left_flipped

        occ = occ_right[:, ::-1]
        occ_right = occ_left[:, ::-1]
        disp = disp_right[:, ::-1]
        disp_right = disp_left[:, ::-1]
    else:
        occ = occ_left
        disp = disp_left

    return img_left, img_right, occ, occ_right, disp, disp_right


def random_crop(
    min_crop_height,
    min_crop_width,
    input_data,
    c_disp_shift,
    split,
    reference=None,
    rng=None,
):
    """Uniform valid stereo crop, with the original signed disparity convention.

    ``reference`` optionally selects the eye before loading its PFM file.
    A local RNG allows repeatable evaluation without changing training randomness.
    """
    rng = random if rng is None else rng

    h, w = input_data["left"].shape[:2]
    ch, cw = min_crop_height, min_crop_width
    if input_data["right"].shape[:2] != (h, w):
        raise ValueError("Stereo images must have matching spatial shapes")
    if ch <= 0 or cw <= 0 or ch > h or cw > w:
        raise ValueError("Crop must be positive and fit inside the source image")
    if reference is None:
        reference = 1 if rng.random() <= 0.5 else -1
    if reference not in (1, -1):
        raise ValueError("reference must be 1 or -1")

    shift = c_disp_shift * 44.0
    if not np.isfinite(shift):
        raise ValueError("Disparity shift must be finite")
    offset = int(shift)
    delta = reference * offset
    low, high = max(0, delta), min(w - cw, w - cw + delta)
    if low > high:
        raise ValueError("No valid stereo crop for this width and disparity shift")

    x = rng.randint(low, high)
    y = rng.randint(0, h - ch)
    source = input_data["disp"] if reference == 1 else input_data["disp_right"]
    if source.shape != (h, w):
        raise ValueError("Selected disparity map must match the source image")

    left, right = input_data["left"], input_data["right"]
    anchor, other = (left, right) if reference == 1 else (right, left)
    input_data["left"] = crop(anchor, x, y, x + cw, y + ch)
    input_data["right"] = crop(other, x - delta, y, x - delta + cw, y + ch)

    # Keep the original full-image width as the clipping bound.
    shifted = crop(source, x, y, x + cw, y + ch) - offset
    input_data["disp"] = np.minimum(shifted, w)
    input_data["ref"] = reference

    return input_data


###
def shift_disparity_map(img_disp, c_disp_shift):

    w = img_disp.shape[-1]

    # shift = int(np.clip(c_disp_shift * np.median(img_disp), 0, 255))
    # shift = np.clip(c_disp_shift * np.mean(img_disp), 0, 255)
    # shift = np.clip(c_disp_shift * np.median(img_disp), 0, 255)

    # shift: mean of disparity map in monkaa training dataset (43.88 pixels)
    # flying training dataset (44.05 pixels)
    shift = c_disp_shift * 44.0
    img_disp_shifted = img_disp - shift

    # clip img_disp_shifted
    img_disp_shifted[img_disp_shifted > w] = w
    # img_disp_shifted = np.clip(img_disp - shift, -128, 127)  # .astype(np.float32)

    return img_disp_shifted, shift


###

"""
Base
"""


class StereoTransform(BasicTransform):
    """Albumentations 2.x adapter for GCNet's left/right dictionary targets.

    Disparities, occlusion masks and reference signs are passed through.
    ``always_apply`` is a compatibility alias for existing callers only.
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(p=1.0 if always_apply else p)

    @property
    def targets(self):
        return {"left": self.apply, "right": self.apply}

    def update_transform_params(self, params, data):
        # BasicTransform 2.x otherwise looks for image/images, which GCNet
        # does not supply. Delegate using a temporary shape reference.
        image = data["left"] if "left" in self.targets else data["right"]
        return super().update_transform_params(params, {"image": image})


class RightOnlyTransform(StereoTransform):
    """Transform only the right image (sensor misalignment augmentation)."""

    @property
    def targets(self):
        return {"right": self.apply}


class StereoTransformAsym(StereoTransform):
    """Share sampled parameters unless the asymmetric branch is selected."""

    def __init__(self, always_apply=False, p=0.5, p_asym=0.2):
        super().__init__(always_apply=always_apply, p=p)
        if not 0 <= p_asym <= 1:
            raise ValueError("p_asym must be in [0, 1]")
        self.p_asym = p_asym

    @property
    def targets(self):
        return {"left": self.apply_l, "right": self.apply_r}

    @property
    def targets_as_params(self):
        return ["left", "right"]

    def asym(self):
        return self.py_random.random() < self.p_asym


def _limit_pair(value, nonnegative=False):
    pair = (0 if nonnegative else -value, value) if np.isscalar(value) else tuple(value)
    if len(pair) != 2 or pair[0] > pair[1] or (nonnegative and pair[0] < 0):
        raise ValueError("Expected an ordered pair of limits")
    return pair


def _clip_like(values, image):
    if image.dtype == np.uint8:
        maximum = 255
    elif image.dtype == np.float32:
        maximum = 1.0
    else:
        raise TypeError("Stereo photometric transforms support uint8 and float32")
    return np.clip(values, 0, maximum).astype(image.dtype)


def _shift_rgb(image, r, g, b):
    # Preserve the 1.3.1 clipping and uint8 truncation behavior.
    values = image.astype(np.float32) + np.array([r, g, b], dtype=np.float32)
    return _clip_like(values, image)


def _brightness_contrast(image, alpha, beta, brightness_by_max):
    # Keep the old mean-based brightness convention, including alpha.
    if image.dtype == np.uint8:
        values = np.arange(256, dtype=np.float32) * alpha
        values += beta * 255 if brightness_by_max else alpha * beta * np.mean(image)
        return cv2.LUT(image, _clip_like(values, image))
    values = image.astype(np.float32) * alpha
    values += beta if brightness_by_max else beta * np.mean(values)
    return _clip_like(values, image)


class Normalize(StereoTransform):
    """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel
    and divide by std per channel.

    Args:
        mean (float, list of float): mean values
        std  (float, list of float): std values
        max_pixel_value (float): maximum possible pixel value

    Targets:
        left, right

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        # mean=(0.5, 0.5, 0.5),
        # std=(0.5, 0.5, 0.5),
        max_pixel_value=255.0,
        always_apply=False,
        p=1.0,
    ):
        super(Normalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image, **params):
        mean = np.asarray(self.mean, dtype=np.float32) * self.max_pixel_value
        scale = np.reciprocal(
            np.asarray(self.std, dtype=np.float32) * self.max_pixel_value
        )
        return (image.astype(np.float32) - mean) * scale

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")


class ToTensor(StereoTransform):
    """Change input from HxWxC to CxHxW

    Targets:
        left, right

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=1.0):
        super(ToTensor, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class ToGrayStereo(StereoTransform):
    def apply(self, image, **params):
        return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)


class GaussNoiseStereo(StereoTransformAsym):
    """Apply gaussian noise to the input image.

    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self, var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5, p_asym=0.2
    ):
        StereoTransformAsym.__init__(self, always_apply, p, p_asym)
        self.var_limit = _limit_pair(var_limit, nonnegative=True)
        self.mean = mean

    def apply_l(self, img, gauss_l=None, **params):
        return _clip_like(img.astype(np.float32) + gauss_l, img)

    def apply_r(self, img, gauss_r=None, **params):
        return _clip_like(img.astype(np.float32) + gauss_r, img)

    def get_params_dependent_on_data(self, params, data):

        image = data["left"]
        var = self.py_random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var**0.5
        random_state = self.random_generator

        gauss_l = random_state.normal(self.mean, sigma, image.shape)

        if self.asym():
            image = data["right"]
            var = self.py_random.uniform(self.var_limit[0], self.var_limit[1])
            sigma = var**0.5
            random_state = self.random_generator

            gauss_r = random_state.normal(self.mean, sigma, image.shape)
        else:
            gauss_r = gauss_l
        return {"gauss_l": gauss_l, "gauss_r": gauss_r}


class RGBShiftStereo(StereoTransformAsym):
    """Randomly shift values for each channel of the input RGB image.

    Args:
        r_shift_limit ((int, int) or int): range for changing values for the red channel. If r_shift_limit is a single
            int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
        g_shift_limit ((int, int) or int): range for changing values for the green channel. If g_shift_limit is a
            single int, the range  will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
        b_shift_limit ((int, int) or int): range for changing values for the blue channel. If b_shift_limit is a single
            int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        r_shift_limit=20,
        g_shift_limit=20,
        b_shift_limit=20,
        always_apply=False,
        p=0.5,
        p_asym=0.2,
    ):
        StereoTransformAsym.__init__(self, always_apply, p, p_asym)
        self.r_shift_limit = _limit_pair(r_shift_limit)
        self.g_shift_limit = _limit_pair(g_shift_limit)
        self.b_shift_limit = _limit_pair(b_shift_limit)

    def apply_l(self, image, r_shift_l=0, g_shift_l=0, b_shift_l=0, **params):
        return _shift_rgb(image, r_shift_l, g_shift_l, b_shift_l)

    def apply_r(self, image, r_shift_r=0, g_shift_r=0, b_shift_r=0, **params):
        return _shift_rgb(image, r_shift_r, g_shift_r, b_shift_r)

    def get_params_dependent_on_data(self, params, data):
        r_shift_l = self.py_random.uniform(self.r_shift_limit[0], self.r_shift_limit[1])
        g_shift_l = self.py_random.uniform(self.g_shift_limit[0], self.g_shift_limit[1])
        b_shift_l = self.py_random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])

        if self.asym():
            r_shift_r = self.py_random.uniform(
                self.r_shift_limit[0], self.r_shift_limit[1]
            )
            g_shift_r = self.py_random.uniform(
                self.g_shift_limit[0], self.g_shift_limit[1]
            )
            b_shift_r = self.py_random.uniform(
                self.b_shift_limit[0], self.b_shift_limit[1]
            )
        else:
            r_shift_r = r_shift_l
            g_shift_r = g_shift_l
            b_shift_r = b_shift_l

        return {
            "r_shift_l": r_shift_l,
            "g_shift_l": g_shift_l,
            "b_shift_l": b_shift_l,
            "r_shift_r": r_shift_r,
            "g_shift_r": g_shift_r,
            "b_shift_r": b_shift_r,
        }


class RandomBrightnessContrastStereo(StereoTransformAsym):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        brightness_limit=0.1,
        contrast_limit=0.1,
        brightness_by_max=True,
        always_apply=False,
        p=0.5,
        p_asym=0.2,
    ):
        StereoTransformAsym.__init__(self, always_apply, p, p_asym)
        self.brightness_limit = _limit_pair(brightness_limit)
        self.contrast_limit = _limit_pair(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply_l(self, img, alpha_l=1.0, beta_l=0.0, **params):
        return _brightness_contrast(img, alpha_l, beta_l, self.brightness_by_max)

    def apply_r(self, img, alpha_r=1.0, beta_r=0.0, **params):
        return _brightness_contrast(img, alpha_r, beta_r, self.brightness_by_max)

    def get_params_dependent_on_data(self, params, data):
        alpha_l = 1.0 + self.py_random.uniform(
            self.contrast_limit[0], self.contrast_limit[1]
        )
        beta_l = 0.0 + self.py_random.uniform(
            self.brightness_limit[0], self.brightness_limit[1]
        )

        if self.asym():
            alpha_r = 1.0 + self.py_random.uniform(
                self.contrast_limit[0], self.contrast_limit[1]
            )
            beta_r = 0.0 + self.py_random.uniform(
                self.brightness_limit[0], self.brightness_limit[1]
            )
        else:
            alpha_r = alpha_l
            beta_r = beta_l

        return {
            "alpha_l": alpha_l,
            "beta_l": beta_l,
            "alpha_r": alpha_r,
            "beta_r": beta_r,
        }


"""
Right Image Only
"""


class RandomShiftRotate(RightOnlyTransform):
    """Randomly apply vertical translate and rotate the input.
    Args:
        max_shift (float): maximum shift in pixels along vertical direction. Default: 1.5.
        max_rotation (float): maximum rotation in degree. Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(self, max_shift=1.5, max_rotation=0.2, always_apply=False, p=1.0):
        super(RandomShiftRotate, self).__init__(always_apply, p)
        self.max_shift = max_shift
        self.max_rotation = max_rotation

    def get_params(self):
        return {
            "shift": self.py_random.uniform(-self.max_shift, self.max_shift),
            "rotation": self.py_random.uniform(-self.max_rotation, self.max_rotation),
        }

    def apply(self, img, shift=0.0, rotation=0.0, **params):
        h, w = img.shape[:2]
        matrix = np.float32(
            [
                [np.cos(np.deg2rad(rotation)), -np.sin(np.deg2rad(rotation)), 0],
                [np.sin(np.deg2rad(rotation)), np.cos(np.deg2rad(rotation)), shift],
            ]
        )

        return cv2.warpAffine(
            img, matrix, (w, h), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )


def aligned_random_crop(height, width, input_data, split):
    """Crop all matching spatial arrays together, without stereo eye swapping."""
    if split != "train":
        return input_data
    h, w = input_data["left"].shape[:2]
    x1, y1, x2, y2 = get_random_crop_coords(h, w, height, width)
    return {
        key: (
            crop(value, x1, y1, x2, y2)
            if isinstance(value, np.ndarray) and value.shape[:2] == (h, w)
            else value
        )
        for key, value in input_data.items()
    }
