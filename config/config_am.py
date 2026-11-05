import warnings
from dataclasses import dataclass


@dataclass
class ConfigAM:
    """
    config for activation maximization
    """

    input_size: tuple = (
        3,
        96,
        96,
    )  # [rgb, H, W]: input size, ideally, the same with training image sizes.
    # here use [3, 96, 96] for visibility. [256, 512] will make the resulting RF structure
    # too small to see
    size_multiple: int = 2  # H,W must be multiples of this (BNN 2; GC_Net_LR 32)
    restarts: int = 4  # the number of independent optimization runs (in parallel)
    # with different random initializations performed for each target neuron.

    iterations: int = 1000
    lr: float = 0.01  # (bnn: 0.01)
    color_mode: str = "rgb"  # rgb or gray

    # --- FFT-prior parameters
    parameterization: str = "fft"  # fft or pixel (both use sigmoid)
    color_decorrelate: bool = (
        True  # # removes rainbow speckle; False = original behaviour
    )
    lowpass: float | None = (
        None  # soft cutoff in cycles/px, e.g. 0.25; None = off (bnn: 0.5)
    )
    eye_init: str = (
        "independent"  # "shared": eyes start as shifted copies(bnn: independent)
    )
    eye_init_shift_px: float | None = (
        None  # shared init disparity; None -> disp_px or 0
    )
    decay: float = 1.0  # FFT spectrum prior (1/f^decay); compare 0 and 1 (BNN: 1.0).
    # larger values increase the penalty on high-frequency energy relative to
    # low-frequency structural components, smoothing out high-frequency
    # grid patterns, checkerboard artifacts (common in transposed convolution/up-sampling),
    # and sharp pixel noise.

    init_std: float = 0.05
    sign: int = 1  # -1 minimizes the raw activation / cost

    l2: float = 0.1  # 0.0, optional penalty around raw gray .5 (bnn: 0.05)
    # L2 regularization (weight decay or ridge penalty) constrains extreme feature
    # activation sizes or pixel intensities. Effect: Raising lambda_L_2 helps
    # prevent high-frequency "checkerboard" artifacts or
    # exploding pixel values caused by unrestrained optimization.

    tv: float = (
        0.1  # total variation regularizer. 0.0, optional squared spatial differences. (bnn: 0.05)
    )
    # Total Variation specifically penalizes sharp, unwanted pixel variations and high-frequency noise
    # while preserving true structural edges.
    # Raising tv smooths out grainy artifacts, ringing, and pixelation,
    # encouraging piecewise-smooth regions.

    ref: int = 1  # +1 physical left reference, -1 right
    disp_idx: int | None = (
        None  # explicit depth INDEX (None and disp_px None -> D//2 = 0 px)
    )
    disp_px: float | None = (
        None  # alternative: disparity in input pixels (needs max_disp)
    )
    max_disp: int = 192  # config.max_disp; needed for disp_px / reporting
    window: int = 3  # controls the spatial neighborhood size (W X W) extracted and
    # averaged from the target feature map to form the optimization objective.
    # must be odd number.
    # For example: 1 is one neuron; 3 is a region mean of 3x3 window (avg of 9 neurons)

    # --- robustness transforms (identical for both eyes)
    jitter: int = 4  # integer px roll, same for both eyes; 0 disables
    scale_range: tuple = (1.0, 1.0)  # zoom also rescales disparity -> off by default
    eval_every: int = 10  # evaluate the UNtransformed pair every k steps

    # --- opt-in: modifies the BACKWARD pass of nn.ReLU (not the true gradient)
    redirected_relu: bool = False
    seed: int = 42
    rf_samples: int = 4  # additional independent random probes
    rf_mass: float = 0.95
    baseline: float = (
        0.5  # raw RGB baseline used for ablation analysis (the pixel values for blank images)
    )

    # --- performance
    batch_size: int = (
        64  # | None = None  # jobs (channel x restart) per pass; None = auto
    )
    fast_path: bool = True  # exact local evaluation of decoder.layer3.0
    store_restart_images: bool = False
    store_gradient: bool = False

    def validate(self):
        c, h, w = self.input_size
        m = self.size_multiple
        if int(m) != m or m < 2 or m % 2:
            raise ValueError("size_multiple must be an even integer >= 2")
        if c != 3 or min(h, w) < 4 or h % m or w % m:
            raise ValueError(
                f"Use RGB and H,W >= 4 that are multiples of {m} for this network"
            )
        if self.restarts < 1 or self.iterations < 0 or self.lr <= 0:
            raise ValueError("restarts>=1, iterations>=0, lr>0 required")
        if self.window < 1 or self.window % 2 != 1:
            raise ValueError("window must be a positive odd integer")
        if self.sign not in (-1, 1) or self.ref not in (-1, 1):
            raise ValueError("sign and ref must be +1 or -1")
        if self.parameterization not in ("fft", "pixel"):
            raise ValueError("parameterization must be fft or pixel")
        if self.color_mode not in ("rgb", "gray"):
            raise ValueError("color_mode must be rgb or gray")
        if self.eye_init not in ("independent", "shared"):
            raise ValueError("eye_init must be independent or shared")
        if self.lowpass is not None and not 0 < self.lowpass <= 0.5:
            raise ValueError("lowpass is a cutoff in cycles/px, in (0, 0.5]")
        if self.parameterization == "pixel" and (
            self.color_decorrelate or self.lowpass or self.eye_init != "independent"
        ):
            warnings.warn(
                "color_decorrelate / lowpass / eye_init only act on the fft parameterization"
            )
        if min(self.l2, self.tv, self.decay, self.rf_samples) < 0:
            raise ValueError("penalties, decay, and rf_samples must be nonnegative")
        if (
            self.init_std <= 0
            or not 0 < self.rf_mass <= 1
            or not 0 <= self.baseline <= 1
        ):
            raise ValueError("invalid init_std, rf_mass, or baseline")
        if int(self.jitter) != self.jitter or self.jitter < 0:
            raise ValueError("jitter must be a nonnegative integer (pixels)")
        if not 0 < self.scale_range[0] <= self.scale_range[1]:
            raise ValueError("0<scale_range[0]<=scale_range[1] required")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        if self.disp_idx is not None and self.disp_px is not None:
            raise ValueError("give disp_idx or disp_px, not both")
        if self.disp_px is not None and self.max_disp is None:
            raise ValueError("disp_px needs max_disp")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_disp is not None and w < 2 * self.max_disp:
            warnings.warn(
                f"width {w} < 2*max_disp ({2 * self.max_disp}): part of the cost volume "
                "is zero padding and normalisation statistics differ from training"
            )
