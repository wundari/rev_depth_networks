"""Collect actual paired-eye ROI predictions from physical-eye-aware models.

This adapter is for GC_Net_LR / GC_NetGN_LR or models with the same contract:
NestedTensor.left/right preserve physical eye identity; ref=+1/-1 selects the
reference; forward returns signed [B,H,W] disparities. Do not apply it directly
to a model trained with an input-swapping convention without an explicit adapter.
Keep corresponding target ROIs away from occlusions and stimulus boundaries.

Some repository networks disable BatchNorm running statistics even in eval.
Their outputs depend on batch composition. Both reference calls below therefore
use the same batch; record batch size/composition in experiment provenance.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

from utilities.misc import NestedTensor


def normalize_rds(image):
    """Use the repository's signed RDS -> ImageNet normalization convention."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected signed RDS image [height,width,3]")
    if not np.isfinite(image).all() or image.min() < -1 or image.max() > 1:
        raise ValueError("RDS pixels must be finite in [-1,1]")
    tensor = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32).permute(2, 0, 1)
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor + 1) / 2 - mean) / std


def _roi(roi, height, width):
    if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) for v in roi):
        raise ValueError("ROI must be four integers: y0,y1,x0,x1")
    y0, y1, x0, x1 = roi
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("ROI must be nonempty and inside the prediction map")
    return slice(y0, y1), slice(x0, x1)


@torch.inference_mode()
def predict_paired_roi(model, left, right, roi_left, *, roi_right=None, amp_dtype=None):
    """Return [batch,2] measured signed means; never enforce an eye sign flip.

    left/right are already normalized [B,3,H,W] tensors on the model device.
    Supply eye-specific ROIs when required to sample the same target surface.
    Model training/eval state is restored even when inference raises an error.
    """
    if left.ndim != 4 or left.shape != right.shape or left.shape[0] == 0:
        raise ValueError("left/right must have matching nonempty [B,C,H,W] shapes")
    if left.device != right.device:
        raise ValueError("left/right must use the same device")
    height, width = left.shape[-2:]
    rois = (_roi(roi_left, height, width),
            _roi(roi_left if roi_right is None else roi_right, height, width))
    training = model.training
    results = []
    model.eval()
    try:
        for sign, roi in zip((1, -1), rois):
            inputs = NestedTensor(left=left, right=right,
                                  ref=torch.full((len(left),), sign, device=left.device))
            context = torch.autocast(device_type=left.device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
            with context:
                disparity = model(inputs)
            if disparity.shape != (len(left), height, width):
                raise ValueError("Expected model output [B,H,W] matching input spatial dimensions")
            target = disparity[:, roi[0], roi[1]].float()
            if not torch.isfinite(target).all():
                raise ValueError("Model produced nonfinite disparity inside the ROI")
            results.append(target.mean(dim=(-2, -1)).cpu().numpy())
    finally:
        model.train(training)
    return np.stack(results, axis=-1)
