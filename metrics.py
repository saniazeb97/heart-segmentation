import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def sensitivity(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Sensitivity (Recall) = TP / (TP + FN)"""
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    tp = np.sum(pred_bin & gt_bin)
    fn = np.sum(~pred_bin & gt_bin)

    return float(tp / (tp + fn + eps))


def precision(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Precision = TP / (TP + FP)"""
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & ~gt_bin)

    return float(tp / (tp + fp + eps))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    """95th percentile Hausdorff distance (HD95) in mm."""
    pred = pred > 0.5
    gt = gt > 0.5

    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")

    struct = np.ones((3, 3, 3), dtype=bool)
    p_surf = pred & ~binary_erosion(pred, struct)
    g_surf = gt & ~binary_erosion(gt, struct)

    dt_p = distance_transform_edt(~p_surf, sampling=spacing)
    dt_g = distance_transform_edt(~g_surf, sampling=spacing)

    dists = np.concatenate([dt_g[p_surf], dt_p[g_surf]])
    if dists.size == 0:
        return float("inf")

    return float(np.percentile(dists, 95))
