import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

from .config import RESULTS_DIR, PRED_DIR, PLOTS_DIR, AUG_DIR
from .augmentations import (
    augment_geometric_replay,
    augment_intensity_wrapper,
    augment_once,
)
from .data import preprocess_pair
from .data import match_pairs


# ================================
# 1) ORIGINAL VS AUGMENTED SLICES
# ================================

def draw_mask_contours(ax, image, mask, color="red", lw=1.2):
    """Overlay mask contours on a grayscale image."""
    ax.imshow(image, cmap="gray")
    contours = find_contours(mask.astype(np.uint8), 0.5)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], linewidth=lw, color=color)
    ax.axis("off")


def show_original_vs_augmented(
    pairs,
    n: int = 3,
    z_mode: str = "center",
    save_tag: str = "train",
):
    """
    Visualize original vs augmented slices.

    Saves:
        RESULTS_DIR/overlay_images/aug_examples_{save_tag}.png
    """
    if len(pairs) == 0:
        raise ValueError("No pairs provided to show_original_vs_augmented.")

    n = min(n, len(pairs))
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, (ip, mp) in enumerate(pairs[:n]):
        img, msk, _ = preprocess_pair(ip, mp, crop=True)

        try:
            comb_img, comb_msk = augment_once(img.copy(), msk.copy())
        except Exception as e:
            print(f"[Show Original vs Augmented] Combined augment failed for {ip}: {e}")
            comb_img, comb_msk = img.copy(), msk.copy()

        D = img.shape[0]
        z = (D // 2) if z_mode == "center" else np.random.randint(0, D)

        orig_slice = img[z, ..., 0]
        orig_mask = msk[z, ..., 0] > 0.5

        comb_slice = comb_img[z, ..., 0]
        comb_mask = comb_msk[z, ..., 0] > 0.5

        draw_mask_contours(axes[i, 0], orig_slice, orig_mask, "red")
        axes[i, 0].set_title("Original")

        draw_mask_contours(axes[i, 3], comb_slice, comb_mask, "lime")
        axes[i, 3].set_title("Combined")

    plt.tight_layout()
    out_path = os.path.join(AUG_DIR, f"aug_examples_{save_tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[show_original_vs_augmented] Saved: {out_path}")


# ================================================
# 2) TRAINING & VALIDATION CURVES (LOSS + METRICS)
# ================================================

def plot_history(run_name: str):
    """
    Plot training & validation curves (loss and Dice) for a given run.

    Expects:
        RESULTS_DIR/logs/{run_name}_hist.json

    Saves:
        RESULTS_DIR/plots/training_curves_{run_name}.png
    """
    hist_path = os.path.join(RESULTS_DIR, "logs", f"{run_name}_hist.json")
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"History file not found: {hist_path}")

    with open(hist_path) as f:
        h = json.load(f)

    fig = plt.figure(figsize=(10, 4))

    # ---- Loss ----
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(h["loss"], label="train")
    ax1.plot(h["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title(f"{run_name} Loss")

    # ---- Dice ----
    ax2 = plt.subplot(1, 2, 2)
    if "dice_coef" in h and "val_dice_coef" in h:
        ax2.plot(h["dice_coef"], label="train")
        ax2.plot(h["val_dice_coef"], label="val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice")
        ax2.legend()
        ax2.set_title(f"{run_name} Dice")
    else:
        ax2.text(0.5, 0.5, "No Dice metrics found", ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"training_curves_{run_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_history] Saved: {out_path}")


# ===========================================
# 3) PREDICTED VS GROUND-TRUTH ON TEST CASES
# ===========================================

def show_test_predictions(
    pairs,
    run_name: str,
    n: int = 3,
    z_mode: str = "center",
):
    """
    Visual examples of predicted vs ground truth segmentations on test cases.

    Assumes inference.py has already saved:
        PRED_DIR/{run_name}_{case}.npz  with "pred" and (optionally) "prob".

    Saves (per case PNG):
        PRED_DIR/pred_vs_gt_{run_name}_{case}.png
    """
    if len(pairs) == 0:
        raise ValueError("No test pairs provided to show_test_predictions.")

    n = min(n, len(pairs))

    for img_path, msk_path in pairs[:n]:
        img, msk, _ = preprocess_pair(img_path, msk_path, crop=False)
        gt = msk[..., 0] > 0.5

        case = os.path.basename(img_path).split(".nii")[0]

        pred_file = os.path.join(PRED_DIR, f"{run_name}_{case}.npz")
        if not os.path.exists(pred_file):
            print(f"[Show Test Predictions] Prediction file not found: {pred_file} (skipping)")
            continue

        data = np.load(pred_file)
        if "pred" not in data:
            print(f"[Show Test Predictions] 'pred' key missing in {pred_file} (skipping)")
            continue

        pred = data["pred"] > 0.5

        # Choose slice
        D = img.shape[0]
        z = (D // 2) if z_mode == "center" else np.random.randint(0, D)

        img_slice = img[z, ..., 0]
        gt_slice = gt[z]
        pred_slice = pred[z]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # GT
        draw_mask_contours(axes[0], img_slice, gt_slice, color="green")
        axes[0].set_title(f"{case} – GT")

        # Prediction
        draw_mask_contours(axes[1], img_slice, pred_slice, color="red")
        axes[1].set_title(f"{case} – {run_name} prediction")

        plt.tight_layout()
        out_path = os.path.join(PRED_DIR, f"pred_vs_gt_{run_name}_{case}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[Show Test Predictions] Saved: {out_path}")
