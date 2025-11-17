import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from .config import (
    TARGET_SPACING,
    RESULTS_DIR,
    PRED_DIR,
    TEST_IMG_DIR,
    TEST_MSK_DIR,
    CKPT_DIR,
    PATCH_SIZE,
)
from .data import preprocess_pair, match_pairs
from .metrics import hd95, sensitivity, precision
from .losses import dice_coef
from .model import build_unet


def sliding_window_predict(vol, model, patch_size, overlap=0.5):
    """Sliding-window prediction over a full 3D volume."""
    D, H, W, _ = vol.shape
    pd, ph, pw = patch_size

    step_z = max(1, int(pd * (1 - overlap)))
    step_y = max(1, int(ph * (1 - overlap)))
    step_x = max(1, int(pw * (1 - overlap)))

    zs = list(range(0, max(D - pd + 1, 1), step_z))
    ys = list(range(0, max(H - ph + 1, 1), step_y))
    xs = list(range(0, max(W - pw + 1, 1), step_x))

    if zs[-1] != D - pd:
        zs.append(D - pd)
    if ys[-1] != H - ph:
        ys.append(H - ph)
    if xs[-1] != W - pw:
        xs.append(W - pw)

    prob = np.zeros(vol.shape[:3], dtype=np.float32)
    count = np.zeros(vol.shape[:3], dtype=np.float32)

    gz = np.linspace(-1, 1, pd) ** 2
    gy = np.linspace(-1, 1, ph) ** 2
    gx = np.linspace(-1, 1, pw) ** 2
    zz, yy, xx = np.meshgrid(gz, gy, gx, indexing="ij")
    w = np.exp(-0.5 * (zz + yy + xx))
    w = w / w.sum()

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                patch = vol[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
                pred = model.predict(patch[None], verbose=0)[0, ..., 0]
                prob[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += pred * w
                count[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += w

    return prob / (count + 1e-8)


def evaluate_model(model, pairs, name, patch_size):
    rows = []
    for img_path, msk_path in tqdm(pairs, desc=f"Eval {name}"):
        img, msk, _ = preprocess_pair(img_path, msk_path, crop=False)
        orig_img = img.copy()
        gt = msk[..., 0]

        prob = sliding_window_predict(img, model, patch_size=patch_size)
        pred = (prob > 0.5).astype(np.uint8)

        dsc = dice_coef(
            tf.constant(gt[None, ..., None], dtype=tf.float32),
            tf.constant(prob[None, ..., None], dtype=tf.float32),
        ).numpy()

        sens = sensitivity(pred, gt)
        prec = precision(pred, gt)
        hd = hd95(pred, gt, list(TARGET_SPACING))

        case = os.path.basename(img_path).split(".nii")[0]
        rows.append(
            {
                "case": case,
                "Dice": float(dsc),
                "HD95_mm": float(hd),
                "Sensitivity": float(sens),
                "Precision": float(prec),
            }
        )

        np.savez(
            os.path.join(PRED_DIR, f"{name}_{case}.npz"), pred=pred, prob=prob
        )

        z = orig_img.shape[0] // 2
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img[z, ..., 0], cmap="gray")
        plt.contour(gt[z] > 0, colors="g", linewidths=1)
        plt.title("GT")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(orig_img[z, ..., 0], cmap="gray")
        plt.contour(pred[z] > 0, colors="r", linewidths=1)
        plt.title(name)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, f"{name}_{case}_overlay.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    df = pd.DataFrame(rows)
    df.loc[len(df)] = [
        "MEAN",
        df.Dice.mean(),
        df.HD95_mm.mean(),
        df.Sensitivity.mean(),
        df.Precision.mean(),
    ]
    df.to_csv(os.path.join(RESULTS_DIR, f"{name}_metrics.csv"), index=False)
    return df


def main():
    test_pairs = match_pairs(TEST_IMG_DIR, TEST_MSK_DIR)

    model_base = build_unet()
    base_ckpt = os.path.join(CKPT_DIR, "best_baseline.weights.h5")
    if os.path.exists(base_ckpt):
        model_base.load_weights(base_ckpt)
    else:
        print(f"WARNING: baseline checkpoint not found: {base_ckpt}")
    df_base = evaluate_model(model_base, test_pairs, "baseline", patch_size=PATCH_SIZE)

    model_aug = build_unet()
    aug_ckpt = os.path.join(CKPT_DIR, "best_augmented.weights.h5")
    if os.path.exists(aug_ckpt):
        model_aug.load_weights(aug_ckpt)
    else:
        print(f"WARNING: augmented checkpoint not found: {aug_ckpt}")
    df_aug = evaluate_model(
        model_aug, test_pairs, "augmented", patch_size=PATCH_SIZE
    )

    print("\n=== BASELINE ===\n", df_base)
    print("\n=== AUGMENTED ===\n", df_aug)


if __name__ == "__main__":
    main()
