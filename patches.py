import numpy as np
from ..config import PATCH_SIZE, FG_SAMPLE_PROB


def sample_patch_coords(shape, patch, center=None):
    D, H, W = shape[:3]
    pd, ph, pw = patch
    if center is None:
        cz = np.random.randint(pd // 2, D - pd // 2 + 1)
        cy = np.random.randint(ph // 2, H - ph // 2 + 1)
        cx = np.random.randint(pw // 2, W - pw // 2 + 1)
    else:
        cz, cy, cx = center
        cz = np.clip(cz, pd // 2, D - pd // 2)
        cy = np.clip(cy, ph // 2, H - ph // 2)
        cx = np.clip(cx, pw // 2, W - pw // 2)
    return (
        slice(cz - pd // 2, cz + pd // 2),
        slice(cy - ph // 2, cy + ph // 2),
        slice(cx - pw // 2, cx + pw // 2),
    )


def extract_patch(img, msk, patch_size=PATCH_SIZE, fg_prob=FG_SAMPLE_PROB):
    msk_bool = msk[..., 0] > 0
    if np.random.rand() < fg_prob and msk_bool.sum() > 0:
        idx = np.argwhere(msk_bool)
        center = idx[np.random.randint(len(idx))]
    else:
        center = None
    z, y, x = sample_patch_coords(img.shape, patch_size, center)
    return img[z, y, x], msk[z, y, x]
