import numpy as np
from .io import load_any, resample_sitk
from ..config import TARGET_SPACING, PATCH_SIZE


def zscore_normalize(vol):
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol = np.clip(vol, lo, hi)
    mean, std = vol.mean(), vol.std()
    return (vol - mean) / (std + 1e-8)


def pad_to_min_size(vol, target=PATCH_SIZE):
    if vol.ndim == 4:
        vol_3d = vol[..., 0]
    else:
        vol_3d = vol

    d, h, w = vol_3d.shape
    td, th, tw = target

    new_d = max(td, d)
    new_h = max(th, h)
    new_w = max(tw, w)

    new_d = ((new_d + 31) // 32) * 32
    new_h = ((new_h + 31) // 32) * 32
    new_w = ((new_w + 31) // 32) * 32

    pd = new_d - d
    ph = new_h - h
    pw = new_w - w

    if pd or ph or pw:
        vol_3d = np.pad(
            vol_3d,
            (
                (pd // 2, pd - pd // 2),
                (ph // 2, ph - ph // 2),
                (pw // 2, pw - pw // 2),
            ),
            mode="constant",
        )

    if vol.ndim == 4:
        vol_3d = vol_3d[..., None]
    return vol_3d


def _preprocess_pair_impl(
    img_path, msk_path, target_spacing=TARGET_SPACING, crop=False, return_spacing=False
):
    img, sp_img = load_any(img_path)
    msk, sp_msk = load_any(msk_path)

    sp_img = [float(x) for x in sp_img]
    sp_msk = [float(x) for x in sp_msk]

    assert np.allclose(sp_img, sp_msk, atol=1e-3), "Spacing mismatch!"

    if target_spacing is not None:
        img = resample_sitk(img, sp_img, target_spacing, is_label=False)
        msk = resample_sitk(msk, sp_msk, target_spacing, is_label=True)

    img = zscore_normalize(img)
    msk = (msk > 0.5).astype(np.uint8)

    if crop:
        d, h, w = img.shape
        s = PATCH_SIZE
        if d < s[0] or h < s[1] or w < s[2]:
            img = pad_to_min_size(img, PATCH_SIZE)
            msk = pad_to_min_size(msk, PATCH_SIZE)
            d, h, w = img.shape
        img = img[(d - s[0]) // 2 : (d + s[0]) // 2,
                  (h - s[1]) // 2 : (h + s[1]) // 2,
                  (w - s[2]) // 2 : (w + s[2]) // 2]
        msk = msk[(d - s[0]) // 2 : (d + s[0]) // 2,
                  (h - s[1]) // 2 : (h + s[1]) // 2,
                  (w - s[2]) // 2 : (w + s[2]) // 2]
    else:
        img = pad_to_min_size(img, PATCH_SIZE)
        msk = pad_to_min_size(msk, PATCH_SIZE)

    img = img[..., None].astype(np.float32)
    msk = msk[..., None].astype(np.uint8)

    if return_spacing:
        return img, msk, np.array(sp_img, dtype=np.float64)
    else:
        return img, msk


def preprocess_pair(img_path, msk_path, target_spacing=TARGET_SPACING, crop=False):
    return _preprocess_pair_impl(
        img_path, msk_path, target_spacing, crop, return_spacing=True
    )


def preprocess_pair_train(
    img_path, msk_path, target_spacing=TARGET_SPACING, crop=False
):
    img, msk, _ = _preprocess_pair_impl(
        img_path, msk_path, target_spacing, crop, return_spacing=True
    )
    return img, msk
