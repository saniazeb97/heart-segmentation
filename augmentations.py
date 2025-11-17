import numpy as np
import albumentations as A
import cv2
import tensorflow as tf
from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure

from .config import PATCH_SIZE


GEOM_AUG_REPLAY = A.ReplayCompose(
    [
        A.RandomRotate90(p=0.30),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.03, 0.03),
            scale=(1.0, 1.0),
            rotate=(-5, 5),
            shear=0,
            interpolation=cv2.INTER_LINEAR,
            p=0.6,
        )],
    p=1.0,
)


def pad_or_crop_to_patch(vol, target=PATCH_SIZE):
    """Pad or center-crop a volume to the target patch size."""
    if vol.ndim == 4:
        arr = vol[..., 0]
    else:
        arr = vol

    d, h, w = arr.shape
    td, th, tw = target

    pd = max(0, td - d)
    ph = max(0, th - h)
    pw = max(0, tw - w)
    if pd or ph or pw:
        arr = np.pad(
            arr,
            (
                (pd // 2, pd - pd // 2),
                (ph // 2, ph - ph // 2),
                (pw // 2, pw - pw // 2),
            ),
            mode="constant",
        )

    d, h, w = arr.shape
    z0 = max(0, (d - td) // 2)
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    z1 = z0 + td
    y1 = y0 + th
    x1 = x0 + tw
    arr = arr[z0:z1, y0:y1, x0:x1]

    if vol.ndim == 4:
        arr = arr[..., None]
    return arr


def add_rician_noise(img, snr=20):
    signal = np.abs(img)
    sigma = signal.mean() / (snr + 1e-8)
    noise1 = np.random.normal(0, sigma, img.shape)
    noise2 = np.random.normal(0, sigma, img.shape)
    noisy = np.sqrt((img + noise1) ** 2 + noise2 ** 2)
    return noisy


def random_intensity(img):
    img = img.copy()

    if np.random.rand() < 0.20:
        r = np.random.rand()

        if r < 1 / 3:
            gamma = np.random.uniform(0.98, 1.02)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_norm = (img - img_min) / (img_max - img_min + 1e-8)
                img_norm = np.power(img_norm, gamma)
                img = img_norm * (img_max - img_min) + img_min

        elif r < 2 / 3:
            img = img * np.random.uniform(0.995, 1.005) + np.random.uniform(-0.005, 0.005)

        else:
             if np.random.rand() < 0.25:  
                snr = np.random.uniform(50, 80) 
                img = add_rician_noise(img, snr)

    img = np.clip(img, -3, 3)
    return (img - img.mean()) / (img.std() + 1e-8)


def keep_largest_cc(mask):
    lbl, n = label(mask > 0)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0 
    largest = np.argmax(sizes)
    return (lbl == largest).astype(mask.dtype)


def clean_binary_mask_3d(mask, do_open=True, do_close=True):
    mask = keep_largest_cc(mask.astype(np.uint8))
    struct = generate_binary_structure(3, 1)
    if do_close:
        mask = binary_closing(mask, structure=struct)
    if do_open:
        mask = binary_opening(mask, structure=struct)
    return mask.astype(np.uint8)


def augment_geometric_replay(img_np, msk_np):

    assert img_np.ndim == 4 and img_np.shape[-1] == 1
    assert msk_np.ndim == 4 and msk_np.shape[-1] == 1

    D = img_np.shape[0]
    base_img = img_np[0, ..., 0].astype(np.float32)
    base_msk = msk_np[0, ..., 0].astype(np.uint8)

    res = GEOM_AUG_REPLAY(image=base_img, mask=base_msk)
    replay_def = res["replay"]

    transformed_imgs = []
    transformed_msks = []

    for z in range(D):
        img_slice = img_np[z, ..., 0].astype(np.float32)
        msk_slice = msk_np[z, ..., 0].astype(np.uint8)

        out = A.ReplayCompose.replay(replay_def, image=img_slice, mask=msk_slice)
        transformed_imgs.append(out["image"])
        transformed_msks.append(out["mask"])

    new_img = np.stack(transformed_imgs, axis=0)[..., None].astype(np.float32)
    new_msk = np.stack(transformed_msks, axis=0)[..., None].astype(np.uint8)

    new_img = pad_or_crop_to_patch(new_img, PATCH_SIZE)
    new_msk = pad_or_crop_to_patch(new_msk, PATCH_SIZE)
    return new_img, new_msk


def augment_intensity_wrapper(img_np):
    out = img_np.copy()
    D = out.shape[0]
    out = random_intensity(out[..., 0])     
    out = out[..., None].astype(np.float32)
    return out

def augment_once(x, y):
    """Combined geometric + intensity + mask cleaning augmentation."""
    x = x.copy()
    y = y.copy()

    if np.random.rand() < 0.5:
        x, y = augment_geometric_replay(x, y)

    x = augment_intensity_wrapper(x)
    return x.astype(np.float32), y.astype(np.uint8)


def safe_tf_augment(img, msk):
    """Wrapper for use inside tf.data pipelines."""
    xi, yi = tf.numpy_function(augment_once, [img, msk], [tf.float32, tf.uint8])
    xi.set_shape(img.shape)
    yi.set_shape(msk.shape)
    return xi, tf.cast(yi, tf.float32)
