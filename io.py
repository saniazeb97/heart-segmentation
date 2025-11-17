import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def match_pairs(img_dir, msk_dir):
    """Match image/mask NIfTI files by filename stem."""
    imgs = sorted(
        [
            p
            for p in glob.glob(os.path.join(img_dir, "*"))
            if p.endswith((".nii", ".nii.gz"))
        ]
    )
    pairs = []
    for ip in imgs:
        stem = os.path.basename(ip).split(".nii")[0]
        candidates = [
            p
            for p in glob.glob(os.path.join(msk_dir, stem + "*"))
            if p.endswith((".nii", ".nii.gz"))
        ]
        if not candidates:
            raise FileNotFoundError(f"Mask not found for {ip}")
        pairs.append((ip, candidates[0]))
    return pairs


def load_any(path):
    """Load a NIfTI and return array + spacing."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)  
    arr = img.get_fdata().astype(np.float32)
    spacing = img.header.get_zooms()[:3]
    return arr, spacing


def resample_sitk(vol_np, spacing_in, spacing_out, is_label=False):
    """Resample a 3D volume with SimpleITK to target spacing."""
    spacing_in = [float(x) for x in spacing_in]
    spacing_out = [float(x) for x in spacing_out]

    img = sitk.GetImageFromArray(vol_np)
    img.SetSpacing(spacing_in[::-1])  # (x,y,z)

    in_size = np.array(vol_np.shape[::-1])  # (W,H,D)
    in_sp = np.array(spacing_in[::-1])
    out_sp = np.array(spacing_out[::-1])
    out_size = np.round(in_size * in_sp / out_sp).astype(int).tolist()

    res = sitk.ResampleImageFilter()
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    res.SetOutputSpacing(spacing_out[::-1])
    res.SetSize(out_size)
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    out = res.Execute(img)
    return sitk.GetArrayFromImage(out)
