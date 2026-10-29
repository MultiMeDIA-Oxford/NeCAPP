"""Export reconstruction previews and NIfTI files without changing predictions."""

import numpy as np
from PIL import Image


def save_raw_projection_png(projection, path):
    """Save a grayscale preview; full-precision values remain in the raw .npy."""
    projection = np.asarray(projection)
    p_min, p_max = projection.min(), projection.max()
    if p_max > p_min:
        pixels = np.rint((projection - p_min) / (p_max - p_min) * 255)
    else:
        pixels = np.zeros_like(projection)
    Image.fromarray(pixels.astype(np.uint8)).save(path)


def save_fine_prediction_nifti(volume, data_config, path):
    """Use YAML/YML nVoxel/dVoxel and the existing centered reconstruction grid.

    Keep the .npy array axis order and values. The input geometry does not
    specify patient-space orientation; the affine uses reconstruction coordinates.
    """
    import nibabel as nib

    volume = np.asarray(volume, dtype=np.float32)
    shape = np.asarray(data_config["nVoxel"], dtype=np.int64)
    spacing = np.asarray(data_config["dVoxel"], dtype=np.float64)
    if shape.shape != (3,) or tuple(shape) != volume.shape:
        raise ValueError("fine_pred shape must match nVoxel in the input YAML/YML")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("dVoxel must contain three positive finite spacings in mm")

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(spacing)
    affine[:3, 3] = -0.5 * (shape - 1) * spacing
    image = nib.Nifti1Image(volume, affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, code=2)
    image.set_sform(affine, code=2)
    nib.save(image, str(path))


def save_binary_prediction_and_dice(volume, ground_truth_path, data_config, output_dir, threshold=0.35):
    """Threshold only the prediction and compare with the input binary 3D mask."""
    import json
    from pathlib import Path

    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("dice_threshold must be finite and between 0 and 1")
    volume = np.asarray(volume)
    ground_truth = np.load(ground_truth_path)
    if volume.ndim != 3 or ground_truth.shape != volume.shape:
        raise ValueError("Dice requires prediction and input binary 3D to have identical shapes")
    if not np.all(np.isfinite(volume)) or not np.all(np.isfinite(ground_truth)):
        raise ValueError("Dice requires finite prediction and ground-truth values")

    prediction_binary = volume >= threshold
    ground_truth_binary = ground_truth != 0
    intersection = int(np.count_nonzero(prediction_binary & ground_truth_binary))
    prediction_count = int(np.count_nonzero(prediction_binary))
    ground_truth_count = int(np.count_nonzero(ground_truth_binary))
    denominator = prediction_count + ground_truth_count
    dice = 2.0 * intersection / denominator if denominator else 1.0

    output_dir = Path(output_dir)
    binary = prediction_binary.astype(np.uint8)
    np.save(output_dir / "fine_pred_binary.npy", binary)
    save_fine_prediction_nifti(binary, data_config, output_dir / "fine_pred_binary.nii")
    metrics = {
        "threshold": threshold,
        "threshold_comparison": ">=",
        "dice": dice,
        "intersection_voxels": intersection,
        "prediction_foreground_voxels": prediction_count,
        "ground_truth_foreground_voxels": ground_truth_count,
        "ground_truth_path": str(ground_truth_path),
    }
    with (output_dir / "fine_pred_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(f"[Dice] threshold={threshold:g}, Dice={dice:.6f}")
    return metrics
