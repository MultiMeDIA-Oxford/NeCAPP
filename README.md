# NeCA++

Code for **Multi-Stage NeRF for Efficient 3D Coronary Artery Reconstruction from Two Narrow-Angle Angiographic Projections** ([paper](https://arxiv.org/abs/2609.15550)).

NeCA++ reconstructs coronary arteries from two angiographic projections. It optimizes a neural implicit representation per case, using coarse localization followed by progressive refinement. No pretrained model is required.

## Installation

Reference setup: **Python 3.10, PyTorch 2.8.0 with CUDA 12.8, and CERN TIGRE**. Both input modes require an NVIDIA GPU. Building the hash-grid extension requires CUDA development tools, a C++ compiler, and Ninja.

Follow [ENVIRONMENT.md](ENVIRONMENT.md) to create the environment, install [requirements.txt](requirements.txt), and build TIGRE from source. The hash-grid extension is compiled automatically on first import.

## Example data

The `data/` folder contains one generated right coronary artery (RCA) example:

| File | Description |
|---|---|
| `test.npy` | Binary 3D volume, shape `(80, 80, 80)` |
| `1_r1_proj1_raw_proj1_raw.npy` | First raw projection, shape `(512, 512)` |
| `1_r1_proj1_raw_proj2_raw.npy` | Second raw projection, shape `(512, 512)` |
| `2DRCAr1r3_80.yml` | Corresponding projection and reconstruction geometry |

## Run the example

Activate your environment and run **from the project root**. These commands use **Windows CMD** syntax; each `^` must be the last character on its line.

### 1. Start from the example 3D volume

Generate two projections from the binary input, reconstruct using projection supervision, and evaluate the final reconstruction against the input volume.

```bat
python train.py ^
  --config ./config/new_test.yaml ^
  --input_mode 3d ^
  --data_dir ./data/test.npy ^
  --dataconfig ./data/2DRCAr1r3_80.yml ^
  --save_root ./logs/demo_3d ^
  --auto_stage
```

`--proj_one` and `--proj_two` are not required in 3D mode.

### 2. Start from the two example projections

Reconstruct directly from two projections. Both projection arguments are required. This mode does not load 3D ground truth or compute Dice.

```bat
python train.py ^
  --config ./config/new_test.yaml ^
  --input_mode 2d ^
  --proj_one ./data/1_r1_proj1_raw_proj1_raw.npy ^
  --proj_two ./data/1_r1_proj1_raw_proj2_raw.npy ^
  --dataconfig ./data/2DRCAr1r3_80.yml ^
  --save_root ./logs/demo_2d ^
  --auto_stage
```

`--auto_stage` runs both stages, passing the coarse mask and weights to fine refinement.

## Configuration and outputs

Edit `config/new_test.yaml` to adjust training:

| Setting | Meaning |
|---|---|
| `train.epoch_coarse` / `train.epoch_fine` | Iteration budget for each stage |
| `train.coarse_mask_top_percent` | Percentage of highest coarse predictions retained before dilation; currently `15` |
| `render.netchunk` | Number of coordinates evaluated per network-query chunk |

For final binarization in **3D mode**, add `--dice_threshold 0.4` to change the default threshold of `0.35`. Predictions at or above the threshold are foreground.

Results are written under `--save_root/<case_name>/`. For the 3D example, this is `logs/demo_3d/test/`.

| Output | Description |
|---|---|
| `fine_pred.npy`, `fine_pred.nii` | Final continuous 3D reconstruction |
| `fine_pred_binary.npy`, `fine_pred_binary.nii` | Thresholded reconstruction; 3D mode only |
| `fine_pred_metrics.json` | Dice score and binarization threshold; 3D mode only |
| `coarse_pred.npy`, `coarse_mask.npy` | Intermediate coarse prediction and support mask |
| `projections_binary/` | Raw projection NPY files and grayscale PNG previews; despite its name, this folder contains no binary projections |

Checkpoints, loss history, and periodic predictions are also saved. NIfTI spacing comes from the geometry file; coordinates are centered on the reconstruction volume.

## Use your own data

For 2D mode, supply two **raw projection `.npy` arrays** in the geometry's view order. Use raw values rather than PNG previews or binary masks. For 3D mode, supply a binary 3D `.npy` volume.

Update detector/voxel dimensions, spacings, and angles in your YAML/YML and pass it with `--dataconfig`. Ensure `network.bound` covers the voxel-center coordinates. `--data_dir` overrides the geometry's `GT` path in 3D mode; 2D mode ignores that field.

## Citation

If you use this work, please cite:

```bibtex
@InProceedings{MenDey_Multistage_MICCAI2026,
        author = { Meng, Deyu AND Lashgari, Mojtaba AND Wang, Yiying AND Banerjee, Abhirup},
        title = { { Multi-stage NeRF for Efficient 3D Coronary Artery Reconstruction from Two Narrow-Angle Angiographic Projections } },
        booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
        year = {2026},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 16889},
        month = {September},
        page = {pending}
}
```
