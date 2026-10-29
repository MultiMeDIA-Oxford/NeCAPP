"""Train the two-stage coronary reconstruction model for each input case."""

import torch
import argparse

from src.config.configloading import load_config
from src.render import run_network
from src.trainer import Trainer

import os
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def config_parser():
    """Define command-line options without changing their existing defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/CCTA.yaml", help="configs file path")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="存放 GT 数据的目录（或单个 .npy 文件路径），会覆盖 dataconfig 里的 GT",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help="保存日志/ckpt/eval 的根目录，不同 case 会在下面建子文件夹",
    )
    parser.add_argument(
        "--save_root2",
        type=str,
        default=None,
        help="保存日志/ckpt/eval 的根目录，不同 case 会在下面建子文件夹",
    )
    parser.add_argument(
        "--proj_one",
        type=str,
        default=None,
        help="First projection .npy; required in 2d mode, optional in 3d mode",
    )

    parser.add_argument(
        "--proj_two",
        type=str,
        default=None,
        help="Second projection .npy; required in 2d mode, optional in 3d mode",
    )

    parser.add_argument(
        "--dataconfig",
        type=str,
        default=None,
        help="Path to data/config.yml (projection geometry etc.). Overrides cfg['exp']['dataconfig']",
    )

    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained model (.pth). If set and resume=False, load pretrained weights.",
    )

    parser.add_argument(
        "--freeze_mode",
        type=str,
        default=None,
        choices=["encoder", "mlp", "encoder+mlp", "coarse"],
        help="Freeze mode for finetuning.",
    )
    parser.add_argument(
        "--auto_stage",
        action="store_true",
        help="Automatically run Stage1 (coarse) then Stage2 (fine)",
    )

    parser.add_argument(
        "--input_mode",
        type=str,
        default="3d",
        choices=["2d", "3d"],
        help="Input mode: 3d (project from GT) or 2d (load projection images)",
    )

    parser.add_argument(
        "--dice_threshold",
        type=float,
        default=0.35,
        help="Prediction binarization threshold for final 3D-mode Dice (default: 0.35)",
    )

    return parser


parser = config_parser()
args = parser.parse_args()
if args.input_mode == "2d" and (args.proj_one is None or args.proj_two is None):
    parser.error("--input_mode 2d requires both --proj_one and --proj_two")
if not 0.0 <= args.dice_threshold <= 1.0:
    parser.error("--dice_threshold must be between 0 and 1")

cfg = load_config(args.config)
cfg.setdefault("train", {})["dice_threshold"] = args.dice_threshold

if args.dataconfig is not None:
    cfg.setdefault("exp", {})
    cfg["exp"]["dataconfig"] = args.dataconfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicTrainer(Trainer):
    """Provide the entry point-specific reconstruction objective."""

    def __init__(
        self,
        gt_path=None,
        case_name=None,
        logs_root=None,
        proj_one_path=None,
        proj_two_path=None,
        pretrained_path=None,
        freeze_mode=None,
        input_mode=None,
    ):

        super().__init__(
            cfg,
            device,
            gt_path=gt_path,
            case_name=case_name,
            logs_root=logs_root,
            proj_one_path=proj_one_path,
            proj_two_path=proj_two_path,
            pretrained_path=pretrained_path,
            freeze_mode=freeze_mode,
            input_mode=input_mode,
        )

        self.l2_loss = torch.nn.MSELoss(reduction="mean")

    def bimodal_density_loss(self, sigma):
        """Average sigma * (1 - sigma) over the supplied reconstruction region."""
        return torch.mean(sigma * (1.0 - sigma))

    def ray_tv_loss_directional(self, vol, ray_dir):
        """Weight axis-wise absolute differences by the normalized absolute ray components."""
        dx = torch.abs(vol[1:, :, :] - vol[:-1, :, :]).mean()
        dy = torch.abs(vol[:, 1:, :] - vol[:, :-1, :]).mean()
        dz = torch.abs(vol[:, :, 1:] - vol[:, :, :-1]).mean()

        weights = torch.abs(ray_dir)
        weights = weights / (weights.sum() + 1e-8)

        loss = weights[0] * dx + weights[1] * dy + weights[2] * dz

        return loss

    def compute_loss(self, data, global_step, idx_epoch):
        """Compute the existing stage-specific objective for the current input projections."""
        loss = {"loss": 0.0}
        projs = data.projs

        # Query the full grid in Stage 1 and active-region coordinates in Stage 2.
        if self.stage == "coarse":
            image_pred = run_network(self.voxels, self.net, self.netchunk).squeeze()

        else:
            pred_masked = run_network(self.voxels_masked, self.net, self.netchunk).squeeze()

            image_pred = torch.zeros(self.mask.numel(), device=pred_masked.device)

            image_pred[self.mask_idx] = pred_masked

            image_pred = image_pred.view(self.mask.shape)

        train_output = image_pred[None, None, ...]

        # Compare rendered projections with the observed views.
        train_projs_list = [proj.forward_project(train_output) for proj in self.ct_projectors]
        train_projs = torch.cat(train_projs_list, 1)

        loss_proj = self.l2_loss(train_projs, projs)

        # Apply the existing stage-specific regularization weights.
        if self.stage == "fine":
            loss_bin = self.bimodal_density_loss(pred_masked)

            loss_ray_tv = torch.zeros((), device=image_pred.device)
            for ray_dir in self.ray_dirs:
                loss_ray_tv = loss_ray_tv + self.ray_tv_loss_directional(image_pred, ray_dir)
            loss_ray_tv = loss_ray_tv / len(self.ray_dirs)

            lambda_bin = 2

            lambda_ray = 7.5

            loss_total = loss_proj + lambda_ray * loss_ray_tv + lambda_bin * loss_bin

            loss["loss_proj"] = loss_proj
            loss["loss_bin"] = loss_bin
            loss["loss_ray"] = loss_ray_tv
            loss["loss"] = loss_total

        else:
            loss_bin = self.bimodal_density_loss(image_pred)

            lambda_bin = 5

            loss_total = loss_proj + lambda_bin * loss_bin

            loss["loss_proj"] = loss_proj

            loss["loss"] = loss_total

        return loss


if __name__ == "__main__":
    dataconfig_path = cfg["exp"]["dataconfig"]

    with open(dataconfig_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    proj_one_arg = args.proj_one
    proj_two_arg = args.proj_two

    if args.input_mode == "3d":
        # Build a case directly from the 3D input; no projection files are needed.
        input_3d_path = args.data_dir if args.data_dir is not None else data_cfg.get("GT")
        if input_3d_path is None:
            parser.error("3d mode requires --data_dir or GT in the geometry YAML/YML")
        if proj_one_arg is not None and proj_two_arg is not None:
            # Preserve case naming for existing commands that supply projection paths.
            case_name = os.path.splitext(os.path.basename(proj_one_arg))[0]
        else:
            case_name = os.path.splitext(os.path.basename(input_3d_path))[0]
        file_pairs = [(proj_one_arg, proj_two_arg, case_name)]
    else:
        # Argument validation above guarantees two explicitly provided 2D inputs.
        case_name = os.path.splitext(os.path.basename(proj_one_arg))[0]
        file_pairs = [(proj_one_arg, proj_two_arg, case_name)]

    for proj_one_path, proj_two_path, case_name in file_pairs:
        # Reuse Stage 1 weights and its spatial support for Stage 2.
        if args.auto_stage:
            cfg["train"]["stage"] = "coarse"
            cfg["train"].pop("mask_path", None)

            trainer = BasicTrainer(
                gt_path=args.data_dir,
                case_name=case_name,
                logs_root=args.save_root,
                proj_one_path=proj_one_path,
                proj_two_path=proj_two_path,
                pretrained_path=cfg["train"].get("pretrained_path"),
                freeze_mode=cfg["train"].get("freeze_mode"),
                input_mode=args.input_mode,
            )

            trainer.start()

            mask_path = os.path.join(trainer.expdir, "coarse_mask.npy")

            if not os.path.exists(mask_path):
                raise RuntimeError(f"Stage1 finished but mask not found: {mask_path}")

            cfg["train"]["stage"] = "fine"

            trainer_fine = BasicTrainer(
                gt_path=args.data_dir,
                case_name=case_name,
                logs_root=args.save_root,
                proj_one_path=proj_one_path,
                proj_two_path=proj_two_path,
                pretrained_path=os.path.join(trainer.expdir, "pretrained_nerf.pth"),
                freeze_mode=cfg["train"].get("freeze_mode"),
                input_mode=args.input_mode,
            )

            trainer_fine.start()

        else:
            trainer = BasicTrainer(
                gt_path=args.data_dir,
                case_name=case_name,
                logs_root=args.save_root,
                proj_one_path=proj_one_path,
                proj_two_path=proj_two_path,
                pretrained_path=cfg["train"].get("pretrained_path"),
                freeze_mode=cfg["train"].get("freeze_mode"),
                input_mode=args.input_mode,
            )

            trainer.start()
