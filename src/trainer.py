"""Manage per-case geometry, staged optimization, and reconstruction outputs."""

import os
import os.path as osp
import json
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    from tensorboardX import SummaryWriter


from tqdm import tqdm
from shutil import copyfile
import numpy as np
from src.output_utils import (
    save_raw_projection_png,
    save_fine_prediction_nifti,
    save_binary_prediction_and_dice,
)
import yaml

from .dataset import TIGREDataset as Dataset
from .network import get_network
from .encoder import get_encoder
from src.render import run_network


from src.render.ct_geometry_projector_tigre import ConeBeam3DProjector

from src.encoder.hashencoder.hashgrid import HashEncoder


from pathlib import Path


class Trainer:
    """Coordinate per-case reconstruction while retaining the existing training behavior."""

    def __init__(
        self,
        cfg,
        device="cuda",
        gt_path=None,
        case_name=None,
        logs_root=None,
        proj_one_path=None,
        proj_two_path=None,
        pretrained_path=None,
        freeze_mode=None,
        input_mode="2d",
    ):

        self.global_step = 0
        self.conf = cfg
        self.n_fine = cfg["render"]["n_fine"]

        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]

        if logs_root is None:
            self.logs_root = Path(r"/users/catz0894/code/neca/NeCA_ORI/logs/CCTA")
        else:
            self.logs_root = Path(logs_root)

        self.expdir = self.logs_root

        self.proj_one_path = proj_one_path
        self.proj_two_path = proj_two_path

        self.input_mode = input_mode

        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.expdir, exist_ok=True)
        os.makedirs(self.evaldir, exist_ok=True)

        from collections import defaultdict

        self.train_losses = defaultdict(list)

        # Select the stage without changing its configured epoch budget.
        self.stage = cfg["train"].get("stage", "coarse")

        if self.stage == "coarse":
            self.epochs = cfg["train"].get("epoch_coarse", cfg["train"]["epoch_coarse"])
        elif self.stage == "fine":
            self.epochs = cfg["train"].get("epoch_fine", cfg["train"]["epoch_fine"])
        else:
            self.epochs = cfg["train"]["epoch"]

        print(f"[Stage] mode={self.input_mode}, stage={self.stage}, epochs={self.epochs}")

        configPath = cfg["exp"]["dataconfig"]
        with open(configPath, "r") as handle:
            data = yaml.safe_load(handle)

        if gt_path is not None:
            data["GT"] = gt_path

        if case_name is None:
            gt_basename = os.path.basename(data["GT"])
            case_name = os.path.splitext(gt_basename)[0]
        self.case_name = case_name

        self.expdir = self.logs_root / self.case_name
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.expdir, exist_ok=True)
        os.makedirs(self.evaldir, exist_ok=True)

        dsd = data["DSD"]
        dso = data["DSO"]

        dde = [a - b for a, b in zip(dsd, dso)]

        proj_size = np.array(data["nDetector"])
        proj_reso = np.array(data["dDetector"])

        image_size = np.array(data["nVoxel"])
        image_reso = np.array(data["dVoxel"])

        if "projection_angles" in data:
            proj_angles_list = data["projection_angles"]
        elif "projection_angles_deg" in data:
            proj_angles_list = data["projection_angles_deg"]
        elif "projection_angle_list" in data:
            proj_angles_list = data["projection_angle_list"]
        else:
            proj_angles_list = [data["first_projection_angle"], data["second_projection_angle"]]

        if not isinstance(proj_angles_list, (list, tuple)) or len(proj_angles_list) == 0:
            raise ValueError("projection_angles 必须是非空列表，例如 [[a0,a1],[b0,b1], ...]")
        if not isinstance(proj_angles_list[0], (list, tuple)):
            raise ValueError("projection_angles 的元素必须是 [angle0, angle1] 形式的列表/元组")

        proj_angles_list = [[float(p[0]), float(p[1])] for p in proj_angles_list]
        n_views = len(proj_angles_list)

        def _as_list(x, n, name):
            """Expand a shared geometry value or validate a per-view list."""
            if isinstance(x, np.ndarray):
                x = x.tolist()
            if isinstance(x, (list, tuple)):
                if len(x) == n:
                    return list(x)
                if len(x) == 1:
                    return list(x) * n
                raise ValueError(f"{name} 的长度({len(x)})与 projection_angles 的数量({n})不一致")
            return [x] * n

        dso_list = _as_list(dso, n_views, "DSO")
        dde_list = _as_list(dde, n_views, "DDE")
        dsd_list = _as_list(dsd, n_views, "DSD")

        # Build one projector and main-ray direction for each configured view.
        self.ct_projectors = []
        for i, euler_deg in enumerate(proj_angles_list):
            self.ct_projectors.append(
                ConeBeam3DProjector(
                    image_size=image_size,
                    image_reso=image_reso,
                    euler_deg=euler_deg,
                    proj_size=proj_size,
                    proj_reso=proj_reso,
                    dde=dde_list[i],
                    dso=dso_list[i],
                    dsd=dsd_list[i],
                )
            )

        self.ct_projector_first = self.ct_projectors[0]
        self.ct_projector_second = self.ct_projectors[1] if n_views > 1 else None

        self.ray_dirs = []
        for proj in self.ct_projectors:
            ray_dir = proj.get_main_ray_direction().to(device)
            self.ray_dirs.append(ray_dir)

        if self.input_mode == "3d":
            phantom_path = data["GT"]
            phantom = np.load(phantom_path)
            phantom = torch.tensor(phantom, dtype=torch.float32)[None, ...]

            train_projs_list = [proj.forward_project(phantom) for proj in self.ct_projectors]

            data["projections"] = torch.cat(train_projs_list, 1)

        else:
            if self.proj_one_path is None or self.proj_two_path is None:
                raise RuntimeError("2D mode requires proj_one_path and proj_two_path")

            p1 = np.load(self.proj_one_path).astype(np.float32)
            p2 = np.load(self.proj_two_path).astype(np.float32)

            if p1.shape != p2.shape:
                raise ValueError("Projection shapes do not match")

            projs_np = np.stack([p1, p2], axis=0)
            projs_np = projs_np[None, ...]

            data["projections"] = torch.from_numpy(projs_np).to(device)

            train_projs_list = [
                torch.from_numpy(p1[None, None, ...]).to(device),
                torch.from_numpy(p2[None, None, ...]).to(device),
            ]

        import numpy as _np

        save_dir = osp.join(str(self.expdir), "projections_binary")
        os.makedirs(save_dir, exist_ok=True)

        def tensor_to_np(t):
            """Convert a projection tensor or array to a NumPy array."""
            if hasattr(t, "detach"):
                return t.detach().cpu().numpy()
            return _np.asarray(t)

        def squeeze_proj(p, name="proj"):
            """Remove singleton dimensions and require a two-dimensional projection."""
            p_s = _np.squeeze(p)
            if p_s.ndim != 2:
                raise RuntimeError(f"{name} squeezed has unexpected shape {p_s.shape}")
            return p_s

        for vi, proj_t in enumerate(train_projs_list):
            p = squeeze_proj(tensor_to_np(proj_t), f"proj{vi}")

            raw_path = osp.join(save_dir, f"{self.case_name}_proj{vi + 1}_raw.npy")
            _np.save(raw_path, p)

            png_path = osp.join(save_dir, f"{self.case_name}_proj{vi + 1}_raw.png")
            save_raw_projection_png(p, png_path)

        print(f"[Info] Saved {len(train_projs_list)} projections (raw NPY/PNG) to {save_dir}")

        self.dataconfig = data
        self.train_dset = Dataset(data, device)
        self.voxels = self.train_dset.voxels

        self.mask = None
        self.mask_idx = None
        self.voxels_masked = None

        # Load the active-region support produced by the coarse stage.
        if self.stage == "fine":
            mask_path = cfg["train"].get("mask_path", None)

            if mask_path is None:
                default_mask = osp.join(self.expdir, "coarse_mask.npy")
                if not osp.exists(default_mask):
                    raise ValueError(
                        "Stage fine requires a mask, but coarse_mask.npy not found. "
                        "Please run Stage1 first."
                    )
                mask_path = default_mask
                print(f"[AutoMask] Using coarse mask from {mask_path}")

            mask_np = np.load(mask_path)

            import scipy.ndimage as ndi

            mask_np = ndi.binary_dilation(
                mask_np, structure=np.ones((3, 3, 3), dtype=bool), iterations=1
            )

            self.mask = torch.from_numpy(mask_np).to(device).bool()

            self.mask_idx = self.mask.reshape(-1)

            if self.voxels.ndim == 4:
                voxels_flat = self.voxels.reshape(-1, self.voxels.shape[-1])
            else:
                voxels_flat = self.voxels

            self.voxels_masked = voxels_flat[self.mask_idx]

            total_voxels = self.mask.numel()
            print(f"[Stage2] masked voxels: {self.voxels_masked.shape[0]} / {total_voxels}")

        network = get_network(cfg["network"]["net_type"])
        net_type = cfg["network"].pop("net_type", None)
        encoder = get_encoder(**cfg["encoder"])

        if isinstance(encoder, HashEncoder):
            if self.stage == "coarse":
                encoder.active_levels = 4
            else:
                encoder.active_levels = encoder.num_levels

        # Keep the existing model architecture and checkpoint parameter names.
        self.net = network(encoder, **cfg["network"]).to(device)
        self.grad_vars = list(self.net.parameters())
        self.net_fine = None
        if self.n_fine > 0:
            self.net_fine = network(encoder, **cfg["network"]).to(device)
            self.grad_vars += list(self.net_fine.parameters())
        cfg["network"]["net_type"] = net_type

        self.pretrained_path = pretrained_path
        self.freeze_mode = freeze_mode

        if self.pretrained_path is not None and not cfg["train"]["resume"]:
            print(f"[LOAD] loading pretrained model from {self.pretrained_path}")
            ckpt = torch.load(self.pretrained_path, map_location=device)

            self.net.load_state_dict(ckpt["network"], strict=False)

            if self.n_fine > 0 and ckpt.get("network_fine") is not None:
                self.net_fine.load_state_dict(ckpt["network_fine"], strict=False)

        self.freeze_network(self.freeze_mode)

        trainable_params = filter(lambda p: p.requires_grad, self.grad_vars)

        if self.stage == "fine":
            self.optimizer = torch.optim.Adam(
                trainable_params, lr=cfg["train"]["lrate_F"], betas=(0.9, 0.999)
            )

            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=300, gamma=0.5
            )
        else:
            self.optimizer = torch.optim.Adam(
                trainable_params, lr=cfg["train"]["lrate_C"], betas=(0.9, 0.999)
            )

            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=9000, gamma=0.2
            )

        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start
            self.net.load_state_dict(ckpt["network"])
            if self.n_fine > 0:
                self.net_fine.load_state_dict(ckpt["network_fine"])

        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        """Format the configuration for experiment logging."""
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def train_step(self, data, global_step, idx_epoch):
        """Clear gradients, backpropagate the objective, and update model parameters."""
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, global_step, idx_epoch)
        loss["loss"].backward()
        self.optimizer.step()
        return loss

    def compute_loss(self, data, global_step, idx_epoch):
        """Compute the existing stage-specific objective for the current input projections."""
        raise NotImplementedError()

    def save_loss_curve(self):
        """Write the existing loss history and any configured plot outputs."""
        if len(self.train_losses) == 0:
            return

        loss_keys = list(self.train_losses.keys())

        n_epoch = len(self.train_losses[loss_keys[0]])
        epochs = np.arange(1, n_epoch + 1, dtype=np.int32)

        csv_path = osp.join(self.expdir, "loss_data.csv")
        header = "epoch," + ",".join(loss_keys)

        rows = []
        for i in range(n_epoch):
            row = [epochs[i]]
            for k in loss_keys:
                row.append(self.train_losses[k][i])
            rows.append(row)

        rows = np.array(rows, dtype=np.float32)
        np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")

    def save_pretrained(self, name="pretrained"):
        """Save network parameters in the existing checkpoint format."""
        save_path = osp.join(self.expdir, f"{name}.pth")
        torch.save(
            {
                "network": self.net.state_dict(),
                "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
            },
            save_path,
        )
        print(f"[PRETRAIN] model saved to {save_path}")

    def freeze_network(self, mode):
        """Apply the requested parameter-freezing mode."""
        if mode is not None:
            print(f"[FREEZE] mode = {mode}")

        def freeze(module):
            """Disable gradient updates for every parameter in the supplied module."""
            for p in module.parameters():
                p.requires_grad = False

        if mode is None:
            return

        if mode == "encoder":
            freeze(self.net.encoder)
            if self.net_fine is not None:
                freeze(self.net_fine.encoder)

        elif mode == "mlp":
            freeze(self.net)

        elif mode == "encoder+mlp":
            freeze(self.net)
            if self.net_fine is not None:
                freeze(self.net_fine)

        elif mode == "coarse":
            freeze(self.net)

        else:
            raise ValueError(f"Unknown freeze mode: {mode}")

    def start(self):
        """Run the existing evaluation, optimization, scheduling, and final export sequence."""
        iter_per_epoch = 1
        pbar = tqdm(total=iter_per_epoch * self.epochs, leave=True)

        best_loss = float("inf")
        best_epoch = -1

        early_stop_patience = 40
        early_stop_delta = 1e-6
        early_stop_counter = 0

        if self.epoch_start > 0:
            pbar.update(self.epoch_start * iter_per_epoch)

        # Preserve evaluation, training, curriculum, and scheduler order.
        for idx_epoch in range(self.epoch_start, self.epochs):
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and self.i_eval > 0:
                self.net.eval()
                with torch.no_grad():
                    image_pred = run_network(
                        self.voxels,
                        self.net_fine if self.net_fine is not None else self.net,
                        self.netchunk,
                    )
                    image_pred = (image_pred.squeeze()).detach().cpu().numpy()

                    out_name = f"{self.case_name}_epoch{idx_epoch}.npy"
                    np.save(osp.join(self.evaldir, out_name), image_pred)

            self.global_step += 1

            self.net.train()
            loss_train = self.train_step(
                self.train_dset, global_step=self.global_step, idx_epoch=idx_epoch
            )

            for k, v in loss_train.items():
                if torch.is_tensor(v):
                    self.train_losses[k].append(float(v.item()))

            if self.stage == "fine" and isinstance(self.net.encoder, HashEncoder):
                max_levels = self.net.encoder.num_levels
                warmup_epochs = 50

                cur_levels = min(max_levels, 4 + idx_epoch // warmup_epochs)

                if cur_levels != self.net.encoder.active_levels:
                    self.net.encoder.active_levels = cur_levels

                    print(f"[Curriculum] Epoch {idx_epoch}: active hash levels = {cur_levels}")

            if self.stage == "fine":
                current_loss = loss_train["loss"].item()
                if current_loss < best_loss - early_stop_delta:
                    best_loss = current_loss
                    best_epoch = idx_epoch
                    early_stop_counter = 0

                else:
                    early_stop_counter += 1

                if early_stop_counter >= early_stop_patience:
                    print(
                        f"[EarlyStop] Stage2 stops at epoch {idx_epoch}. "
                        f"Best epoch was {best_epoch} with loss {best_loss:.6f}"
                    )
                    break

            pbar.set_description(
                f"epoch={idx_epoch}/{self.epochs}, {loss_train['loss']}, lr={self.optimizer.param_groups[0]['lr']:.3g}"
            )
            pbar.update(1)

            if (
                (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs)
                and self.i_save > 0
                and idx_epoch > 0
            ):
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            self.writer.add_scalar(
                "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
            )
            self.lr_scheduler.step()

        self.save_pretrained("pretrained_nerf")

        with torch.no_grad():
            if self.stage == "coarse":
                image_pred = run_network(self.voxels, self.net, self.netchunk).squeeze()

                image_pred_np = image_pred.cpu().numpy()

                pred_path = osp.join(self.expdir, "coarse_pred.npy")
                np.save(pred_path, image_pred_np)

                # Retain the configured top percentage of coarse occupancy predictions.
                top_percent = float(self.conf["train"].get("coarse_mask_top_percent", 40))
                if not 0.0 < top_percent <= 100.0:
                    raise ValueError("train.coarse_mask_top_percent must be in (0, 100]")

                thresh = np.percentile(image_pred_np, 100 - top_percent)

                coarse_mask = (image_pred_np >= thresh).astype(np.uint8)

                print(
                    f"[Stage1] Percentile mask: top {top_percent}% | "
                    f"threshold={thresh:.6f} | "
                    f"voxels={coarse_mask.sum()}"
                )

                mask_path = osp.join(self.expdir, "coarse_mask.npy")
                np.save(mask_path, coarse_mask)

                print(f"[Stage1] coarse_pred saved to {pred_path}")
                print(f"[Stage1] coarse_mask saved to {mask_path}")

            else:
                pred_masked = run_network(self.voxels_masked, self.net, self.netchunk).squeeze()

                image_pred = torch.zeros(self.mask.numel(), device=pred_masked.device)

                image_pred[self.mask_idx] = pred_masked

                image_pred = image_pred.view(self.mask.shape)

                image_pred_np = image_pred.cpu().numpy()

                out_path = osp.join(self.expdir, "fine_pred.npy")
                np.save(out_path, image_pred_np)
                save_fine_prediction_nifti(
                    image_pred_np, self.dataconfig, osp.join(self.expdir, "fine_pred.nii")
                )

                # Evaluate binary overlap only when an input 3D mask is available.
                if self.input_mode == "3d":
                    save_binary_prediction_and_dice(
                        image_pred_np,
                        self.dataconfig["GT"],
                        self.dataconfig,
                        self.expdir,
                        threshold=self.conf["train"].get("dice_threshold", 0.35),
                    )

                print(f"[Stage2] fine_pred.npy and fine_pred.nii saved in {self.expdir}")

        self.save_loss_curve()
        tqdm.write(f"[Stage] {self.stage} complete. Outputs: {self.expdir}")
