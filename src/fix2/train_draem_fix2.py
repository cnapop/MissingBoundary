#!/usr/bin/env python3
"""
fix2 双流训练 (BN→train/good, BD→MirrorEM 双流)
=================================================
fork 自 MirrorEM/train_draem_with_mirrorem.py，两处改动:
  1. `--init-random`：跳过 checkpoint 加载，随机初始化（与 baseline_200 公平对比）。
     随机初始化时应用与 DRAEM train_DRAEM.py 相同的 weights_init，保证从头训练
     的初始化分布与 baseline 一致。
  2. 保存为 DRAEM 兼容命名 `{base}_{cat}_.pckl` / `{base}_{cat}__seg.pckl`，
     使 test_DRAEM.py（base_model_name + obj_name）可直接评估。

额外（为与 baseline_200 严格公平）：加入 DRAEM 的 MultiStepLR
  ([epochs*0.8, epochs*0.9], gamma=0.2)，每 epoch 一次 lr 衰减；每 epoch 保存一次
  checkpoint（覆盖同名，最终文件即 epoch 200）。

用法:
  python src/train_draem_fix2.py \
    --manifest outputs/fix2/bottle/bd_manifest.csv \
    --normal-data-dir outputs/fix2/bottle/train_good_plus_bn \
    --anomaly-source-path /data/chenjiawen/DRAEM/datasets/dtd/images \
    --init-random --base-name DRAEM_test_0.0001_200_bs8 --category bottle \
    --output-dir outputs/fix2/bottle/checkpoints/fix2 \
    --epochs 200 --lr 0.0001 --mirror-weight 0.5 \
    --batch-size 4 --pre-batch-size 4 --draem-repo /data/chenjiawen/DRAEM \
    --device cuda:<空闲GPU>
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim


class SelectedSeaSDataset(Dataset):
    """DRAEM-format samples selected by the MB v3 closed-loop (fix2 manifest)."""

    def __init__(self, manifest, size=256):
        with open(manifest, newline="", encoding="utf-8-sig") as f:
            self.rows = list(csv.DictReader(f))
        self.size = size
        if not self.rows:
            raise ValueError("fix2 manifest contains no selected samples")

    def __len__(self):
        return len(self.rows)

    def _image(self, path):
        import cv2
        value = cv2.imread(path, cv2.IMREAD_COLOR)
        if value is None:
            raise FileNotFoundError(path)
        value = cv2.resize(value, (self.size, self.size)).astype(np.float32) / 255.0
        return np.transpose(value, (2, 0, 1))

    def __getitem__(self, index):
        row = self.rows[index]
        candidate = self._image(row["image"])
        target = self._image(row["reference"]) if row.get("reference") else candidate.copy()
        if row.get("mask"):
            import cv2
            mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(row["mask"])
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)[None]
        else:
            mask = np.zeros((1, self.size, self.size), np.float32)
        if row.get("type") == "normal":
            mask.fill(0)
            target = candidate.copy()  # preserve difficult but valid normal appearance
        return {"image": target, "augmented_image": candidate, "anomaly_mask": mask,
                "weight": np.float32(max(float(row.get("utility", 1.0)), 1e-3))}


def infinite_loader(loader):
    """Yield batches forever so unequal datasets contribute at every update."""
    while True:
        yield from loader


def load_state_dict(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def weights_init(m):
    """DRAEM 随机初始化（与 baseline_200 相同，保证公平）。"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def draem_loss(batch, rec, seg, mse, ssim, focal, device, use_utility=False):
    target = batch["image"].to(device)
    candidate = batch["augmented_image"].to(device)
    mask = batch["anomaly_mask"].to(device)

    restored = rec(candidate)
    probabilities = torch.softmax(seg(torch.cat((restored, candidate), dim=1)), dim=1)

    per_sample_l2 = mse(restored, target).flatten(1).mean(1)
    if use_utility:
        utility = batch["weight"].to(device)
        # Normalize within the mini-batch so utility changes relative sample
        # importance without silently changing the global MirrorEM loss scale.
        utility = utility / utility.mean().clamp_min(1e-6)
        l2_loss = (per_sample_l2 * utility).mean()
    else:
        l2_loss = per_sample_l2.mean()

    ssim_loss = ssim(restored, target)
    focal_loss = focal(probabilities, mask)
    total = l2_loss + ssim_loss + focal_loss
    return total, {
        "l2": l2_loss.detach(),
        "ssim": ssim_loss.detach(),
        "focal": focal_loss.detach(),
    }


def main():
    p = argparse.ArgumentParser(
        description="fix2: DRAEM dual-stream training (BN->train/good, BD->MirrorEM)"
    )
    p.add_argument("--manifest", required=True, help="BD manifest (image/mask/reference/type/utility)")
    p.add_argument("--draem-repo", default="DRAEM-main")
    p.add_argument(
        "--normal-data-dir",
        required=True,
        help="train/good + BN folder (original DRAEM normal training data + BN accepts)",
    )
    p.add_argument(
        "--anomaly-source-path",
        required=True,
        help="DTD image root used by the original DRAEM synthetic anomaly loader",
    )
    p.add_argument("--reconstruction-checkpoint", default=None,
                   help="required unless --init-random")
    p.add_argument("--segmentation-checkpoint", default=None,
                   help="required unless --init-random")
    p.add_argument("--init-random", action="store_true",
                   help="random init instead of loading checkpoints (fair vs baseline_200)")
    p.add_argument("--base-name", default="DRAEM_test_0.0001_200_bs8",
                   help="checkpoint name base, must match test_DRAEM.py --base_model_name")
    p.add_argument("--category", required=True,
                   help="object category, appended to run_name (e.g. bottle)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4,
                   help="MirrorEM mini-batch size")
    p.add_argument("--pre-batch-size", type=int, default=None,
                   help="Original DRAEM mini-batch size; defaults to --batch-size")
    p.add_argument("--mirror-weight", type=float, default=0.5,
                   help="w in (1-w)*L_pre + w*L_mirror")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Optimizer steps per epoch; defaults to max of the two loader lengths")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not 0.0 <= args.mirror_weight <= 1.0:
        p.error("--mirror-weight must be within [0, 1]")
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    if args.pre_batch_size is not None and args.pre_batch_size < 1:
        p.error("--pre-batch-size must be positive")
    if not args.init_random and (not args.reconstruction_checkpoint or not args.segmentation_checkpoint):
        p.error("--reconstruction-checkpoint and --segmentation-checkpoint are required "
                "unless --init-random is set")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # 复现性修复 (2026-09-01): cuDNN 默认可能选非确定性卷积算法 (Winograd 等),
    # 微小浮点差经 mirror-loss 反馈环放大 200 epoch 会把模型推向不同吸引子,
    # 导致"同数据同 seed"训练产出行为差异巨大的模型 (selection study S2/S3/S9
    # 验证: pixel AUC 0.25~0.60)。强制确定性算法后同数据同 seed 逐字节一致。
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    sys.path.insert(0, str(Path(args.draem_repo).resolve()))
    from data_loader import MVTecDRAEMTrainDataset
    from loss import FocalLoss, SSIM
    from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork

    device = torch.device(args.device)
    # DRAEM loss.py 的 SSIM() 构造里用隐式 .cuda()，走"当前默认设备"而非 --device。
    # 必须先把当前设备设为目标卡，否则分配会落到 cuda:0（可能被其他用户占满）。
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rec = ReconstructiveSubNetwork(3, 3).to(device)
    seg = DiscriminativeSubNetwork(6, 2).to(device)

    if args.init_random:
        rec.apply(weights_init)
        seg.apply(weights_init)
        print("[fix2] random init (DRAEM weights_init); checkpoints ignored")
    else:
        rec.load_state_dict(load_state_dict(args.reconstruction_checkpoint, device))
        seg.load_state_dict(load_state_dict(args.segmentation_checkpoint, device))
        print(f"[fix2] loaded from {args.reconstruction_checkpoint} / {args.segmentation_checkpoint}")

    pre_dataset = MVTecDRAEMTrainDataset(
        args.normal_data_dir,
        args.anomaly_source_path,
        resize_shape=[256, 256],
    )
    mirror_dataset = SelectedSeaSDataset(args.manifest)
    if not pre_dataset.image_paths:
        raise ValueError(f"No PNG normal images found in {args.normal_data_dir}")
    if not pre_dataset.anomaly_source_paths:
        raise ValueError(
            f"No DTD JPG anomaly sources found under {args.anomaly_source_path}/*/*.jpg"
        )

    pre_batch_size = args.pre_batch_size or args.batch_size
    loader_kwargs = {
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    pre_loader = DataLoader(
        pre_dataset,
        batch_size=pre_batch_size,
        **loader_kwargs,
    )
    mirror_loader = DataLoader(
        mirror_dataset,
        batch_size=args.batch_size,
        **loader_kwargs,
    )
    steps_per_epoch = args.steps_per_epoch or max(len(pre_loader), len(mirror_loader))
    if steps_per_epoch < 1:
        raise ValueError("--steps-per-epoch must be positive")

    optimizer = torch.optim.Adam(list(rec.parameters()) + list(seg.parameters()), lr=args.lr)
    # DRAEM lr schedule (matches baseline_200) for strict fairness.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [int(args.epochs * 0.8), int(args.epochs * 0.9)], gamma=0.2, last_epoch=-1
    )
    mse, ssim, focal = torch.nn.MSELoss(reduction="none"), SSIM(), FocalLoss()
    pre_batches = infinite_loader(pre_loader)
    mirror_batches = infinite_loader(mirror_loader)
    rec.train()
    seg.train()

    run_name = f"{args.base_name}_{args.category}_"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / f"{run_name}.pckl"
    seg_path = out / f"{run_name}_seg.pckl"

    print(
        f"pre_samples={len(pre_dataset)} mirror_samples={len(mirror_dataset)} "
        f"pre_batch={pre_batch_size} mirror_batch={args.batch_size} "
        f"steps_per_epoch={steps_per_epoch} mirror_weight={args.mirror_weight:.3f} "
        f"lr={args.lr} epochs={args.epochs} device={args.device}"
    )
    print(f"run_name={run_name}  save-> {rec_path} / {seg_path}")
    for epoch in range(args.epochs):
        running = {"total": 0.0, "pre": 0.0, "mirror": 0.0}
        for _ in range(steps_per_epoch):
            pre_loss, _ = draem_loss(
                next(pre_batches), rec, seg, mse, ssim, focal, device
            )
            mirror_loss, _ = draem_loss(
                next(mirror_batches),
                rec,
                seg,
                mse,
                ssim,
                focal,
                device,
                use_utility=True,
            )
            loss = (
                (1.0 - args.mirror_weight) * pre_loss
                + args.mirror_weight * mirror_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running["total"] += float(loss.detach())
            running["pre"] += float(pre_loss.detach())
            running["mirror"] += float(mirror_loss.detach())

        scheduler.step()

        scale = 1.0 / steps_per_epoch
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={running['total'] * scale:.6f} "
            f"pre={running['pre'] * scale:.6f} "
            f"mirror={running['mirror'] * scale:.6f} "
            f"lr={get_lr(optimizer):.1e}"
        )

        torch.save(rec.state_dict(), rec_path)
        torch.save(seg.state_dict(), seg_path)

    print(f"[fix2] done -> {rec_path} / {seg_path}")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    main()
