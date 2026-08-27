#!/usr/bin/env python3
"""
diag_seg_floor.py — DRAEM seg 背景地板与异常/正常分离诊断
==========================================================
复测 §7.2 口径: 按 GT mask 把 test 图 seg map 分为异常/正常像素，
统计:
  - GT 正常像素 seg 均值（背景地板）—— 地板越高，正常像素越易被误报
  - GT 异常像素 seg 均值
  - 分离比（Cohen's d 与 均值比 μ_anom/μ_norm）
  - 全像素 seg 全局均值、单图 seg 最大值分布

用法:
  $DRAEM_PY src/diag_seg_floor.py \
      --checkpoint_path outputs/fix2_align_L0/pipe_fryum/checkpoints/fix2 \
      --label L0 --out outputs/fix2_align_L0/pipe_fryum/seg_floor.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True,
                   help="DRAEM 兼容命名 checkpoint 目录（含 {base}_{cat}_.pckl 与 __seg.pckl）")
    p.add_argument("--draem_repo", default="/data/chenjiawen/DRAEM")
    p.add_argument("--test_path", default=None,
                   help="test 根（默认 outputs/visa_datasets/{cat}/test）")
    p.add_argument("--base_model_name", default="DRAEM_test_0.0001_200_bs8")
    p.add_argument("--category", default="pipe_fryum")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--label", default="", help="结果标签（如 L0/L1/L2）")
    p.add_argument("--out", default=None, help="JSON 输出路径")
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.draem_repo))
    from data_loader import MVTecDRAEMTestDataset
    from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork

    test_path = args.test_path or os.path.join(ROOT, "outputs", "visa_datasets",
                                               args.category, "test")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    run_name = f"{args.base_model_name}_{args.category}_"
    ckpt_rec = os.path.join(args.checkpoint_path, run_name + ".pckl")
    ckpt_seg = os.path.join(args.checkpoint_path, run_name + "_seg.pckl")
    for c in (ckpt_rec, ckpt_seg):
        if not os.path.exists(c):
            p.error(f"checkpoint 不存在: {c}")

    rec = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    rec.load_state_dict(torch.load(ckpt_rec, map_location=device))
    seg.load_state_dict(torch.load(ckpt_seg, map_location=device))
    rec.to(device).eval()
    seg.to(device).eval()

    ds = MVTecDRAEMTestDataset(test_path, resize_shape=[256, 256])
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    anom_means, norm_means, img_maxs = [], [], []
    anom_pix, norm_pix = [], []
    n_anom = 0
    with torch.no_grad():
        for i, b in enumerate(dl):
            gt = b["mask"].numpy()[0, 0]
            gt_bin = (gt > 0.01).astype(np.float32)
            x = b["image"].to(device)
            r = rec(x)
            joined = torch.cat((r.detach(), x), dim=1)
            o = torch.softmax(seg(joined), dim=1)[0, 1].detach().cpu().numpy()
            a_pix = o[gt_bin == 1]
            n_pix = o[gt_bin == 0]
            if a_pix.size > 0:
                anom_means.append(a_pix.mean())
                anom_pix.append(a_pix)
            norm_means.append(n_pix.mean())
            norm_pix.append(n_pix)
            img_maxs.append(o.max())
            if bool(b["has_anomaly"].numpy()[0, 0]):
                n_anom += 1

    anom_means = np.array(anom_means)
    norm_means = np.array(norm_means)
    anom_pix_all = np.concatenate(anom_pix) if anom_pix else np.zeros(0)
    norm_pix_all = np.concatenate(norm_pix)

    mu_a = anom_means.mean() if anom_means.size else float("nan")
    mu_n = norm_means.mean()
    pooled = np.sqrt((anom_pix_all.std() ** 2 + norm_pix_all.std() ** 2) / 2) \
        if anom_pix_all.size else np.nan
    cohen_d = (mu_a - mu_n) / pooled if pooled and pooled > 1e-9 else float("inf")
    ratio_mean = mu_a / mu_n if mu_n > 1e-9 else float("inf")

    result = {
        "label": args.label or os.path.basename(args.checkpoint_path),
        "checkpoint_path": args.checkpoint_path,
        "n_test_images": len(dl),
        "n_anomaly_images": n_anom,
        "gt_anomaly_pixel_seg_mean": float(mu_a),
        "gt_normal_pixel_seg_mean": float(mu_n),   # 背景地板（§7.2 口径）
        "background_floor": float(mu_n),
        "separation_cohens_d": float(cohen_d),
        "separation_ratio_mean": float(ratio_mean),  # μ_anom / μ_norm
        "all_pixel_seg_global_mean": float(norm_pix_all.mean()),
        "img_seg_max_mean": float(np.mean(img_maxs)),
        "img_seg_max_max": float(np.max(img_maxs)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
