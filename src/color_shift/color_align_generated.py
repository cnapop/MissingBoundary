#!/usr/bin/env python3
"""
color_align_generated.py — Reinhard LAB 颜色传递校准 SeaS 生成图
=================================================================
把 MB v3 闭环接受样本的 VAE 偏色（SUMMARY: 生成 gap~9x 的一部分）校准到其参考图
train/good/<ref>.png 的 LAB 统计。mask 不参与颜色变换（像素坐标不变），直接复制。

剂量控制（t = 插值系数，控制残留色差）:
  --t 1.0 → L0 完全对齐（per-image Reinhard 传递，ΔE≈0）
  --t 0.5 → L1 保留一半色差（原图与对齐图各半）
  --t 0.0 → 原图（剂量报告基线）

参考映射:
  boundary_normal/sample_<X>/acc_*.png → meta.json source_path（= train/good/<X>.png）
  blind_defect/sample_<X>/acc_*.png     → meta.json source_path（同源）

输出:
  <out_root>/<category>/{boundary_normal,blind_defect}/... — 与原 gen_root 同目录结构
  <out_root>/<category>/color_shift_report.json — 变换前后每通道 Δμ/Δσ + mean ΔE76
  <out_root>/<category>/_align_review/ — 抽样对比拼图（原图 | 对齐图 | 参考图）

用法:
  $DRAEM_PY src/color_align_generated.py --category pipe_fryum \
      --gen_root outputs/generated_v3_visa_clean --t 1.0 --level L0
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reinhard_lab(img_bgr, ref_bgr, t):
    """Per-image Reinhard LAB 颜色传递。

    img_bgr/ref_bgr: uint8 BGR (HxWx3)。每通道 (x - μ_img)/σ_img * σ_ref + μ_ref。
    t=1 全量传递; t=0.5 原图与对齐图各半; t=0 原图不变。
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    aligned = np.empty_like(lab)
    for c in range(3):
        m = lab[..., c].mean()
        s = lab[..., c].std()
        mr = ref_lab[..., c].mean()
        sr = ref_lab[..., c].std()
        aligned[..., c] = (lab[..., c] - m) / (s + 1e-6) * sr + mr
    aligned_8 = np.clip(aligned, 0, 255).astype(np.uint8)
    aligned_bgr = cv2.cvtColor(aligned_8, cv2.COLOR_LAB2BGR)
    if t >= 1.0:
        return aligned_bgr
    return cv2.addWeighted(aligned_bgr, t, img_bgr, 1.0 - t, 0)


def global_stats_match(img_bgr, ref_stats):
    """全局统计匹配（回退方案）：生成集整体匹配真实集每通道 μ/σ，不用参考图。

    ref_stats: {channel: (mu, sigma)} 为全局真实集的每通道均值/标准差。
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    aligned = np.empty_like(lab)
    for c in range(3):
        mu, sigma = ref_stats[c]
        m = lab[..., c].mean()
        s = lab[..., c].std()
        aligned[..., c] = (lab[..., c] - m) / (s + 1e-6) * sigma + mu
    return cv2.cvtColor(np.clip(aligned, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def lab_delta(img_bgr, ref_bgr):
    """生成图 vs 参考图在 LAB 空间的每通道 Δμ/Δσ + mean ΔE76 + 每通道 rms Δ。"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dmu = [float(lab[..., c].mean() - ref_lab[..., c].mean()) for c in range(3)]
    dsigma = [float(lab[..., c].std() - ref_lab[..., c].std()) for c in range(3)]
    dE = float(np.sqrt(((lab - ref_lab) ** 2).sum(axis=2)).mean())
    dch = [float(np.sqrt(((lab[..., c] - ref_lab[..., c]) ** 2).mean())) for c in range(3)]
    return dmu, dsigma, dE, dch


def ref_from_meta(meta_path):
    """从 meta.json 取 source_path（绝对或相对）作为颜色参考。"""
    with open(meta_path) as f:
        m = json.load(f)
    return m.get("source_path")


def load_meta_refs(split_dir):
    """读 split（boundary_normal / blind_defect）下每个 sample 的 meta.json，
    返回 {sample_dir: ref_abs_path}。"""
    refs = {}
    for meta in sorted(__import__("glob").glob(os.path.join(split_dir, "*", "meta.json"))):
        src = ref_from_meta(meta)
        sample_dir = os.path.basename(os.path.dirname(meta))
        refs[sample_dir] = src
    return refs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", default="pipe_fryum")
    p.add_argument("--gen_root", default=None,
                   help="生成图根 (默认 outputs/generated_v3_visa_clean)")
    p.add_argument("--out_root", default=None,
                   help="对齐输出根 (默认 outputs/generated_v3_visa_clean_align_L<t>)")
    p.add_argument("--normal_ref_dir", default=None,
                   help="真实参考目录 (默认 outputs/visa_datasets/{cat}/train/good)")
    p.add_argument("--t", type=float, default=1.0,
                   help="插值系数: 1.0=L0 全量对齐, 0.5=L1 半量, 0=原图")
    p.add_argument("--level", default=None,
                   help="剂量标签 (默认 L0/L1，由 t 推导)")
    p.add_argument("--method", choices=["per_image", "global"], default="per_image",
                   help="per_image=逐图对齐其参考; global=全局统计匹配（回退方案）")
    p.add_argument("--n_review", type=int, default=8,
                   help="抽样对比拼图数量")
    args = p.parse_args()

    cat = args.category
    gen_root = os.path.join(ROOT, args.gen_root) if args.gen_root else \
        os.path.join(ROOT, "outputs", "generated_v3_visa_clean")
    gen_cat = os.path.join(gen_root, cat)
    if not os.path.isdir(gen_cat):
        p.error(f"gen_cat 不存在: {gen_cat}")
    normal_ref_dir = args.normal_ref_dir or \
        os.path.join(ROOT, "outputs", "visa_datasets", cat, "train", "good")
    level = args.level or (f"L{args.t}" if args.t in (0.0, 0.5, 1.0) else f"t{args.t:.2f}")
    out_root = os.path.join(ROOT, args.out_root) if args.out_root else \
        os.path.join(ROOT, f"outputs", f"generated_v3_visa_clean_align_{level}")
    out_cat = os.path.join(out_root, cat)
    review_dir = os.path.join(out_cat, "_align_review")

    if args.method == "global":
        # 全局统计：真实参考集每通道 μ/σ
        ref_stats = {}
        lab_all = []
        for f in sorted(os.listdir(normal_ref_dir)):
            if not f.endswith(".png"):
                continue
            lab_all.append(cv2.cvtColor(cv2.imread(os.path.join(normal_ref_dir, f)),
                                        cv2.COLOR_BGR2LAB).astype(np.float32))
        lab_all = np.concatenate([x[None] for x in lab_all], axis=0) if lab_all else None
        if lab_all is None:
            p.error(f"normal_ref_dir 无 PNG: {normal_ref_dir}")
        for c in range(3):
            ref_stats[c] = (float(lab_all[..., c].mean()), float(lab_all[..., c].std()))

    # 真实参考文件名 → 绝对路径（供 per_image 解析 meta source_path）
    ref_name2path = {f: os.path.join(normal_ref_dir, f)
                     for f in os.listdir(normal_ref_dir) if f.endswith(".png")}

    splits = ["boundary_normal", "blind_defect"]
    report = {
        "level": level,
        "t": args.t,
        "method": args.method,
        "category": cat,
        "reference_root": normal_ref_dir,
        "n_images": 0,
        "before": {"dmu": [0.0] * 3, "dsigma": [0.0] * 3, "dE": 0.0},
        "after": {"dmu": [0.0] * 3, "dsigma": [0.0] * 3, "dE": 0.0},
        "per_split": {},
        "skipped": [],
    }
    all_before_dE, all_after_dE = [], []
    all_before_dmu, all_after_dmu = defaultdict(list), defaultdict(list)
    all_before_dsigma, all_after_dsigma = defaultdict(list), defaultdict(list)
    all_before_dch, all_after_dch = defaultdict(list), defaultdict(list)
    review_rows = []  # (label, orig, aligned, ref)

    for split in splits:
        split_dir = os.path.join(gen_cat, split)
        if not os.path.isdir(split_dir):
            print(f"  [warn] split 不存在: {split_dir}")
            continue
        meta_refs = load_meta_refs(split_dir)  # {sample_dir: ref_rel_or_abs}
        acc_imgs = sorted(__import__("glob").glob(os.path.join(split_dir, "*", "acc_*.png")))
        # 排除 mask（acc_XX_mask.png）
        acc_imgs = [f for f in acc_imgs if not f.endswith("_mask.png")]
        out_split = os.path.join(out_cat, split)
        before_agg = {"dmu": [0.0] * 3, "dsigma": [0.0] * 3, "dE": 0.0, "dch": [0.0] * 3}
        after_agg = {"dmu": [0.0] * 3, "dsigma": [0.0] * 3, "dE": 0.0, "dch": [0.0] * 3}
        n = 0
        sp_bdmu, sp_admu = defaultdict(list), defaultdict(list)
        sp_bds, sp_ads = defaultdict(list), defaultdict(list)
        sp_bdE, sp_adE = [], []
        sp_bdch, sp_adch = defaultdict(list), defaultdict(list)
        for img_path in acc_imgs:
            sample_dir = os.path.basename(os.path.dirname(img_path))
            ref_path = meta_refs.get(sample_dir)
            if ref_path is None:
                report["skipped"].append(img_path)
                print(f"  [skip no-ref] {img_path}")
                continue
            ref_abs = ref_path if os.path.isabs(ref_path) else \
                os.path.join(ROOT, ref_path)
            if not os.path.exists(ref_abs):
                base = os.path.basename(ref_abs)
                if base in ref_name2path:
                    ref_abs = ref_name2path[base]
                else:
                    report["skipped"].append(img_path)
                    print(f"  [skip ref-missing] {img_path} ref={ref_abs}")
                    continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            ref = cv2.imread(ref_abs, cv2.IMREAD_COLOR)
            if img is None or ref is None:
                report["skipped"].append(img_path)
                print(f"  [skip unreadable] {img_path}")
                continue
            if ref.shape != img.shape:
                ref = cv2.resize(ref, (img.shape[1], img.shape[0]))

            dmu_b, dsig_b, dE_b, dch_b = lab_delta(img, ref)
            aligned = reinhard_lab(img, ref, args.t) if args.method == "per_image" \
                else global_stats_match(img, ref_stats)
            dmu_a, dsig_a, dE_a, dch_a = lab_delta(aligned, ref)

            # 写出：同目录结构
            rel = os.path.relpath(img_path, gen_cat)
            out_img = os.path.join(out_cat, rel)
            os.makedirs(os.path.dirname(out_img), exist_ok=True)
            cv2.imwrite(out_img, aligned)

            # 剂量聚合（全局 + 本 split）
            for c in range(3):
                all_before_dmu[c].append(dmu_b[c])
                all_after_dmu[c].append(dmu_a[c])
                all_before_dsigma[c].append(dsig_b[c])
                all_after_dsigma[c].append(dsig_a[c])
                all_before_dch[c].append(dch_b[c])
                all_after_dch[c].append(dch_a[c])
                sp_bdmu[c].append(dmu_b[c])
                sp_admu[c].append(dmu_a[c])
                sp_bds[c].append(dsig_b[c])
                sp_ads[c].append(dsig_a[c])
                sp_bdch[c].append(dch_b[c])
                sp_adch[c].append(dch_a[c])
            all_before_dE.append(dE_b)
            all_after_dE.append(dE_a)
            sp_bdE.append(dE_b)
            sp_adE.append(dE_a)
            n += 1

            # 抽样存拼图
            if len(review_rows) < args.n_review:
                review_rows.append((f"{split}/{os.path.relpath(img_path, gen_cat)}",
                                    img, aligned, ref))

        if n > 0:
            before_agg = {
                "dmu": [float(np.mean(sp_bdmu[c])) for c in range(3)],
                "dsigma": [float(np.mean(sp_bds[c])) for c in range(3)],
                "dE": float(np.mean(sp_bdE)),
                "dch": [float(np.mean(sp_bdch[c])) for c in range(3)],
            }
            after_agg = {
                "dmu": [float(np.mean(sp_admu[c])) for c in range(3)],
                "dsigma": [float(np.mean(sp_ads[c])) for c in range(3)],
                "dE": float(np.mean(sp_adE)),
                "dch": [float(np.mean(sp_adch[c])) for c in range(3)],
            }
        report["per_split"][split] = {
            "n": n,
            "before": before_agg,
            "after": after_agg,
        }
        report["n_images"] += n
        print(f"{split}: {n} 张 | before dE={before_agg['dE']:.3f} after dE={after_agg['dE']:.3f}")

        # 复制 mask 与 meta.json（mask 不参与颜色变换）
        for extra in __import__("glob").glob(os.path.join(split_dir, "*", "*.png")) + \
                       __import__("glob").glob(os.path.join(split_dir, "*", "meta.json")):
            if extra.endswith("_mask.png") or extra.endswith("meta.json"):
                out_extra = os.path.join(out_cat, os.path.relpath(extra, gen_cat))
                os.makedirs(os.path.dirname(out_extra), exist_ok=True)
                shutil.copy2(extra, out_extra)

    # 全量聚合
    if report["n_images"] > 0:
        report["before"] = {
            "dmu": [float(np.mean(all_before_dmu[c])) for c in range(3)],
            "dsigma": [float(np.mean(all_before_dsigma[c])) for c in range(3)],
            "dE": float(np.mean(all_before_dE)),
            "dE_median": float(np.median(all_before_dE)),
            "dE_p90": float(np.percentile(all_before_dE, 90)),
            "dch": [float(np.mean(all_before_dch[c])) for c in range(3)],
        }
        report["after"] = {
            "dmu": [float(np.mean(all_after_dmu[c])) for c in range(3)],
            "dsigma": [float(np.mean(all_after_dsigma[c])) for c in range(3)],
            "dE": float(np.mean(all_after_dE)),
            "dE_median": float(np.median(all_after_dE)),
            "dE_p90": float(np.percentile(all_after_dE, 90)),
            "dch": [float(np.mean(all_after_dch[c])) for c in range(3)],
        }

    # 真实集内天然 ΔE 基线（同尺寸两两采样）：判断残留 ΔE 是否已到"真实图间"水平
    try:
        goods = sorted(f for f in os.listdir(normal_ref_dir) if f.endswith(".png"))
        sample_n = min(40, len(goods))
        rng = np.random.default_rng(0)
        pairs = [(i, j) for i in range(sample_n) for j in range(i + 1, sample_n)]
        rng.shuffle(pairs)
        pairs = pairs[:60]
        lab_cache = {}
        dE_vals = []
        for i, j in pairs:
            if i not in lab_cache:
                lab_cache[i] = cv2.cvtColor(
                    cv2.imread(os.path.join(normal_ref_dir, goods[i])),
                    cv2.COLOR_BGR2LAB).astype(np.float32)
            if j not in lab_cache:
                lab_cache[j] = cv2.cvtColor(
                    cv2.imread(os.path.join(normal_ref_dir, goods[j])),
                    cv2.COLOR_BGR2LAB).astype(np.float32)
            g0, g1 = lab_cache[i], lab_cache[j]
            if tuple(g0.shape) != tuple(g1.shape):
                g1 = cv2.resize(g1, (g0.shape[1], g0.shape[0]))
            dE_vals.append(float(np.sqrt(((g0 - g1) ** 2).sum(axis=2)).mean()))
        report["real_reference_baseline"] = {
            "n_pairs": len(dE_vals),
            "dE_mean": float(np.mean(dE_vals)),
            "dE_std": float(np.std(dE_vals)),
            "note": "train/good 内两两 ΔE —— 残留 dE 若与其相当，说明生成图颜色已与真实图不可区分",
        }
    except Exception as e:  # 基线计算失败不影响主流程
        report["real_reference_baseline"] = {"error": str(e)}

    # 复制 generation_closedloop_summary.json（供审计）
    summary = os.path.join(gen_cat, "generation_closedloop_summary.json")
    if os.path.exists(summary):
        shutil.copy2(summary, os.path.join(out_cat, "generation_closedloop_summary.json"))

    # 抽样拼图：三列 (原图 | 对齐图 | 参考图)
    os.makedirs(review_dir, exist_ok=True)
    if review_rows:
        h = img.shape[0] if "img" in dir() else 256
        pad = 6
        cols = 3
        rows = len(review_rows)
        canvas = np.full((rows * h + (rows + 1) * pad, cols * h + (cols + 1) * pad, 3),
                         255, np.uint8)
        for i, (label, orig, aligned, ref) in enumerate(review_rows):
            y = pad + i * (h + pad)
            for j, arr in enumerate((orig, aligned, ref)):
                x = pad + j * (h + pad)
                canvas[y:y + h, x:x + h] = cv2.resize(arr, (h, h))
        cv2.imwrite(os.path.join(review_dir, "align_review.png"), canvas)
        with open(os.path.join(review_dir, "review_labels.txt"), "w") as f:
            for label, _, _, _ in review_rows:
                f.write(label + "\n")

    report_path = os.path.join(out_cat, "color_shift_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n== 剂量报告 {report_path} ==")
    print(json.dumps(report["before"], ensure_ascii=False, indent=2))
    print("after:", json.dumps(report["after"], ensure_ascii=False, indent=2))
    print(f"抽样拼图: {os.path.join(review_dir, 'align_review.png')}")
    print(f"总计 {report['n_images']} 张, skip {len(report['skipped'])}")


if __name__ == "__main__":
    main()
