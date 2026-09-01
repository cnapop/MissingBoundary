#!/usr/bin/env python3
"""
align_pool_l1.py — 把 selection pool 生成图做 L1 (t=0.5) 颜色对齐并重算全部特征
================================================================================
色差消融 (color-shift-ablation) 证实 VAE 偏色主导 pool 的 M/gap 特征
(M≈0.5·gap, PB≈0.506 常数; gap 的根源是生成图偏色)。本脚本把选择研究
的控制变量补齐:
  1. 对 candidate.csv 每张生成图做 Reinhard LAB L1 对齐 (对齐到 ref_path, t=0.5)
     —— 保留半量色差, 与色差实验 L1 sweet spot 一致
  2. 用同一 oracle 重算 M/gap/pb/score + amap 统计 (mask 不变, 不重算 mask 形状)
  3. 输出 selection_pool_align_L1/pipe_fryum/candidate.csv

用法:
  $DRAEM_PY src/selection/align_pool_l1.py \
    --pool_csv outputs/selection_pool/pipe_fryum/candidate.csv \
    --oracle_dir outputs/features_visa/pipe_fryum/missing_boundary/mx_oracle \
    --out_root outputs/selection_pool_align_L1 --score_gpu 7 --t 0.5
"""
import argparse, os, sys, csv, time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'selection'))
from build_candidate_pool import mask_amap_features
from mx_oracle import MxOracle
from color_shift.color_align_generated import reinhard_lab


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--category', default='pipe_fryum')
    p.add_argument('--pool_csv', required=True)
    p.add_argument('--oracle_dir', required=True)
    p.add_argument('--dataset_dir', default=os.path.join(ROOT, 'outputs', 'visa_datasets'))
    p.add_argument('--out_root', default=os.path.join(ROOT, 'outputs', 'selection_pool_align_L1'))
    p.add_argument('--draem_ckpt_dir', default=os.path.join(ROOT, 'outputs', 'checkpoints', 'visa'))
    p.add_argument('--draem_base_name', default='DRAEM_test_0.0001_200_bs8')
    p.add_argument('--draem_dir', default='/data/chenjiawen/DRAEM')
    p.add_argument('--t', type=float, default=0.5)
    p.add_argument('--score_gpu', type=int, default=7)
    args = p.parse_args()

    with open(args.pool_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    print(f'load {len(rows)} candidates from {args.pool_csv}')

    oracle = MxOracle.load(args.oracle_dir, draem_dir=args.draem_dir,
                           checkpoint_dir=args.draem_ckpt_dir,
                           base_model_name=args.draem_base_name,
                           category=args.category,
                           device=f'cuda:{args.score_gpu}')
    print(f'oracle on cuda:{args.score_gpu}')

    # mask 形状特征不随颜色对齐变化; 只需重算 amap 统计 → 这里复算全部 mask_amap_features
    fieldnames = list(rows[0].keys())
    out_rows = []
    start = time.time()
    n_fail = 0
    for i, row in enumerate(rows):
        img_path = row['img_path']
        mask_path = row['mask_path']
        ref_path = row['ref_path']
        # 对齐输出路径: 镜像 pool 结构, 换 out_root 前缀
        aligned_path = img_path.replace(
            os.path.join('outputs', 'selection_pool', ''),
            os.path.join('outputs', os.path.basename(args.out_root.rstrip('/')), ''))
        if aligned_path == img_path:
            aligned_path = img_path.replace('/selection_pool/', '/selection_pool_align_L1/')
        os.makedirs(os.path.dirname(aligned_path), exist_ok=True)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if img is None or ref is None:
            print(f'  [fail read] {img_path} / {ref_path}')
            n_fail += 1
            continue
        if ref.shape != img.shape:
            ref = cv2.resize(ref, (img.shape[1], img.shape[0]))
        aligned = reinhard_lab(img, ref, args.t)
        cv2.imwrite(aligned_path, aligned)

        try:
            r = oracle.score_full(aligned_path)
        except Exception as e:
            print(f'  [score ERR] {aligned_path}: {e}')
            n_fail += 1
            continue
        amap_np = r.pop('amap_np')
        try:
            f = mask_amap_features(mask_path, amap_np)
        except Exception as e:
            print(f'  [mask ERR] {mask_path}: {e}')
            n_fail += 1
            continue
        new = dict(row)
        new['img_path'] = aligned_path
        new['M'] = r['M']
        new['gap'] = r['gap']
        new['pb'] = r['pb']
        new['score'] = r['score']
        new['amap_cov'] = r['amap_cov']
        new.update(f)
        out_rows.append(new)
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(rows)} ({time.time()-start:.0f}s)')

    out_cat = os.path.join(args.out_root, args.category)
    os.makedirs(out_cat, exist_ok=True)
    out_csv = os.path.join(out_cat, 'candidate.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    arr = np.array([r['mask_area'] for r in out_rows])
    print(f'\n对齐池完成: {len(out_rows)} rows ({n_fail} fail), {time.time()-start:.0f}s')
    print(f'  csv: {out_csv}')
    print(f'  mask_area: min={arr.min():.5f} med={np.median(arr):.5f} P90={np.percentile(arr,90):.5f}')
    # 对比: 对齐前后 M 分布变化 (验证色差分量被压掉)
    for col in ('M', 'gap'):
        old = np.array([float(r[col]) for r in rows])
        new_v = np.array([float(r[col]) for r in out_rows])
        print(f'  {col}: L2 med={np.median(old):.4f} -> L1 med={np.median(new_v):.4f} '
              f'(mean {old.mean():.4f}->{new_v.mean():.4f})')


if __name__ == '__main__':
    main()
