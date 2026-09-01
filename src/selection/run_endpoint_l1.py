#!/usr/bin/env python3
"""
run_endpoint_l1.py — L1 对齐池上的选择验证 (selectors 可配置)
================================================================
色差受控条件 (L1, t=0.5) 下对指定 selectors 重选 + 组装 fix2 数据 + 训练。
目的: 判断"随机选择 > 准则选择"是否在色差受控下仍成立。

用法:
  $DRAEM_PY src/selection/run_endpoint_l1.py --phase select --selectors S5_m_topk,S8_m_soft
  $DRAEM_PY src/selection/run_endpoint_l1.py --phase train   (每卡 1 模型)
"""
import argparse, os, sys, csv, glob, shutil, subprocess, json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'selection'))
from study_selectors import Pool, run_all, screening_stats
from run_study import K, NORMAL_SRC, DTD_SRC, DRAEM_REPO, BASE, CAT

POOL_ALIGN_CSV = os.path.join(ROOT, 'outputs', 'selection_pool_align_L1', CAT, 'candidate.csv')
SEL_L1_ROOT = os.path.join(ROOT, 'outputs', 'selection_L1' if CAT == 'pipe_fryum'
                           else f'selection_L1_{CAT}')
GPUS = [4, 5, 6, 7]


def assemble_one(pool, name, bn_idx, bd_idx):
    out = os.path.join(SEL_L1_ROOT, name)
    normal_dir = os.path.join(out, 'train_good_plus_bn')
    os.makedirs(normal_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(NORMAL_SRC, '*.png'))):
        shutil.copy2(f, os.path.join(normal_dir, os.path.basename(f)))
    for i, idx in enumerate(bn_idx):
        p = pool.data['img_path'][idx]
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(normal_dir, f'mb_bn_{i:03d}.png'))
    manifest = os.path.join(out, 'bd_manifest.csv')
    with open(manifest, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'mask', 'reference', 'type', 'utility'])
        for idx in bd_idx:
            img, mask, ref = pool.data['img_path'][idx], pool.data['mask_path'][idx], pool.data['ref_path'][idx]
            if os.path.exists(img) and os.path.exists(mask) and os.path.exists(ref):
                w.writerow([os.path.abspath(img), os.path.abspath(mask),
                            os.path.abspath(ref), 'defect', '1.0'])
    n_bn = len(glob.glob(os.path.join(normal_dir, 'mb_bn_*.png')))
    n_bd = sum(1 for _ in open(manifest)) - 1
    print(f'  [{name}] train_good_plus_bn={len(os.listdir(normal_dir))} (+{n_bn} BN), BD={n_bd}')
    return out


def phase_select(names):
    with open(POOL_ALIGN_CSV, newline='') as f:
        pool = Pool(list(csv.DictReader(f)))
    results = run_all(pool, K=K, seed=42)
    stats = screening_stats(pool, results)
    print(f'\nL1 对齐池 {pool.n} candidates, K={K}')
    hdr = f"{'selector':<12}{'bd_area_med':>12}{'bn_M_med':>9}{'bd_M_med':>9}{'bd_reg_med':>11}{'bd_small%':>10}"
    print(hdr)
    for name in names:
        s = stats[name]
        print(f"{name:<12}{s['bd_median_area']:>12.5f}{s['bn_median_M']:>9.3f}"
              f"{s['bd_median_M']:>9.3f}{s['bd_median_region']:>11.3f}"
              f"{s['bd_small_frac']*100:>9.1f}%")
    # 与 L2 端点选择的重叠
    with open(os.path.join(ROOT, 'outputs', 'selection_pool', CAT, 'candidate.csv'), newline='') as f:
        pool_l2 = Pool(list(csv.DictReader(f)))
    res_l2 = run_all(pool_l2, K=K, seed=42)
    for name in names:
        a, b = set(res_l2[name][1]), set(results[name][1])
        print(f'  {name} BD 集合 vs L2: Jaccard={len(a & b)/max(len(a|b),1):.3f} '
              f'(L2 {len(a)} BD -> L1 新 {len(b - a)} 个)')
    print('\n组装 fix2 数据 (L1)...')
    for name in names:
        bn, bd = results[name]
        assemble_one(pool, name, bn, bd)
    return pool, results


def train_cmd(name, gpu):
    out = os.path.join(SEL_L1_ROOT, name)
    py = '/home/chenjiawen/anaconda3/envs/DRAEM/bin/python'
    cmd = [py, os.path.join(ROOT, 'src', 'fix2', 'train_draem_fix2.py'),
           '--manifest', os.path.join(out, 'bd_manifest.csv'),
           '--normal-data-dir', os.path.join(out, 'train_good_plus_bn'),
           '--anomaly-source-path', DTD_SRC,
           '--init-random', '--base-name', BASE, '--category', CAT,
           '--output-dir', os.path.join(out, 'checkpoints', 'fix2'),
           '--epochs', '200', '--lr', '0.0001', '--mirror-weight', '0.5',
           '--batch-size', '4', '--steps-per-epoch', '138',
           '--draem-repo', DRAEM_REPO, '--device', f'cuda:{gpu}', '--seed', '42']
    return cmd


def phase_train(names):
    os.makedirs(os.path.join(SEL_L1_ROOT, '_logs'), exist_ok=True)
    pids = {}
    for i, name in enumerate(names):
        gpu = GPUS[i % len(GPUS)]
        log = os.path.join(SEL_L1_ROOT, '_logs', f'{name}.train.log')
        with open(log, 'w') as f:
            proc = subprocess.Popen(train_cmd(name, gpu), stdout=f, stderr=subprocess.STDOUT,
                                    cwd=ROOT, start_new_session=True)
        pids[name] = (proc.pid, gpu)
        print(f'  [{name}] gpu={gpu} pid={proc.pid} -> {log}')
    with open(os.path.join(SEL_L1_ROOT, '_logs', 'train_pids.json'), 'w') as f:
        json.dump(pids, f)
    return pids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True, choices=['select', 'train'])
    p.add_argument('--selectors', default='S0_random,S9_m_pareto',
                   help='逗号分隔的 selector 名')
    args = p.parse_args()
    names = [x.strip() for x in args.selectors.split(',') if x.strip()]
    if args.phase == 'select':
        phase_select(names)
    elif args.phase == 'train':
        phase_train(names)


if __name__ == '__main__':
    main()
