#!/usr/bin/env python3
"""
run_study.py — Selection Study Phase B/D/E 驱动
================================================
  --phase select : 跑 10 个 selector → screening 统计 → 组装 fix2 数据 (快)
  --phase train  : 组装全部 10 个 fix2 训练并后台启动 (round-robin GPU)
  --phase eval   : 评估全部 10 个 checkpoint → 汇总对比表

固定训练条件 (公平性): init-random, 200ep, lr1e-4, w0.5, bs4,
  steps_per_epoch=138 (K=100 → pre=550), seed=42。
"""
import argparse, os, sys, json, csv, glob, shutil, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'selection'))
from study_selectors import Pool, run_all, screening_stats, set_overlap

CAT = 'pipe_fryum'
K = 100
GPUS = [0, 2, 6, 7]
BASE = 'DRAEM_test_0.0001_200_bs8'
POOL_CSV = os.path.join(ROOT, 'outputs', 'selection_pool', CAT, 'candidate.csv')
SEL_ROOT = os.path.join(ROOT, 'outputs', 'selection')
NORMAL_SRC = os.path.join(ROOT, 'outputs', 'visa_datasets', CAT, 'train', 'good')
DTD_SRC = '/data/chenjiawen/DRAEM/datasets/dtd/images'
DRAEM_REPO = '/data/chenjiawen/DRAEM'
EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_study.py')
TEST_DATA = os.path.join(ROOT, 'outputs', 'visa_datasets')

SELECTOR_NAMES = ['S0_random', 'S1_area', 'S2_m_only', 'S3_m_area', 'S4_m_score',
                  'S5_m_topk', 'S6_m_region', 'S7_m_multi', 'S8_m_soft', 'S9_m_pareto']


def load_pool():
    with open(POOL_CSV, newline='') as f:
        rows = list(csv.DictReader(f))
    return Pool(rows)


def assemble_one(pool, name, bn_idx, bd_idx):
    """为单个 selector 组装 fix2 数据 (train_good_plus_bn + bd_manifest.csv)."""
    out = os.path.join(SEL_ROOT, name)
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
            img = pool.data['img_path'][idx]
            mask = pool.data['mask_path'][idx]
            ref = pool.data['ref_path'][idx]
            if os.path.exists(img) and os.path.exists(mask) and os.path.exists(ref):
                w.writerow([os.path.abspath(img), os.path.abspath(mask),
                            os.path.abspath(ref), 'defect', '1.0'])
    n_bn = len(glob.glob(os.path.join(normal_dir, 'mb_bn_*.png')))
    n_bd = sum(1 for _ in open(manifest)) - 1
    print(f'  [{name}] train_good_plus_bn={len(os.listdir(normal_dir))} '
          f'(+{n_bn} BN), BD manifest={n_bd}')
    return out


def phase_select(pool=None):
    pool = pool if pool is not None else load_pool()
    results = run_all(pool, K=K, seed=42)
    stats = screening_stats(pool, results)
    ov = set_overlap(results)

    os.makedirs(SEL_ROOT, exist_ok=True)
    with open(os.path.join(SEL_ROOT, 'screening_stats.json'), 'w') as f:
        json.dump({'K': K, 'n_pool': pool.n, 'stats': stats, 'overlap': ov}, f, indent=2)

    print(f'\n候选池: {pool.n} candidates, K={K}  (BD overlap = Jaccard)')
    header = (f"{'selector':<12}{'n_bn':>5}{'n_bd':>5}{'acc%':>6}{'bd_small%':>9}"
              f"{'bd_area_med':>12}{'bn_M_med':>9}{'bd_M_med':>9}{'bd_reg_med':>11}"
              f"{'bd_score':>9}{'bd_amax':>9}")
    print(header)
    for name in SELECTOR_NAMES:
        s = stats[name]
        print(f"{name:<12}{s['n_bn']:>5}{s['n_bd']:>5}{s['accept_rate']*100:>6.1f}"
              f"{s['bd_small_frac']*100:>9.1f}{s['bd_median_area']:>12.4f}"
              f"{s['bn_median_M']:>9.3f}{s['bd_median_M']:>9.3f}"
              f"{s['bd_median_region']:>11.3f}{s['bd_median_score']:>9.3f}"
              f"{s['bd_median_amap_max']:>9.3f}")
    print('\nBD 集合重叠 (Jaccard):')
    for k, v in ov.items():
        print(f'  {k}: {v}')

    print('\n组装 fix2 数据...')
    for name in SELECTOR_NAMES:
        bn, bd = results[name]
        assemble_one(pool, name, bn, bd)
    return results


def train_cmd(name, gpu):
    out = os.path.join(SEL_ROOT, name)
    manifest = os.path.join(out, 'bd_manifest.csv')
    normal_dir = os.path.join(out, 'train_good_plus_bn')
    ckpt = os.path.join(out, 'checkpoints', 'fix2')
    py = '/home/chenjiawen/anaconda3/envs/DRAEM/bin/python'
    cmd = [py, os.path.join(ROOT, 'src', 'train_draem_fix2.py'),
           '--manifest', manifest,
           '--normal-data-dir', normal_dir,
           '--anomaly-source-path', DTD_SRC,
           '--init-random', '--base-name', BASE, '--category', CAT,
           '--output-dir', ckpt,
           '--epochs', '200', '--lr', '0.0001', '--mirror-weight', '0.5',
           '--batch-size', '4', '--steps-per-epoch', '138',
           '--draem-repo', DRAEM_REPO, '--device', f'cuda:{gpu}', '--seed', '42']
    return cmd, ckpt


def phase_train():
    os.makedirs(os.path.join(SEL_ROOT, '_logs'), exist_ok=True)
    pids = {}
    for i, name in enumerate(SELECTOR_NAMES):
        gpu = GPUS[i % len(GPUS)]
        cmd, ckpt = train_cmd(name, gpu)
        log = os.path.join(SEL_ROOT, '_logs', f'{name}.train.log')
        with open(log, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    cwd=ROOT, start_new_session=True)
        pids[name] = proc.pid
        print(f'  [{name}] gpu={gpu} pid={proc.pid} -> {log}')
    with open(os.path.join(SEL_ROOT, '_logs', 'train_pids.json'), 'w') as f:
        json.dump(pids, f)
    print(f'\n已启动 {len(pids)} 个训练 (round-robin {GPUS}). 日志: {SEL_ROOT}/_logs/')
    return pids


def phase_eval(gpus=GPUS):
    """评估全部已训练 checkpoint → selection_results.json + 打印表."""
    results = {}
    import importlib.util
    for i, name in enumerate(SELECTOR_NAMES):
        ckpt = os.path.join(SEL_ROOT, name, 'checkpoints', 'fix2')
        rec = os.path.join(ckpt, f'{BASE}_{CAT}_.pckl')
        if not os.path.exists(rec):
            print(f'  [{name}] MISSING checkpoint, skip')
            continue
        gpu = gpus[i % len(gpus)]
        py = '/home/chenjiawen/anaconda3/envs/DRAEM/bin/python'
        cmd = [py, EVAL_SCRIPT, '--gpu_id', str(gpu),
               '--base_model_name', BASE,
               '--data_path', TEST_DATA,
               '--checkpoint_path', ckpt]
        env = dict(os.environ)
        env['PYTHONPATH'] = DRAEM_REPO + os.pathsep + env.get('PYTHONPATH', '')
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=DRAEM_REPO,
                           timeout=900, env=env)
        if r.returncode != 0:
            print(f'  [{name}] eval FAILED: {r.stderr[-300:]}')
            results[name] = None
            continue
        # 解析 JSON 行
        metric = None
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith('METRIC:'):
                metric = json.loads(line[7:])
        results[name] = metric
        m = metric or {}
        print(f"  [{name}] img_auc={m.get('image_auc'):.4f} img_ap={m.get('image_ap'):.4f} "
              f"pix_auc={m.get('pixel_auc'):.4f} pix_ap={m.get('pixel_ap'):.4f}")
    with open(os.path.join(SEL_ROOT, 'selection_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def report(results):
    """汇总成对比表 (含 baseline)。"""
    baseline = {'image_auc': 0.9336, 'image_ap': 0.9709, 'pixel_auc': 0.6982, 'pixel_ap': 0.1012}
    print('\n' + '=' * 80)
    print('Selection Study — pipe_fryum 原始 test (150 张)')
    print('=' * 80)
    hdr = f"{'selector':<12}{'Img_AUC':>9}{'Img_AP':>9}{'Pix_AUC':>9}{'Pix_AP':>9}{'ΔPix_AUC':>10}"
    print(hdr)
    print(f"{'baseline':<12}{baseline['image_auc']:>9.4f}{baseline['image_ap']:>9.4f}"
          f"{baseline['pixel_auc']:>9.4f}{baseline['pixel_ap']:>9.4f}{'--':>10}")
    for name in SELECTOR_NAMES:
        m = results.get(name)
        if m is None:
            print(f"{name:<12}{'--':>9}")
            continue
        d = m['pixel_auc'] - baseline['pixel_auc']
        print(f"{name:<12}{m['image_auc']:>9.4f}{m['image_ap']:>9.4f}"
              f"{m['pixel_auc']:>9.4f}{m['pixel_ap']:>9.4f}{d:>+10.4f}")
    print('=' * 80)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True, choices=['select', 'train', 'eval'])
    args = p.parse_args()
    if args.phase == 'select':
        phase_select()
    elif args.phase == 'train':
        phase_train()
    elif args.phase == 'eval':
        results = phase_eval()
        report(results)


if __name__ == '__main__':
    main()
