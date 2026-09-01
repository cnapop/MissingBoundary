#!/usr/bin/env python3
"""eval_candle.py — candle 8 模型 + baseline 汇总评估.

用法: MB_CAT=candle python src/selection/eval_candle.py
逐模型调 eval_study.py (gpu_id round-robin 4/6/7), 解析 METRIC 行,
输出 outputs/selection_candle/candle_results.json + 对比表.
"""
import os, sys, json, subprocess, glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'selection'))
from run_study import BASE, CAT, TEST_DATA, DRAEM_REPO
import run_endpoint_l1

EVAL = os.path.join(ROOT, 'src', 'selection', 'eval_study.py')
PY = '/home/chenjiawen/anaconda3/envs/DRAEM/bin/python'
GPUS = [4, 6, 7]
NAMES = ['S0_random', 'S5_m_topk', 'S8_m_soft', 'S9_m_pareto']
L2_ROOT = os.path.join(ROOT, 'outputs', 'selection' if CAT == 'pipe_fryum' else f'selection_{CAT}')
L1_ROOT = os.path.join(ROOT, 'outputs', 'selection_L1' if CAT == 'pipe_fryum' else f'selection_L1_{CAT}')
BASELINE = os.path.join(ROOT, 'outputs', 'selection_pool', f'{CAT}_baseline.json')


def eval_one(name, ckpt, gpu):
    env = dict(os.environ)
    env['MB_CAT'] = CAT
    env['PYTHONPATH'] = DRAEM_REPO + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [PY, EVAL, '--gpu_id', str(gpu), '--base_model_name', BASE,
           '--data_path', TEST_DATA, '--checkpoint_path', ckpt]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=DRAEM_REPO, timeout=900, env=env)
    if r.returncode != 0:
        return {'error': r.stderr[-300:]}
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith('METRIC:'):
            return json.loads(line[7:])
    return {'error': 'no METRIC line', 'stdout_tail': r.stdout[-300:]}


def main():
    results = {}
    # baseline
    if os.path.exists(BASELINE):
        results['baseline'] = json.load(open(BASELINE))
    for i, name in enumerate(NAMES):
        gpu = GPUS[i % len(GPUS)]
        for cond, root in [('L2', L2_ROOT), ('L1', L1_ROOT)]:
            ckpt = os.path.join(root, name, 'checkpoints', 'fix2')
            rec = os.path.join(ckpt, f'{BASE}_{CAT}_.pckl')
            if not os.path.exists(rec):
                print(f'  [{cond}_{name}] MISSING checkpoint, skip')
                continue
            m = eval_one(name, ckpt, gpu)
            results[f'{cond}_{name}'] = m
            if 'error' in m:
                print(f'  [{cond}_{name}] FAILED: {m["error"][-200:]}')
            else:
                print(f"  [{cond}_{name}] img_auc={m.get('image_auc'):.4f} "
                      f"img_ap={m.get('image_ap'):.4f} pix_auc={m.get('pixel_auc'):.4f} "
                      f"pix_ap={m.get('pixel_ap'):.4f}")
    out = os.path.join(L2_ROOT, f'{CAT}_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)

    # 汇总表
    b = results.get('baseline', {})
    print('\n' + '=' * 78)
    print(f'Candle Selection Study (baseline pixel {b.get("pixel_auc", "?"):.4f})')
    print('=' * 78)
    hdr = f"{'model':<14}{'Img_AUC':>9}{'Img_AP':>9}{'Pix_AUC':>9}{'Pix_AP':>9}{'ΔPix':>8}"
    print(hdr)
    print(f"{'baseline':<14}{b.get('image_auc', 0):>9.4f}{b.get('image_ap', 0):>9.4f}"
          f"{b.get('pixel_auc', 0):>9.4f}{b.get('pixel_ap', 0):>9.4f}{'--':>8}")
    for cond in ['L2', 'L1']:
        for name in NAMES:
            m = results.get(f'{cond}_{name}')
            if not m or 'error' in m:
                print(f"{cond}_{name:<12}{'--':>9}")
                continue
            d = m['pixel_auc'] - b.get('pixel_auc', 0)
            print(f"{cond}_{name:<12}{m['image_auc']:>9.4f}{m['image_ap']:>9.4f}"
                  f"{m['pixel_auc']:>9.4f}{m['pixel_ap']:>9.4f}{d:>+8.4f}")
    print('=' * 78)
    print(f'\n结果: {out}')


if __name__ == '__main__':
    main()
