#!/usr/bin/env python3
"""
run_g1_summary.py — G1 结果汇总 (mean ± std)
=============================================
自动读取 experiments/mb_validation/G1_seed_variance/{cat}/{method}_seed{s}/eval.json,
生成方案 §4.6 要求的表 + §二十六 的逐 run 明细表, 并给出 §五 的判定。

不手工抄任何数字。缺 eval.json 的 run 记为缺失并标出。

用法:
  {PY} scripts/mb_validation/run_g1_summary.py [--out .../summary.json]
"""
import argparse, csv, json, os, statistics as st

ROOT = '/data/chenjiawen/MissingBoundary'
EXP_ROOT = os.path.join(ROOT, 'experiments', 'mb_validation', 'G1_seed_variance')
METHODS = ['baseline', 'S8']
METRICS = ['pixel_auc', 'image_auc', 'pixel_ap', 'image_ap']


def load(cat, method, seed):
    p = os.path.join(EXP_ROOT, cat, f'{method}_seed{seed}', 'eval.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return st.mean(vals), st.stdev(vals), len(vals)


def verdict(mean_b, std_b, mean_s, std_s, n):
    """按方案 §五 给出 indicative 判定 (n=3 时 std 本身不确定, 仅作参考)。"""
    if mean_b is None or mean_s is None:
        return '数据不全', None
    d = mean_s - mean_b
    pooled = (std_b ** 2 + std_s ** 2) ** 0.5
    if pooled == 0:
        return 'A 强阳性 (方差为 0)', d
    k = d / pooled
    if std_b > 0.08:
        return f'C 不稳定 (baseline std={std_b:.3f} 过大)', d
    if k >= 3:
        return f'A 强阳性 (Δ={d:+.4f}, {k:.1f}×pooled std)', d
    if k >= 1:
        return f'B 中等阳性 (Δ={d:+.4f}, {k:.1f}×pooled std, 论文须用 mean±std)', d
    if abs(d) < 0.02:
        return f'D +0.19 消失 (Δ={d:+.4f})', d
    return f'C 不稳定 (Δ={d:+.4f}, {k:.1f}×pooled std)', d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(EXP_ROOT, 'summary.json'))
    args = ap.parse_args()

    cfg_path = os.path.join(EXP_ROOT, 'config.json')
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cats = cfg.get('categories', ['fryum', 'pipe_fryum'])
    seeds = cfg.get('seeds', [0, 1, 2])

    rows = []       # 逐 run 明细 (方案 §二十六)
    agg = {}        # (cat, method) -> per-metric mean/std
    for cat in cats:
        for method in METHODS:
            per_metric = {}
            for m in METRICS:
                vals = []
                for s in seeds:
                    e = load(cat, method, s)
                    vals.append(e[m] if e else None)
                per_metric[m] = mean_std(vals)
            agg[(cat, method)] = per_metric
            for s in seeds:
                e = load(cat, method, s)
                rc = None
                rcp = os.path.join(EXP_ROOT, cat, f'{method}_seed{s}', 'run_config.json')
                if os.path.exists(rcp):
                    rc = json.load(open(rcp))
                rows.append({
                    'category': cat, 'method': method, 'seed': s,
                    'pixel_auc': e['pixel_auc'] if e else None,
                    'image_auc': e['image_auc'] if e else None,
                    'pixel_ap': e['pixel_ap'] if e else None,
                    'image_ap': e['image_ap'] if e else None,
                    'steps_per_epoch': rc.get('steps_per_epoch') if rc else None,
                    'training_time_s': rc.get('training_time_s') if rc else None,
                    'done': e is not None,
                })

    # 表 1: pixel AUC, mean ± std (方案 §4.6)
    print('=' * 78)
    print('G1 — Baseline Seed Variance (VisA, pixel AUC)')
    print('=' * 78)
    hdr = f"{'Category':<12}{'Method':<10}" + ''.join(f'{("seed"+str(s)):>10}' for s in seeds)
    hdr += f"{'Mean':>10}{'Std':>9}{'Δ(S8-Base)':>13}"
    print(hdr)
    verdicts = {}
    for cat in cats:
        for method in METHODS:
            vals = [load(cat, method, s) for s in seeds]
            cells = ''.join(
                f"{v['pixel_auc']:>10.4f}" if v else f"{'--':>10}" for v in vals)
            mu, sd, n = agg[(cat, method)]['pixel_auc']
            mus = f'{mu:.4f}' if mu is not None else '--'
            sds = f'{sd:.4f}' if sd is not None else '--'
            d = ''
            if method == 'S8':
                mb, _, _ = agg[(cat, 'baseline')]['pixel_auc']
                vd, dd = verdict(mb, agg[(cat, 'baseline')]['pixel_auc'][1],
                                 mu, sd, n)
                verdicts[cat] = vd
                d = f'{dd:+.4f}' if dd is not None else '--'
            print(f'{cat:<12}{method:<10}{cells}{mus:>10}{sds:>9}{d:>13}')
    print()
    for cat, vd in verdicts.items():
        print(f'  [{cat}] 判定: {vd}')

    # 表 2: 四指标 mean ± std
    print()
    print('=' * 78)
    print('四指标 mean ± std')
    print('=' * 78)
    print(f"{'Category':<12}{'Method':<10}{'pixel_auc':>18}{'image_auc':>18}"
          f"{'pixel_ap':>18}{'image_ap':>18}")
    for cat in cats:
        for method in METHODS:
            cells = ''
            for m in METRICS:
                mu, sd, n = agg[(cat, method)][m]
                cells += (f'{mu:.4f}±{sd:.4f}' if mu is not None else '--').rjust(18)
            print(f'{cat:<12}{method:<10}{cells}')

    # 表 3: 逐 run 明细 (方案 §二十六)
    print()
    print('=' * 78)
    print('逐 run 明细')
    print('=' * 78)
    print(f"{'category':<12}{'method':<10}{'seed':>5}{'pix_auc':>10}{'img_auc':>10}"
          f"{'pix_ap':>10}{'img_ap':>10}{'steps':>7}{'time(min)':>11}")
    for r in rows:
        t = f"{r['training_time_s']/60:.1f}" if r['training_time_s'] else '--'
        f_ = lambda k: f"{r[k]:.4f}" if r[k] is not None else '--'
        print(f"{r['category']:<12}{r['method']:<10}{r['seed']:>5}"
              f"{f_('pixel_auc'):>10}{f_('image_auc'):>10}{f_('pixel_ap'):>10}"
              f"{f_('image_ap'):>10}{str(r['steps_per_epoch'] or '--'):>7}{t:>11}")

    n_done = sum(1 for r in rows if r['done'])
    print(f"\n完成 {n_done}/{len(rows)} runs")

    out = {
        'experiment': 'G1_seed_variance',
        'config': cfg,
        'per_run': rows,
        'agg': {f'{c}|{m}': {k: {'mean': v[0], 'std': v[1], 'n': v[2]}
                             for k, v in agg[(c, m)].items()}
                for c in cats for m in METHODS},
        'verdict': verdicts,
    }
    json.dump(out, open(args.out, 'w'), indent=2, ensure_ascii=False)
    csv_path = args.out.replace('.json', '.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n-> {args.out}\n-> {csv_path}')


if __name__ == '__main__':
    main()
