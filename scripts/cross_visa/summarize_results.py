#!/usr/bin/env python3
"""
summarize_results.py — 汇总跨类交叉验证结果
=============================================
读取各类别 selection_g3_{cat}/selection_results.json + cross_visa/baseline/{cat}.json,
输出对比表 (含 baseline) 并保存汇总 JSON/CSV。

用法:
  {PY} scripts/cross_visa/summarize_results.py [--categories ...] [--out outputs/cross_visa/summary.json]
"""
import argparse, os, json, csv

ROOT = '/data/chenjiawen/MissingBoundary'
BASELINE_OUT = os.path.join(ROOT, 'outputs', 'cross_visa', 'baseline')

ALL_CATS = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
            'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


def load_baseline(cat):
    p = os.path.join(BASELINE_OUT, f'{cat}.json')
    if os.path.exists(p):
        return json.load(open(p))
    return None


def load_results(cat):
    p = os.path.join(ROOT, 'outputs', f'selection_g3_{cat}', 'selection_results.json')
    if os.path.exists(p):
        return json.load(open(p))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', type=str, default=None)
    ap.add_argument('--out', type=str, default=os.path.join(ROOT, 'outputs', 'cross_visa', 'summary.json'))
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(',')] if args.categories else ALL_CATS

    rows = []  # each: {cat, selector, image_auc, image_ap, pixel_auc, pixel_ap, dpix}
    for cat in cats:
        base = load_baseline(cat)
        res = load_results(cat)
        if not res:
            print(f'[skip] {cat}: no selection_results.json')
            continue
        print(f'\n=== {cat} ===')
        print(f"{'selector':<12}{'Img_AUC':>9}{'Img_AP':>9}{'Pix_AUC':>9}{'Pix_AP':>9}{'ΔPix_AUC':>10}")
        if base:
            print(f"{'baseline':<12}{base.get('image_auc',0):>9.4f}{base.get('image_ap',0):>9.4f}"
                  f"{base.get('pixel_auc',0):>9.4f}{base.get('pixel_ap',0):>9.4f}{'--':>10}")
            rows.append({'cat': cat, 'selector': 'baseline', **{k: base.get(k, 0) for k in
                        ('image_auc', 'image_ap', 'pixel_auc', 'pixel_ap')}, 'dpix': 0.0})
        for name, m in res.items():
            if not m:
                continue
            d = (m['pixel_auc'] - (base['pixel_auc'] if base else 0))
            print(f"{name:<12}{m['image_auc']:>9.4f}{m['image_ap']:>9.4f}"
                  f"{m['pixel_auc']:>9.4f}{m['pixel_ap']:>9.4f}{d:>+10.4f}")
            rows.append({'cat': cat, 'selector': name, 'image_auc': m['image_auc'],
                         'image_ap': m['image_ap'], 'pixel_auc': m['pixel_auc'],
                         'pixel_ap': m['pixel_ap'], 'dpix': d})

    if rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump({'rows': rows, 'categories': cats}, open(args.out, 'w'), indent=2)
        csv_path = args.out.replace('.json', '.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['cat', 'selector', 'image_auc', 'image_ap',
                                              'pixel_auc', 'pixel_ap', 'dpix'])
            w.writeheader()
            w.writerows(rows)
        print(f'\nsummary -> {args.out} / {csv_path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
