#!/usr/bin/env python3
"""
VisA pipe_fryum 缺陷子类 → SeaS 训练布局
==========================================
不归并组合标签, 直接把 image_anno.csv 的逐图缺陷类型作为子类分桶, 构建
SeaS 多子类训练布局:

  outputs/visa_seas_subtype/{cat}/normal/              (450 张, 复用 visa_seas 的 normal)
  outputs/visa_seas_subtype/{cat}/instance/{type}/     (每子类从 000 密集重编号)
  outputs/visa_seas_subtype/{cat}/mask/{type}/         (与 instance 同名, VisA 约定)

SeaS 的 anomaly_categories_num = instance 下非 good 子目录数; 每子类分到 4 个
sks token (字母序), 生成时用对应 sks 块即可定向出该类型缺陷.

用法:
  python src/closedloop_v3/prepare_visa_subtype.py --category pipe_fryum
"""
import argparse, os, json, csv, re, shutil
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sanitize_label(label):
    """'corner and edge breakage' → 'corner_and_edge_breakage';
    'burnt,small scratches' → 'burnt_small_scratches'."""
    s = label.strip()
    s = s.replace(',', '_')
    s = s.replace(' ', '_')
    s = re.sub(r'_+', '_', s)
    return s


def convert_image(src, dst):
    """JPG/PNG → PNG (RGB)."""
    img = Image.open(src)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(dst)


def load_anno(visa_root, category):
    """返回 anomaly rows: list of dict(image, label, mask)."""
    anno = os.path.join(visa_root, category, 'image_anno.csv')
    with open(anno) as f:
        rows = list(csv.DictReader(f))
    anom = [r for r in rows if r['label'].strip().lower() not in ('normal', 'good')]
    return anom


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--visa_root', type=str, default='/data/chenjiawen/Datasets/VisA')
    p.add_argument('--category', type=str, required=True)
    p.add_argument('--out_seas', type=str, default=os.path.join(ROOT, 'outputs', 'visa_seas_subtype'))
    p.add_argument('--existing_seas_normal', type=str,
                   default=os.path.join(ROOT, 'outputs', 'visa_seas', '{cat}', 'normal'),
                   help='既有 visa_seas/{cat}/normal (450 张 train 正常图), 直接复用')
    args = p.parse_args()

    cat = args.category
    anom = load_anno(args.visa_root, cat)
    print(f'{cat}: {len(anom)} anomaly rows with subtypes')

    seas = os.path.join(args.out_seas, cat)
    seas_normal = os.path.join(seas, 'normal')
    seas_instance_root = os.path.join(seas, 'instance')
    seas_mask_root = os.path.join(seas, 'mask')
    os.makedirs(seas_normal, exist_ok=True)
    os.makedirs(seas_instance_root, exist_ok=True)
    os.makedirs(seas_mask_root, exist_ok=True)

    # ---- normal: 复用既有 visa_seas/{cat}/normal (与 v1/v2/v3 训练一致) ----
    src_normal = args.existing_seas_normal.format(cat=cat)
    if os.path.isdir(src_normal):
        for f in sorted(os.listdir(src_normal)):
            if f.endswith('.png'):
                shutil.copy2(os.path.join(src_normal, f), os.path.join(seas_normal, f))
        n_normal = len(os.listdir(seas_normal))
        print(f'  normal ← {src_normal}: {n_normal} 张')
    else:
        raise FileNotFoundError(f'既有 normal 目录不存在: {src_normal}')

    # ---- anomaly: 按 label 分桶, 每子类从 000 密集重编号 ----
    # 先确定子目录字母序 → sks 块映射 (与 SeaS sorted(iterdir) 一致)
    groups = {}
    for r in anom:
        typ = sanitize_label(r['label'])
        groups.setdefault(typ, []).append(r)
    types_sorted = sorted(groups.keys())

    report = {'category': cat, 'normal_n': n_normal, 'subtypes': {}, 'sks_map': {}}
    for i, typ in enumerate(types_sorted):
        rows = groups[typ]
        inst_dir = os.path.join(seas_instance_root, typ)
        mask_dir = os.path.join(seas_mask_root, typ)
        os.makedirs(inst_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        sks_start = i * 4 + 1
        report['sks_map'][typ] = f'sks{sks_start}-sks{sks_start+3}'
        missing_mask = 0
        for k, r in enumerate(rows):
            idx = f'{k:03d}'
            img_src = os.path.join(args.visa_root, r['image'])
            convert_image(img_src, os.path.join(inst_dir, f'{idx}.png'))
            m = r['mask'] if r['mask'] else ''
            if m and os.path.exists(os.path.join(args.visa_root, m)):
                shutil.copy2(os.path.join(args.visa_root, m), os.path.join(mask_dir, f'{idx}.png'))
            else:
                missing_mask += 1
        report['subtypes'][typ] = {'n': len(rows), 'sks': report['sks_map'][typ],
                                   'missing_mask': missing_mask}
        print(f'  [{i:02d}] {typ}: {len(rows)} 图, sks{sks_start}-sks{sks_start+3}'
              + (f' (缺 mask {missing_mask})' if missing_mask else ''))

    total = sum(v['n'] for v in report['subtypes'].values())
    report['anomaly_n'] = total
    report['subtype_n'] = len(types_sorted)
    out_json = os.path.join(args.out_seas, f'{cat}_subtype_report.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('=' * 60)
    print(f'共 {total} 张 anomaly, {len(types_sorted)} 个子类 → {seas}')
    print(f'报告: {out_json}')


if __name__ == '__main__':
    main()
