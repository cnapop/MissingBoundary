#!/usr/bin/env python3
"""
VisA → MVTec / SeaS 布局转换
==============================
把 VisA 类别 (pipe_fryum) 从原始布局转成两个消费端需要的布局:

  MVTec 布局 (DRAEM 训练 + 特征提取):
    outputs/visa_datasets/{cat}/train/good/*.png        (450 train normal)
    outputs/visa_datasets/{cat}/test/good/*.png         (50 test normal)
    outputs/visa_datasets/{cat}/test/anomaly/*.png      (100 test anomaly)
    outputs/visa_datasets/{cat}/test/anomaly/{idx}_mask.png  (像素评估 mask)

  SeaS 训练布局 (visa_seas/{cat}/):
    normal/                    (450 平铺)
    instance/{cat}_bad/        (100 图, 命名 {idx:03}.png, 与 mask 索引对齐)
    mask/{cat}_bad/            (100 mask, 命名 {idx:03}.png)

VisA anomaly 无公开缺陷子类 → 单一子类 ({cat}_bad), SeaS 用 sks1-sks4 四 token,
与闭环 BD prompt "a ob1 with sks1 sks2 sks3 sks4" 一致.
"""
import argparse, os, json, csv
import shutil
from PIL import Image


def load_split(split_csv, category):
    """返回 rows: list of dict(object, split, label, image, mask)."""
    rows = []
    with open(split_csv) as f:
        for r in csv.DictReader(f):
            if r['object'] == category:
                rows.append(r)
    return rows


def convert_image(src, dst):
    """JPG/PNG → PNG (RGB)."""
    img = Image.open(src)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(dst)


def main():
    parser = argparse.ArgumentParser(description='Convert VisA category to MVTec + SeaS layouts')
    parser.add_argument('--visa_root', type=str, default='/data/chenjiawen/Datasets/VisA')
    parser.add_argument('--split_csv', type=str, default=None)
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--out_datasets', type=str, default='outputs/visa_datasets')
    parser.add_argument('--out_seas', type=str, default='outputs/visa_seas')
    parser.add_argument('--skip_mask_copy', action='store_true')
    args = parser.parse_args()

    if args.split_csv is None:
        args.split_csv = os.path.join(args.visa_root, 'split_csv', '1cls.csv')
    cat = args.category

    rows = load_split(args.split_csv, cat)
    report = {'category': cat, 'train_normal': 0, 'test_normal': 0, 'test_anomaly': 0}

    # 目录
    mvtec = os.path.join(args.out_datasets, cat)
    for sub in ['train/good', 'test/good', 'test/anomaly']:
        os.makedirs(os.path.join(mvtec, sub), exist_ok=True)
    seas = os.path.join(args.out_seas, cat)
    seas_normal = os.path.join(seas, 'normal')
    seas_instance = os.path.join(seas, 'instance', f'{cat}_bad')
    seas_mask = os.path.join(seas, 'mask', f'{cat}_bad')
    for d in [seas_normal, seas_instance, seas_mask]:
        os.makedirs(d, exist_ok=True)

    counters = {'train_good': 0, 'test_good': 0, 'test_anomaly': 0}

    def img_src(row):
        return os.path.join(args.visa_root, row['image'])

    def mask_src(row):
        return os.path.join(args.visa_root, row['mask']) if row['mask'] else None

    for row in rows:
        if row['split'] == 'train' and row['label'] == 'normal':
            dst = os.path.join(mvtec, 'train/good', f"{counters['train_good']:03d}.png")
            convert_image(img_src(row), dst)
            shutil.copy2(dst, os.path.join(seas_normal, f"{counters['train_good']:03d}.png"))
            counters['train_good'] += 1
            report['train_normal'] += 1

        elif row['split'] == 'test' and row['label'] == 'normal':
            dst = os.path.join(mvtec, 'test/good', f"{counters['test_good']:03d}.png")
            convert_image(img_src(row), dst)
            counters['test_good'] += 1
            report['test_normal'] += 1

        elif row['split'] == 'test' and row['label'] == 'anomaly':
            # 用源文件名索引 (与 mask 对齐): e.g. Anomaly/042.JPG → 042
            idx = os.path.splitext(os.path.basename(row['image']))[0]
            dst = os.path.join(mvtec, 'test/anomaly', f"{idx}.png")
            convert_image(img_src(row), dst)
            m = mask_src(row)
            if m and os.path.exists(m) and not args.skip_mask_copy:
                shutil.copy2(m, os.path.join(mvtec, 'test/anomaly', f"{idx}_mask.png"))
                shutil.copy2(m, os.path.join(seas_mask, f"{idx}.png"))
            shutil.copy2(dst, os.path.join(seas_instance, f"{idx}.png"))
            counters['test_anomaly'] += 1
            report['test_anomaly'] += 1
        else:
            print(f"  [skip] {row}")

    # 报告
    report['mvtec'] = mvtec
    report['seas'] = seas
    report['seas_instance_subdir'] = f'{cat}_bad'
    report['instance_n'] = counters['test_anomaly']
    report['normal_n'] = counters['train_good']
    out_json = os.path.join(args.out_datasets, f'{cat}_conversion_report.json')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print(f"VisA {cat} 转换完成")
    print(f"  train_normal: {report['train_normal']}")
    print(f"  test_normal:  {report['test_normal']}")
    print(f"  test_anomaly: {report['test_anomaly']}")
    print(f"  MVTec 布局:  {mvtec}")
    print(f"  SeaS 布局:   {seas}")
    print(f"  报告:        {out_json}")


if __name__ == '__main__':
    main()
