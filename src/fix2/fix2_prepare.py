#!/usr/bin/env python3
"""
fix2 数据准备
==============
从 MB v3 闭环接受样本构建 fix2 双流训练所需数据:
  1. train_good_plus_bn/ — 原始 bottle/train/good (209) + BN 接受样本 (8) → 217 张
  2. bd_manifest.csv — BD 接受样本 manifest (image/mask/reference/type/utility),
     供 train_draem_fix2.py 的 SelectedSeaSDataset 消费。

用法: python src/fix2_prepare.py --category bottle
"""
import argparse, os, shutil, csv, glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--category', default='bottle')
    p.add_argument('--mvtec_root', default='/data/chenjiawen/Datasets/MVTec-AD')
    p.add_argument('--gen_root', default=None, help='generated_v3 根目录 (默认 outputs/generated_v3; VisA 用 outputs/generated_v3_visa)')
    p.add_argument('--out_root', default=None, help='fix2 输出根 (默认 outputs/fix2)')
    p.add_argument('--normal_src_dir', default=None,
                   help='原始 normal 目录 (默认 {mvtec_root}/{cat}/train/good; VisA 用 outputs/visa_datasets/{cat}/train/good)')
    p.add_argument('--bn_max', type=int, default=None,
                   help='BN 接受样本子集上限 (等距抽样, 用于 BN 比例消融; 默认全量)')
    args = p.parse_args()

    cat = args.category
    gen_root = args.gen_root or os.path.join(ROOT, 'outputs', 'generated_v3')
    out_root = args.out_root or os.path.join(ROOT, 'outputs', 'fix2')
    gen_cat = os.path.join(gen_root, cat)
    out_cat = os.path.join(out_root, cat)
    normal_dir = os.path.join(out_cat, 'train_good_plus_bn')
    os.makedirs(normal_dir, exist_ok=True)

    # ---- 1. 原始 normal + BN 接受样本 → train_good_plus_bn ----
    norm_src = args.normal_src_dir or os.path.join(args.mvtec_root, cat, 'train', 'good')
    src_normals = sorted(glob.glob(os.path.join(norm_src, '*.png')))
    bn_accs_all = sorted(glob.glob(os.path.join(gen_cat, 'boundary_normal', '*', 'acc_[0-9][0-9].png')))
    bn_accs = bn_accs_all
    if args.bn_max is not None and len(bn_accs_all) > args.bn_max:
        # 等距抽样（覆盖全区间）: 0..len-1 均匀取 bn_max 个索引
        idx = [int(round(i * (len(bn_accs_all) - 1) / (args.bn_max - 1))) for i in range(args.bn_max)]
        bn_accs = [bn_accs_all[i] for i in sorted(set(idx))]
        print(f'  [ablation] BN 等距抽 {len(bn_accs)}/{len(bn_accs_all)} (bn_max={args.bn_max})')
    print(f'原始 normal: {len(src_normals)}, BN 接受: {len(bn_accs)}')
    n_copied = 0
    for f in src_normals:
        shutil.copy2(f, os.path.join(normal_dir, os.path.basename(f)))
        n_copied += 1
    for i, f in enumerate(bn_accs):
        shutil.copy2(f, os.path.join(normal_dir, f'mb_bn_{i:02d}.png'))
        n_copied += 1
    print(f'  → {normal_dir}: {n_copied} 张')

    # ---- 2. BD 接受样本 → manifest ----
    bd_accs = sorted(glob.glob(os.path.join(gen_cat, 'blind_defect', '*', 'acc_[0-9][0-9].png')))
    # refs 以 "sample_dir/saved_as" 为 key —— 不同 sample 目录的 acc_XX.png 名字相同，
    # 只按文件名会互相覆盖（多类别/多样本目录时必须带上 sample 目录）。
    refs = {}
    for meta in glob.glob(os.path.join(gen_cat, 'blind_defect', '*', 'meta.json')):
        import json
        m = json.load(open(meta))
        sample_dir = os.path.basename(os.path.dirname(meta))
        for r in m['accepted']:
            refs[f"{sample_dir}/{r.get('saved_as') or os.path.basename(r['img'])}"] = m['source_path']
    print(f'BD 接受: {len(bd_accs)}, ref 映射: {len(refs)}')

    manifest = os.path.join(out_cat, 'bd_manifest.csv')
    with open(manifest, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'mask', 'reference', 'type', 'utility'])
        for a in bd_accs:
            sample_dir = os.path.basename(os.path.dirname(a))
            mask = a.replace('.png', '_mask.png')
            ref = refs.get(f"{sample_dir}/{os.path.basename(a)}")
            if ref is None or not os.path.exists(mask):
                print(f'  [skip] {a}: mask={os.path.exists(mask)} ref={ref}')
                continue
            w.writerow([os.path.abspath(a), os.path.abspath(mask),
                        os.path.abspath(ref), 'defect', '1.0'])
    print(f'  → {manifest}: {sum(1 for _ in open(manifest)) - 1} 行')


if __name__ == '__main__':
    main()
