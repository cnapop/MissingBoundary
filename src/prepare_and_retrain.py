#!/usr/bin/env python3
"""
Prepare augmented datasets and retrain DRAEM
===============================================
建立三个训练集:
  A) Baseline — 原始训练集
  B) Random Aug — 原始 + 随机 SeaS 样本
  C) Missing Boundary Aug — 原始 + Boundary Normal + Blind Defect

然后调用 DRAEM 训练脚本进行 retrain。
"""
import argparse, os, sys, json, csv, shutil, subprocess
import numpy as np
from glob import glob


def prepare_dataset_a(original_dir, target_dir, category):
    """
    Dataset A: Baseline — just copy original train/good
    """
    out_dir = os.path.join(target_dir, category, 'train', 'good')
    os.makedirs(out_dir, exist_ok=True)

    src_dir = os.path.join(original_dir, category, 'train', 'good')
    files = sorted(glob(os.path.join(src_dir, '*.png')))

    for f in files:
        shutil.copy2(f, os.path.join(out_dir, os.path.basename(f)))

    print(f"  Dataset A (Baseline): {len(files)} images → {out_dir}")
    return out_dir


def prepare_dataset_b(original_dir, seas_output_dir, target_dir, category,
                       random_seas_images=None):
    """
    Dataset B: Original + Random SeaS samples.
    If random_seas_images is None, use all available SeaS images.
    """
    out_good = os.path.join(target_dir, category, 'train', 'good')
    os.makedirs(out_good, exist_ok=True)

    # Copy original normal images
    src_dir = os.path.join(original_dir, category, 'train', 'good')
    for f in sorted(glob(os.path.join(src_dir, '*.png'))):
        shutil.copy2(f, os.path.join(out_good, os.path.basename(f)))

    # Copy random SeaS generated images as "anomaly" training data
    out_anom = os.path.join(target_dir, category, 'train', 'anomaly')
    os.makedirs(out_anom, exist_ok=True)

    if random_seas_images is not None:
        for src_path in random_seas_images:
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(out_anom, os.path.basename(src_path)))
        count = len(random_seas_images)
    else:
        # Use all available SeaS images across all generation types
        count = 0
        for subdir in ['boundary_normal', 'blind_defect']:
            img_dir = os.path.join(seas_output_dir, category, subdir, 'image')
            if os.path.isdir(img_dir):
                for f in sorted(glob(os.path.join(img_dir, '*.png'))):
                    shutil.copy2(f, os.path.join(out_anom, f"rand_{subdir}_{os.path.basename(f)}"))
                    count += 1

    print(f"  Dataset B (Random Aug): {len(os.listdir(out_good))} good + {count} anomaly → {target_dir}")
    return target_dir


def prepare_dataset_c(original_dir, seas_output_dir, target_dir, category,
                       bn_output_dir, bd_output_dir):
    """
    Dataset C: Original + Boundary Normal + Blind Defect samples
    """
    out_good = os.path.join(target_dir, category, 'train', 'good')
    out_anom = os.path.join(target_dir, category, 'train', 'anomaly')
    os.makedirs(out_good, exist_ok=True)
    os.makedirs(out_anom, exist_ok=True)

    # Copy original normal images
    src_dir = os.path.join(original_dir, category, 'train', 'good')
    for f in sorted(glob(os.path.join(src_dir, '*.png'))):
        shutil.copy2(f, os.path.join(out_good, os.path.basename(f)))
    normal_count = len(os.listdir(out_good))

    # Copy Boundary Normal generated images (treated as normal)
    if bn_output_dir:
        bn_img_dir = os.path.join(bn_output_dir, 'image')
        if os.path.isdir(bn_img_dir):
            for f in sorted(glob(os.path.join(bn_img_dir, '*.png'))):
                shutil.copy2(f, os.path.join(out_good, f"bn_{os.path.basename(f)}"))
    bn_count = len(os.listdir(out_good)) - normal_count

    # Copy Blind Defect generated images (treated as anomaly)
    defect_count = 0
    if bd_output_dir:
        bd_img_dir = os.path.join(bd_output_dir, 'image')
        if os.path.isdir(bd_img_dir):
            for f in sorted(glob(os.path.join(bd_img_dir, '*.png'))):
                shutil.copy2(f, os.path.join(out_anom, f"bd_{os.path.basename(f)}"))
                defect_count += 1

    print(f"  Dataset C (Missing Boundary Aug): {normal_count} good + {bn_count} BN + {defect_count} BD")
    return target_dir


def run_draem_training(draem_python, draem_dir, data_path, dtd_dir, checkpoint_dir,
                        category, lr, epochs, bs, gpu_id, run_name_suffix=''):
    """
    Run DRAEM training on a prepared dataset.
    We need to modify how DRAEM loads data to use our augmented dataset.
    """
    log_path = os.path.join(draem_dir, 'logs')
    base_name = f"DRAEM_mb_{lr}_{epochs}_bs{bs}{run_name_suffix}"

    cmd = [
        draem_python, os.path.join(draem_dir, 'train_DRAEM.py'),
        '--gpu_id', str(gpu_id),
        '--obj_id', '-1',  # Train all but we'll only have our category
        '--lr', str(lr),
        '--bs', str(bs),
        '--epochs', str(epochs),
        '--data_path', data_path,
        '--anomaly_source_path', dtd_dir,
        '--checkpoint_path', checkpoint_dir,
        '--log_path', log_path,
    ]

    print(f"\n  Running DRAEM training: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  DRAEM training ERROR: {result.stderr[:500]}")
        print(f"  stdout: {result.stdout[:500]}")
        return None

    print(f"  DRAEM training completed")
    print(f"  stdout (last 500): {result.stdout[-500:]}")
    return base_name


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets and retrain DRAEM')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Original MVTec dataset path')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seas_output', type=str, required=True,
                        help='SeaS generation output directory')
    parser.add_argument('--bn_output', type=str, default=None,
                        help='Boundary Normal output (from generate_samples.py)')
    parser.add_argument('--bd_output', type=str, default=None,
                        help='Blind Defect output (from generate_samples.py)')
    parser.add_argument('--draem_dir', type=str, required=True)
    parser.add_argument('--draem_python', type=str, required=True)
    parser.add_argument('--dtd_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--skip_retrain', action='store_true')
    args = parser.parse_args()

    # Prepare datasets
    print("=" * 60)
    print("Preparing Datasets")
    print("=" * 60)

    datasets = {
        'baseline': os.path.join(args.output_dir, 'datasets', 'A_baseline'),
        'random_aug': os.path.join(args.output_dir, 'datasets', 'B_random'),
        'mb_aug': os.path.join(args.output_dir, 'datasets', 'C_mb'),
    }

    # Dataset A: Baseline (just ensure it exists)
    prepare_dataset_a(args.dataset_dir, datasets['baseline'], args.category)

    # Dataset B: Random Aug
    prepare_dataset_b(args.dataset_dir, args.seas_output,
                      datasets['random_aug'], args.category)

    # Dataset C: Missing Boundary Aug
    prepare_dataset_c(args.dataset_dir, args.seas_output,
                      datasets['mb_aug'], args.category,
                      args.bn_output, args.bd_output)

    print(f"\nDatasets prepared:")
    for name, path in datasets.items():
        good_dir = os.path.join(path, args.category, 'train', 'good')
        anom_dir = os.path.join(path, args.category, 'train', 'anomaly')
        n_good = len(glob(os.path.join(good_dir, '*.png'))) if os.path.isdir(good_dir) else 0
        n_anom = len(glob(os.path.join(anom_dir, '*.png'))) if os.path.isdir(anom_dir) else 0
        print(f"  {name}: {n_good} good + {n_anom} anomaly")

    if args.skip_retrain:
        print("\nSkipping retraining (--skip_retrain)")
        return

    # Retrain DRAEM for each dataset
    print("\n" + "=" * 60)
    print("Retraining DRAEM")
    print("=" * 60)

    for ds_name, ds_path in datasets.items():
        print(f"\n{'='*40}")
        print(f"Retraining on: {ds_name}")
        print(f"{'='*40}")

        train_dir = os.path.join(ds_path, args.category, 'train')
        if not os.path.isdir(train_dir):
            print(f"  WARNING: training dir not found: {train_dir}, skipping")
            continue

        run_draem_training(
            args.draem_python, args.draem_dir,
            ds_path, args.dtd_dir,
            args.checkpoint_dir,
            args.category, args.lr, args.epochs, args.bs, args.gpu_id,
            run_name_suffix=f'_{ds_name}',
        )

    print("\nRetraining complete!")


if __name__ == '__main__':
    main()
