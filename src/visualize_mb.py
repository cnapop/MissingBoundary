#!/usr/bin/env python3
"""
Missing Boundary Visualization
================================
生成图2: anomaly score / Gap map / PB map / M(x)
并可视化 top-M 样本
"""
import argparse, os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image


def load_mb_results(csv_path):
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'img_name': row['img_name'],
                'anomaly_score': float(row['anomaly_score']),
                'pb': float(row['pb']),
                'gap_mean': float(row['gap_mean']),
                'M': float(row['M']),
                'split': row['split'],
            })
    return results


def load_amap(amap_dir, img_name):
    """Load anomaly map matching img_name."""
    stem = img_name.replace('/', '_').replace('.png', '').replace('.jpg', '')
    candidates = glob(os.path.join(amap_dir, f'{stem}_amap.npy'))
    if not candidates:
        # Try fuzzy match
        for f in glob(os.path.join(amap_dir, '*_amap.npy')):
            if stem in os.path.basename(f):
                candidates.append(f)
    if candidates:
        return np.load(candidates[0])
    return None


def resolve_image_path(dataset_dir, category, img_name):
    """Resolve image path from various name formats."""
    if os.path.exists(img_name):
        return img_name
    # Try direct path
    full = os.path.join(dataset_dir, category, img_name)
    if os.path.exists(full):
        return full
    # Try underscore format: train_good_203
    if '_' in img_name:
        parts = img_name.split('_')
        # Try train/good/X.png
        if parts[0] == 'train' and parts[1] == 'good':
            p = os.path.join(dataset_dir, category, 'train', 'good', f'{parts[-1]}.png')
            if os.path.exists(p):
                return p
        if parts[0] == 'test' and len(parts) >= 3:
            # defect type may contain underscores (e.g. broken_small)
            idx = parts[-1]
            defect = '_'.join(parts[1:-1])
            p = os.path.join(dataset_dir, category, 'test', defect, f'{idx}.png')
            if os.path.exists(p):
                return p
    return None


def plot_image_with_amap(image, amap, title, ax):
    ax.imshow(image)
    if amap is not None:
        ax.imshow(amap, cmap='jet', alpha=0.5)
    ax.set_title(title)
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser(description='Visualize Missing Boundary results')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='/data/chenjiawen/Datasets/MVTec-AD')
    parser.add_argument('--top_k', type=int, default=6)
    args = parser.parse_args()

    mb_dir = os.path.join(args.output_dir, args.category, 'missing_boundary')
    results = load_mb_results(os.path.join(mb_dir, 'missing_boundary.csv'))

    # Split into test and train
    test_results = [r for r in results if r['split'] == 'test']
    train_results = [r for r in results if r['split'] == 'train_good']

    # Top-M by each metric
    test_by_m = sorted(test_results, key=lambda r: r['M'], reverse=True)
    test_by_gap = sorted(test_results, key=lambda r: r['gap_mean'], reverse=True)
    test_by_pb = sorted(test_results, key=lambda r: r['pb'], reverse=True)
    train_by_m = sorted(train_results, key=lambda r: r['M'], reverse=True)

    vis_dir = os.path.join(mb_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # ===== Figure 2a: Top-M test images with anomaly maps =====
    fig, axes = plt.subplots(4, args.top_k, figsize=(args.top_k * 3, 12))
    fig.suptitle(f'Missing Boundary: Top-{args.top_k} Test Images by M(x)', fontsize=14)

    for col, r in enumerate(test_by_m[:args.top_k]):
        img_path = resolve_image_path(args.dataset_dir, args.category, r['img_name'])
        if img_path is None:
            print(f"  WARNING: cannot resolve {r['img_name']}")
            continue
        image = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))

        amap_dir = os.path.join(args.output_dir, args.category, 'test', 'anomaly_maps')
        amap = load_amap(amap_dir, r['img_name'])

        axes[0, col].imshow(image)
        axes[0, col].set_title(f"{os.path.basename(img_path)}\nM={r['M']:.3f}", fontsize=8)
        axes[0, col].axis('off')

        if amap is not None:
            axes[1, col].imshow(image)
            axes[1, col].imshow(amap, cmap='jet', alpha=0.6)
            axes[1, col].set_title(f"Anomaly map\nscore={r['anomaly_score']:.3f}", fontsize=8)
            axes[1, col].axis('off')
        else:
            axes[1, col].text(0.5, 0.5, 'No amap', ha='center')
            axes[1, col].axis('off')

        # PB map
        pb = 1 - 2 * np.abs(amap - 0.5) if amap is not None else None
        axes[2, col].imshow(image)
        if pb is not None:
            axes[2, col].imshow(pb, cmap='viridis', alpha=0.6)
        axes[2, col].set_title(f"PB map\nPB={r['pb']:.3f}", fontsize=8)
        axes[2, col].axis('off')

        # Gap placeholder (we have image-level gap)
        axes[3, col].imshow(image)
        axes[3, col].set_title(f"Gap={r['gap_mean']:.3f}\n" +
                                (f"std={r['gap_max']:.3f}" if 'gap_max' in r else ""), fontsize=8)
        axes[3, col].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(vis_dir, 'top_m_test.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # ===== Figure 2b: Top-M train (Boundary Normal) images =====
    fig, axes = plt.subplots(2, args.top_k, figsize=(args.top_k * 3, 6))
    fig.suptitle(f'Boundary Normal Candidates: Top-{args.top_k} Train Images by M(x)', fontsize=14)

    for col, r in enumerate(train_by_m[:args.top_k]):
        img_path = resolve_image_path(args.dataset_dir, args.category, r['img_name'])
        if img_path is None:
            print(f"  WARNING: cannot resolve {r['img_name']}")
            continue
        image = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
        axes[0, col].imshow(image)
        axes[0, col].set_title(f"M={r['M']:.3f}", fontsize=8)
        axes[0, col].axis('off')

        amap_dir = os.path.join(args.output_dir, args.category, 'train_good', 'anomaly_maps')
        amap = load_amap(amap_dir, r['img_name'])
        axes[1, col].imshow(image)
        if amap is not None:
            axes[1, col].imshow(amap, cmap='jet', alpha=0.6)
        axes[1, col].set_title(f"score={r['anomaly_score']:.3f}", fontsize=8)
        axes[1, col].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(vis_dir, 'boundary_normal_candidates.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # ===== Figure 2c: Score distribution =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Missing Boundary Score Distributions (bottle)', fontsize=14)

    test_scores = [r['anomaly_score'] for r in test_results]
    train_scores = [r['anomaly_score'] for r in train_results]

    axes[0].hist(test_scores, bins=30, alpha=0.7, label='test')
    axes[0].hist(train_scores, bins=30, alpha=0.5, label='train/good')
    axes[0].set_xlabel('DRAEM anomaly score')
    axes[0].set_ylabel('count')
    axes[0].legend()
    axes[0].set_title('Anomaly Score')

    test_pbs = [r['pb'] for r in test_results]
    train_pbs = [r['pb'] for r in train_results]
    axes[1].hist(test_pbs, bins=30, alpha=0.7, label='test')
    axes[1].hist(train_pbs, bins=30, alpha=0.5, label='train/good')
    axes[1].set_xlabel('PB')
    axes[1].legend()
    axes[1].set_title('Probabilistic Boundary')

    test_gaps = [r['gap_mean'] for r in test_results]
    train_gaps = [r['gap_mean'] for r in train_results]
    axes[2].hist(test_gaps, bins=30, alpha=0.7, label='test')
    axes[2].hist(train_gaps, bins=30, alpha=0.5, label='train/good')
    axes[2].set_xlabel('Gap')
    axes[2].legend()
    axes[2].set_title('Density Gap')

    plt.tight_layout()
    fig_path = os.path.join(vis_dir, 'score_distributions.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    print(f"\nAll visualizations saved to: {vis_dir}")


if __name__ == '__main__':
    main()
