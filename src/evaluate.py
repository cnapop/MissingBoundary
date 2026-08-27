#!/usr/bin/env python3
"""
Evaluate and Compare DRAEM models (v2)
=========================================
对比: Baseline vs Random Aug vs Missing Boundary Aug

每个模型用各自的 checkpoint 目录和 base_model_name 测试。
"""
import argparse, os, sys, json, csv, subprocess
import numpy as np


def run_draem_test(draem_python, draem_dir, base_model_name, data_path, ckpt_path, gpu_id, category):
    """
    Run DRAEM test_DRAEM.py for a single category.
    base_model_name is the FULL name including category suffix,
    e.g. 'DRAEM_test_0.0001_700_bs8_bottle_'
    """
    # Ensure test script has only our category
    cmd = [
        draem_python, os.path.join(draem_dir, 'test_DRAEM.py'),
        '--gpu_id', str(gpu_id),
        '--base_model_name', base_model_name,
        '--data_path', data_path,
        '--checkpoint_path', ckpt_path,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=draem_dir)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}")
        print(f"  stdout tail: {result.stdout[-300:]}")
        return None

    # Parse metrics from stdout
    metrics = {}
    for line in result.stdout.split('\n'):
        line = line.strip()
        for metric, key in [('AUC Image', 'AUC Image'), ('AP Image', 'AP Image'),
                             ('AUC Pixel', 'AUC Pixel'), ('AP Pixel', 'AP Pixel')]:
            if line.startswith(metric + ':'):
                try:
                    metrics[key] = float(line.split(':')[1].strip())
                except (IndexError, ValueError):
                    pass
            elif line.startswith(metric + ' '):
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        metrics[key] = float(parts[1].strip())
                    except ValueError:
                        pass
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare DRAEM models')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--draem_dir', type=str, required=True)
    parser.add_argument('--draem_python', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--baseline_ckpt', type=str, required=True,
                        help='Baseline checkpoint dir')
    parser.add_argument('--baseline_base', type=str, required=True,
                        help='Baseline base_model_name e.g. DRAEM_test_0.0001_700_bs8')
    parser.add_argument('--mb_ckpt', type=str, default=None,
                        help='MB Aug checkpoint dir')
    parser.add_argument('--mb_base', type=str, default=None,
                        help='MB Aug base_model_name')
    parser.add_argument('--random_ckpt', type=str, default=None,
                        help='Random Aug checkpoint dir')
    parser.add_argument('--random_base', type=str, default=None,
                        help='Random Aug base_model_name')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Define models to evaluate
    models = [
        ('Baseline', args.baseline_ckpt, args.baseline_base),
    ]
    if args.mb_ckpt and args.mb_base:
        models.append(('Missing Boundary', args.mb_ckpt, args.mb_base))
    if args.random_ckpt and args.random_base:
        models.append(('Random Aug', args.random_ckpt, args.random_base))

    # Verify checkpoints exist
    print("Checking checkpoints:")
    for name, ckpt, base in models:
        rec = os.path.join(ckpt, f'{base}_{args.category}_.pckl')
        seg = os.path.join(ckpt, f'{base}_{args.category}__seg.pckl')
        print(f"  {name}: rec={'OK' if os.path.exists(rec) else 'MISSING'} "
              f"seg={'OK' if os.path.exists(seg) else 'MISSING'}")
        if not (os.path.exists(rec) and os.path.exists(seg)):
            print(f"    -> skipping {name}")
            continue

        print(f"\n--- Evaluating {name} ---")
        full_base = f'{base}_{args.category}_'
        metrics = run_draem_test(
            args.draem_python, args.draem_dir, full_base,
            args.data_path, ckpt, args.gpu_id, args.category
        )
        if metrics:
            print(f"  {name}: {metrics}")

    # Note: test_DRAEM.py prints per-category and mean results.
    # We parse from the results.txt it appends to instead for reliability.
    results = {}
    results_file = os.path.join(args.draem_dir, 'outputs', 'results.txt')
    if os.path.isfile(results_file):
        with open(results_file, 'r') as f:
            content = f.read()
        # Parse the LAST occurrence of each model's metrics
        blocks = content.strip().split('--------------------------')
        for name, ckpt, base in models:
            full_base = f'{base}_{args.category}_'
            for block in reversed(blocks):
                if full_base in block:
                    metrics = {}
                    for line in block.split('\n'):
                        parts = line.split(',')
                        if len(parts) >= 3:
                            key, rname = parts[0], parts[1]
                            if rname.strip() == full_base:
                                vals = [float(v) for v in parts[2:] if v]
                                if vals:
                                    metric_map = {
                                        'img_auc': 'Image AUROC',
                                        'pixel_auc': 'Pixel AUROC',
                                        'img_ap': 'Image AP',
                                        'pixel_ap': 'Pixel AP',
                                    }
                                    metrics[metric_map[key]] = np.mean(vals)
                    if metrics:
                        results[name] = metrics
                        print(f"\n{name} (from results.txt): {metrics}")
                    break

    # Also re-run test for each model to get clean metrics
    print("\n" + "=" * 60)
    print("Re-running tests for reliable metrics")
    print("=" * 60)

    for name, ckpt, base in models:
        rec = os.path.join(ckpt, f'{base}_{args.category}_.pckl')
        seg = os.path.join(ckpt, f'{base}_{args.category}__seg.pckl')
        if not (os.path.exists(rec) and os.path.exists(seg)):
            continue
        full_base = f'{base}_{args.category}_'
        print(f"\n--- {name} ---")
        metrics = run_draem_test(
            args.draem_python, args.draem_dir, full_base,
            args.data_path, ckpt, args.gpu_id, args.category
        )
        if metrics:
            results[name] = metrics
            print(f"  Parsed: {metrics}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)

    header = f"{'Method':<20} | {'Img AUROC':<10} | {'Pix AUROC':<10} | {'Pix AP':<10} | {'Img AP':<10}"
    print(header)
    print("-" * len(header))

    for name, _, _ in models:
        if name in results:
            m = results[name]
            print(f"{name:<20} | {m.get('Image AUROC', 0):<10.4f} | "
                  f"{m.get('Pixel AUROC', 0):<10.4f} | {m.get('Pixel AP', 0):<10.4f} | "
                  f"{m.get('Image AP', 0):<10.4f}")
        else:
            print(f"{name:<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")

    # Save
    result_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(args.output_dir, 'comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Image AUROC', 'Pixel AUROC', 'Image AP', 'Pixel AP'])
        for name, _, _ in models:
            if name in results:
                m = results[name]
                writer.writerow([name, m.get('Image AUROC', ''), m.get('Pixel AUROC', ''),
                                 m.get('Image AP', ''), m.get('Pixel AP', '')])

    print(f"\nResults saved to: {result_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
