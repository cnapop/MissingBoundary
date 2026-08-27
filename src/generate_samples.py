#!/usr/bin/env python3
"""
SeaS Missing Boundary Guided Generation
=========================================
利用 M(x) 结果引导 SeaS 生成两类困难样本:

  Task A: Boundary Normal — 正常但靠近决策边界 → 降低 FP
  Task B: Blind Defect — 微小/不易检测的异常 → 降低 FN

执行方式: 通过 subprocess 调用 seas 环境下的 SeaS_infer.py
"""
import argparse, os, sys, json, shutil, subprocess, csv
import numpy as np
from glob import glob


def load_high_m_regions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def copy_reference_images(reference_paths, target_dir, prefix=''):
    """Copy reference images to a temp directory for SeaS."""
    os.makedirs(target_dir, exist_ok=True)
    copied = []
    for i, img_name in enumerate(reference_paths):
        # The img_name is like "broken_large/001.png"
        # Find the actual file
        src_path = img_name
        if not os.path.exists(src_path):
            # Try to find in dataset
            continue
        dst_path = os.path.join(target_dir, f"{prefix}{i:04d}.png")
        shutil.copy2(src_path, dst_path)
        copied.append(dst_path)
    return copied


def generate_boundary_normal(seas_python, seas_dir, category, output_dir,
                              gen_model_path, rmp_model_path, sd_model_path,
                              ref_images, num_samples=100, gpu_id=0):
    """
    生成 Boundary Normal 样本:
    以高-M 的正常图为参考，生成 "正常但靠近决策边界" 的变体。
    """
    print(f"\n{'='*60}")
    print(f"Generating Boundary Normal samples ({category})")
    print(f"{'='*60}")

    bn_output = os.path.abspath(os.path.join(output_dir, category, 'boundary_normal'))
    os.makedirs(bn_output, exist_ok=True)

    # Create a ref directory with selected reference images
    bn_ref_dir = os.path.abspath(os.path.join(output_dir, category, '_bn_refs'))
    if os.path.exists(bn_ref_dir):
        shutil.rmtree(bn_ref_dir)
    os.makedirs(bn_ref_dir)

    # Copy reference normal images
    dataset_dir = '/data/chenjiawen/Datasets/MVTec-AD'
    normal_dir = os.path.join(dataset_dir, category, 'train', 'good')

    # Resolve image names from various formats to actual files
    def resolve_to_file(name, normal_dir):
        """Handle names like 'train_good_000' or 'train/good/000.png'."""
        # Direct check
        if os.path.exists(name):
            return name

        # Try as train/good/X.png
        for fname in sorted(glob(os.path.join(normal_dir, '*.png'))):
            base = os.path.splitext(os.path.basename(fname))[0]
            if base in name or name.endswith(base) or name == f"train_good_{base}":
                return fname
            if name.endswith(f"_{base}") or f"good/{base}" in name:
                return fname
        return None

    # Use the high-M normal images as references
    for i, sample in enumerate(ref_images[:10]):  # Use up to 10 reference images
        resolved = resolve_to_file(sample['img_name'], normal_dir)
        if resolved:
            shutil.copy2(resolved, os.path.join(bn_ref_dir, f"ref_{i:04d}.png"))
            print(f"  Ref {i}: {resolved}")

    # Also add some random normal images for diversity
    all_normals = sorted(glob(os.path.join(normal_dir, '*.png')))[:20]
    for i, f in enumerate(all_normals):
        idx = i + len(ref_images[:10])
        if idx >= 20:
            break
        shutil.copy2(f, os.path.join(bn_ref_dir, f"ref_{idx:04d}.png"))

    ref_count = len(glob(os.path.join(bn_ref_dir, '*.png')))
    if ref_count == 0:
        print("  WARNING: No reference images found, using all normal images")
        for i, f in enumerate(sorted(glob(os.path.join(normal_dir, '*.png')))[:20]):
            shutil.copy2(f, os.path.join(bn_ref_dir, f"ref_{i:04d}.png"))
        ref_count = min(20, len(glob(os.path.join(normal_dir, '*.png'))))

    print(f"  Using {ref_count} reference images from {normal_dir}")

    # Boundary Normal: use NORMAL prompt (only ob token, no sks anomaly tokens)
    # This generates normal-looking variations of the boundary-adjacent references.
    prompt = "a ob1"

    # Run SeaS inference
    cmd = [
        seas_python, os.path.join(seas_dir, 'examples', 'SeaS_infer.py'),
        '--output_dir', bn_output,
        '--ref_data_dir', bn_ref_dir,
        '--gen_model_path', gen_model_path,
        '--rmp_model_path', rmp_model_path,
        '--stable_diffusion_model_path', sd_model_path,
        '--prompt', prompt,
        '--total_infer_num', str(num_samples),
        '--num_inference_steps', '25',
        '--guidance_scale', '8',
        '--threshold', '0.2',
        '--seed_start', '42',
        '--gpu_id', str(gpu_id),
        '--gen_mask',
        '--onlyfinal',
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=seas_dir)
    if result.returncode != 0:
        print(f"  SeaS ERROR: {result.stderr[:500]}")
        print(f"  stdout (last 500): {result.stdout[-500:]}")
        return None

    print(f"  SeaS output: {result.stdout[-300:]}")

    # Check generated images
    gen_image_dir = os.path.join(bn_output, 'image')
    if os.path.exists(gen_image_dir):
        gen_count = len(glob(os.path.join(gen_image_dir, '*.png')))
        print(f"  Generated {gen_count} boundary normal images")

    # Save prompt info
    info = {
        'type': 'boundary_normal',
        'category': category,
        'prompt': prompt,
        'num_generated': num_samples,
        'reference_count': ref_count,
        'output_dir': bn_output,
    }
    with open(os.path.join(bn_output, 'generation_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    return bn_output


def generate_blind_defect(seas_python, seas_dir, category, output_dir,
                           gen_model_path, rmp_model_path, sd_model_path,
                           high_m_samples, num_samples=100, gpu_id=0):
    """
    生成 Blind Defect 样本:
    利用 SeaS 生成微小/难检测的异常样本。
    """
    print(f"\n{'='*60}")
    print(f"Generating Blind Defect samples ({category})")
    print(f"{'='*60}")

    bd_output = os.path.abspath(os.path.join(output_dir, category, 'blind_defect'))
    os.makedirs(bd_output, exist_ok=True)

    # Use normal images as reference (for anomaly generation)
    dataset_dir = '/data/chenjiawen/Datasets/MVTec-AD'
    normal_dir = os.path.join(dataset_dir, category, 'train', 'good')

    bd_ref_dir = os.path.abspath(os.path.join(output_dir, category, '_bd_refs'))
    if os.path.exists(bd_ref_dir):
        shutil.rmtree(bd_ref_dir)
    os.makedirs(bd_ref_dir)

    # Copy reference images
    for i, f in enumerate(sorted(glob(os.path.join(normal_dir, '*.png')))[:20]):
        shutil.copy2(f, os.path.join(bd_ref_dir, f"ref_{i:04d}.png"))

    ref_count = len(glob(os.path.join(bd_ref_dir, '*.png')))
    print(f"  Using {ref_count} reference images from {normal_dir}")

    # Blind Defect: use ANOMALY prompt (ob + sks tokens)
    # Generates subtle/hard-to-detect anomaly images.
    prompt = "a ob1 with sks1 sks2 sks3 sks4"

    cmd = [
        seas_python, os.path.join(seas_dir, 'examples', 'SeaS_infer.py'),
        '--output_dir', bd_output,
        '--ref_data_dir', bd_ref_dir,
        '--gen_model_path', gen_model_path,
        '--rmp_model_path', rmp_model_path,
        '--stable_diffusion_model_path', sd_model_path,
        '--prompt', prompt,
        '--total_infer_num', str(num_samples),
        '--num_inference_steps', '25',
        '--guidance_scale', '8',
        '--threshold', '0.2',
        '--seed_start', '142',
        '--gpu_id', str(gpu_id),
        '--gen_mask',
        '--onlyfinal',
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=seas_dir)
    if result.returncode != 0:
        print(f"  SeaS ERROR: {result.stderr[:500]}")
        print(f"  stdout (last 500): {result.stdout[-500:]}")
        return None

    print(f"  SeaS output: {result.stdout[-300:]}")

    gen_image_dir = os.path.join(bd_output, 'image')
    if os.path.exists(gen_image_dir):
        gen_count = len(glob(os.path.join(gen_image_dir, '*.png')))
        print(f"  Generated {gen_count} blind defect images")

    info = {
        'type': 'blind_defect',
        'category': category,
        'prompt': prompt,
        'num_generated': num_samples,
        'reference_count': ref_count,
        'output_dir': bd_output,
    }
    with open(os.path.join(bd_output, 'generation_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    return bd_output


def main():
    parser = argparse.ArgumentParser(description='Missing Boundary guided SeaS generation')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--high_m_json', type=str, required=True)
    parser.add_argument('--seas_dir', type=str, default='/data/chenjiawen/SeaS')
    parser.add_argument('--seas_python', type=str, default='/home/chenjiawen/anaconda3/envs/seas/bin/python')
    parser.add_argument('--gen_model_path', type=str, required=True)
    parser.add_argument('--rmp_model_path', type=str, required=True)
    parser.add_argument('--sd_model_path', type=str,
                        default='/data/chenjiawen/SeaS/model_hub/stable-diffusion-v1-4')
    parser.add_argument('--num_boundary_normal', type=int, default=100,
                        help='Number of boundary normal samples')
    parser.add_argument('--num_blind_defect', type=int, default=100,
                        help='Number of blind defect samples')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID for SeaS generation')
    parser.add_argument('--skip_boundary_normal', action='store_true')
    parser.add_argument('--skip_blind_defect', action='store_true')
    args = parser.parse_args()

    # Load high-M regions
    high_m = load_high_m_regions(args.high_m_json)
    print(f"Loaded high-M regions from: {args.high_m_json}")
    print(f"  Boundary normal candidates: {high_m['num_boundary_normal']}")
    print(f"  Blind defect candidates: {high_m['num_blind_defect']}")

    # Task A: Boundary Normal Generation
    bn_output = None
    if not args.skip_boundary_normal:
        bn_samples = high_m.get('boundary_normal_samples', [])
        bn_output = generate_boundary_normal(
            args.seas_python, args.seas_dir, args.category, args.output_dir,
            args.gen_model_path, args.rmp_model_path, args.sd_model_path,
            bn_samples, args.num_boundary_normal, args.gpu_id,
        )

    # Task B: Blind Defect Generation
    bd_output = None
    if not args.skip_blind_defect:
        bd_samples = high_m.get('blind_defect_samples', [])
        bd_output = generate_blind_defect(
            args.seas_python, args.seas_dir, args.category, args.output_dir,
            args.gen_model_path, args.rmp_model_path, args.sd_model_path,
            bd_samples, args.num_blind_defect, args.gpu_id,
        )

    # Summary
    print(f"\n{'='*60}")
    print("Generation Summary")
    print(f"{'='*60}")
    print(f"  Boundary Normal: {bn_output}")
    print(f"  Blind Defect: {bd_output}")

    # Save summary
    summary = {
        'category': args.category,
        'boundary_normal_output': bn_output,
        'blind_defect_output': bd_output,
    }
    summary_dir = os.path.abspath(os.path.join(args.output_dir, args.category))
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, 'generation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nGeneration complete! Summary saved.")


if __name__ == '__main__':
    main()
