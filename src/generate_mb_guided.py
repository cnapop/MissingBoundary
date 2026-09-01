#!/usr/bin/env python3
"""
M(x)-Guided SeaS Generation (v2)
=================================
真正让 M(x) 驱动生成 —— 逐高-M 样本定向生成:

对每个高-M Boundary Normal 样本 x_i:
  - 单独用 x_i 作为 ref (ref 目录只放 x_i)
  - add_noise_step 用 SeaS 官方默认 1500 (低于 1500 会产大片暗色块, 实测 300~1200
    全黑; "低 noise 贴近参考图" 是误解, BN/BD 的区别靠 prompt 而非 noise)
  - 正常 prompt "a ob1"
  输出: boundary_normal/sample_{i}/ (含 meta.json 记录源 M, ref, prompt)

对每个高-M 参考 + 异常 prompt 生成 Blind Defect:
  - 用高-M 正常图作参考 + 异常 prompt
  - 标准 add_noise_step
  输出: blind_defect/sample_{i}/

支持多 GPU 并行。每个样本记录:
  - source_M: 参考图的 M(x) 值
  - source_img: 参考图路径
  - prompt, add_noise_step
"""
import argparse, os, re, sys, json, shutil, subprocess, time
from glob import glob
from concurrent.futures import ThreadPoolExecutor


def resolve_image(dataset_dir, category, img_name):
    """Resolve img_name (e.g. 'train_good_203') to actual file path."""
    if os.path.exists(img_name):
        return img_name
    # train_good_203
    if img_name.startswith('train_good_'):
        idx = img_name.split('_')[-1]
        return os.path.join(dataset_dir, category, 'train', 'good', f'{idx}.png')
    if img_name.startswith('test_'):
        parts = img_name.split('_')
        idx = parts[-1]
        defect = '_'.join(parts[1:-1])
        return os.path.join(dataset_dir, category, 'test', defect, f'{idx}.png')
    return None


def build_seas_cmd(seas_python, seas_dir, output_dir, ref_dir, prompt,
                   gen_ckpt, rmp_ckpt, sd_path, gpu_id, total_infer_num,
                   config_path, seed_start=0):
    """
    Build SeaS command. Uses a temp config file to control add_noise_step
    (SeaS load_args overwrites CLI add_noise_step with config value).
    """
    cmd = [
        seas_python, os.path.join(seas_dir, 'examples', 'SeaS_infer.py'),
        '--config', config_path,
        '--output_dir', output_dir,
        '--ref_data_dir', ref_dir,
        '--gen_model_path', gen_ckpt,
        '--rmp_model_path', rmp_ckpt,
        '--stable_diffusion_model_path', sd_path,
        '--prompt', prompt,
        '--total_infer_num', str(total_infer_num),
        '--threshold', '0.2',
        '--seed_start', str(seed_start),
        '--gpu_id', str(gpu_id),
        '--gen_mask',
        '--onlyfinal',
    ]
    return cmd


def guidance_from_dataset(dataset_dir):
    """Map a dataset directory to the SeaS guidance_scale it needs.

    SeaS configs/seas.yaml documents the per-dataset values:
        MVTec AD   -> 8
        VisA       -> 2
        MVTec 3D AD -> 5
    Missing Boundary can run on any of the three, so generation must not
    silently reuse the MVTec default for VisA: guidance=8 on VisA produces
    pixel-block / noisy images (verified in the SeaS subtype experiment).
    """
    p = (dataset_dir or '').lower()
    if 'visa' in p:
        return 2
    if '3d' in p:
        return 5
    return 8


def make_temp_config(seas_dir, add_noise_step, tag, guidance=None):
    """
    Create a temp seas.yaml config with custom add_noise_step (and optionally
    guidance_scale). SeaS load_args reads add_noise_step / guidance_scale etc.
    from the config and OVERWRITES the CLI values, so they must be set here.
    gpu_id is set to null so the CLI --gpu_id (per-worker) takes effect.
    """
    src_config = os.path.join(seas_dir, 'configs', 'seas.yaml')
    with open(src_config) as f:
        content = f.read()
    content = re.sub(r'(add_noise_step:\s*)\d+', f'\\g<1>{add_noise_step}', content)
    if guidance is not None:
        content = re.sub(r'(guidance_scale:\s*)\d+', f'\\g<1>{guidance}', content)
    content = re.sub(r'(gpu_id:\s*)\S+', 'gpu_id: null', content)
    # device must be cuda:0 because CUDA_VISIBLE_DEVICES remaps the physical GPU
    content = re.sub(r'(device:\s*["\']?)cuda:\d+(["\']?)', r'\1cuda:0\2', content)
    g_tag = f'_g{guidance}' if guidance is not None else ''
    tmp_path = f'/tmp/seas_mb_{tag}_{add_noise_step}{g_tag}.yaml'
    with open(tmp_path, 'w') as f:
        f.write(content)
    return tmp_path


def generate_for_sample(seas_python, seas_dir, gen_ckpt, rmp_ckpt, sd_path,
                        gpu_id, sample, dataset_dir, category, output_dir,
                        kind, num_variants, config_path, prompt,
                        seed_start=0):
    """
    Generate variants around a single high-M sample.
    kind: 'boundary_normal' or 'blind_defect'
    """
    src_path = resolve_image(dataset_dir, category, sample['img_name'])
    if src_path is None:
        print(f"  [skip] cannot resolve {sample['img_name']}")
        return None

    # Create ref dir for this high-M sample (ABSOLUTE path — SeaS subprocess cwd=seas_dir).
    # NOTE: SeaS requires the ref dir to have >= batch_size images; a single image
    # triggers an einsum batch mismatch in its attention-store hook. We replicate the
    # SAME high-M image 10x (distinct filenames) so every generated variant is anchored
    # on this sample — this is what makes M(x) truly guide the generation.
    ref_dir = os.path.abspath(os.path.join(
        output_dir, category, f'_{kind}_refs_tmp_{abs(hash(sample["img_name"])) % 100000}'))
    os.makedirs(ref_dir, exist_ok=True)
    for rep in range(10):
        shutil.copy2(src_path, os.path.join(ref_dir, f'ref_{rep:02d}.png'))

    sample_out = os.path.abspath(os.path.join(
        output_dir, category, kind, f'sample_{os.path.basename(src_path)[:-4]}'))
    os.makedirs(sample_out, exist_ok=True)

    cmd = build_seas_cmd(
        seas_python, seas_dir, sample_out, ref_dir, prompt,
        gen_ckpt, rmp_ckpt, sd_path, gpu_id, num_variants,
        config_path, seed_start=seed_start,
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=seas_dir, timeout=1800)
        if result.returncode != 0:
            print(f"  [ERROR] {sample['img_name']}: {result.stderr[-300:]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {sample['img_name']}")
        return None

    # Record meta
    meta = {
        'kind': kind,
        'source_img': sample['img_name'],
        'source_path': src_path,
        'source_M': sample.get('M', None),
        'source_score': sample.get('score', None),
        'prompt': prompt,
        'config': os.path.basename(config_path),
        'num_generated': num_variants,
        'gpu': gpu_id,
        'output_dir': sample_out,
    }
    with open(os.path.join(sample_out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Cleanup ref dir
    shutil.rmtree(ref_dir, ignore_errors=True)

    # Count generated
    n_img = len(glob(os.path.join(sample_out, 'image', '*.png')))
    print(f"  [OK] {kind} {sample['img_name']} (M={sample.get('M', 0):.4f}) -> {n_img} imgs on GPU{gpu_id}")
    return meta


def main():
    parser = argparse.ArgumentParser(description='M(x)-guided per-sample SeaS generation')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--high_m_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='/data/chenjiawen/Datasets/MVTec-AD')
    parser.add_argument('--seas_dir', type=str, default='/data/chenjiawen/SeaS')
    parser.add_argument('--seas_python', type=str,
                        default='/home/chenjiawen/anaconda3/envs/seas/bin/python')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Comma-separated GPU IDs to use in parallel')
    parser.add_argument('--num_variants', type=int, default=10,
                        help='Number of variants per high-M sample')
    parser.add_argument('--num_bn', type=int, default=20,
                        help='Number of boundary-normal samples to generate for')
    parser.add_argument('--num_bd', type=int, default=10,
                        help='Number of blind-defect samples to generate for')
    parser.add_argument('--bn_noise_step', type=int, default=1500,
                        help='add_noise_step for boundary normal (SeaS 官方默认 1500; <1500 产暗色块)')
    parser.add_argument('--bd_noise_step', type=int, default=1500,
                        help='add_noise_step for blind defect')
    parser.add_argument('--guidance', type=float, default=None,
                        help='SeaS guidance_scale override (default: inferred from '
                             'dataset_dir — MVTec AD=8, VisA=2, MVTec 3D AD=5)')
    args = parser.parse_args()

    with open(args.high_m_json) as f:
        high_m = json.load(f)

    bn_samples = high_m.get('boundary_normal_samples', [])[:args.num_bn]
    bd_samples = high_m.get('boundary_normal_samples', [])[:args.num_bd]

    gen_ckpt = os.path.join(args.seas_dir, 'outputs', 'checkpoints', args.category, 'generation-checkpoint')
    rmp_ckpt = os.path.join(args.seas_dir, 'outputs', 'checkpoints', args.category, 'mask-checkpoint', 'rmp')
    sd_path = os.path.join(args.seas_dir, 'model_hub', 'stable-diffusion-v1-4')

    gpu_list = [int(g) for g in args.gpus.split(',')]
    os.makedirs(os.path.join(args.output_dir, args.category), exist_ok=True)

    # Create temp configs to control add_noise_step AND guidance_scale
    # (SeaS load_args overrides both from the config, so CLI values are ignored).
    guidance = args.guidance if args.guidance is not None else guidance_from_dataset(args.dataset_dir)
    bn_config = make_temp_config(args.seas_dir, args.bn_noise_step, 'bn', guidance=guidance)
    bd_config = make_temp_config(args.seas_dir, args.bd_noise_step, 'bd', guidance=guidance)

    jobs = []
    # Boundary Normal: 官方 1500 noise step, normal prompt, high-M normal refs
    for i, sample in enumerate(bn_samples):
        jobs.append(('boundary_normal', sample, 'a ob1', bn_config))
    # Blind Defect: standard noise step, anomaly prompt, high-M refs
    for i, sample in enumerate(bd_samples):
        jobs.append(('blind_defect', sample, 'a ob1 with sks1 sks2 sks3 sks4', bd_config))

    print(f"Total jobs: {len(jobs)} (BN={len(bn_samples)}, BD={len(bd_samples)})")
    print(f"GPUs: {gpu_list}, BN noise_step={args.bn_noise_step}, BD noise_step={args.bd_noise_step}, "
          f"guidance={guidance} (dataset: {args.dataset_dir})")

    def worker(idx_job):
        idx, (kind, sample, prompt, config_path) = idx_job
        gpu = gpu_list[idx % len(gpu_list)]
        seed = idx * 137
        return generate_for_sample(
            args.seas_python, args.seas_dir, gen_ckpt, rmp_ckpt, sd_path,
            gpu, sample, args.dataset_dir, args.category, args.output_dir,
            kind, args.num_variants, config_path, prompt, seed_start=seed,
        )

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = [executor.submit(worker, (idx, job)) for idx, job in enumerate(jobs)]
        for f in futures:
            results.append(f.result())

    elapsed = time.time() - start
    done = [r for r in results if r]
    print(f"\n{'='*60}")
    print(f"Generation complete: {len(done)}/{len(jobs)} succeeded in {elapsed:.1f}s")
    print(f"Output: {os.path.join(args.output_dir, args.category)}")

    # Summary JSON
    summary = {
        'num_jobs': len(jobs),
        'num_succeeded': len(done),
        'bn_noise_step': args.bn_noise_step,
        'bd_noise_step': args.bd_noise_step,
        'guidance': guidance,
        'dataset_dir': args.dataset_dir,
        'num_variants': args.num_variants,
        'samples': done,
    }
    with open(os.path.join(args.output_dir, args.category, 'generation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
