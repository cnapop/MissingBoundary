#!/usr/bin/env python3
"""
按 sks 块生成 VisA pipe_fryum 子类缺陷
========================================
对重训后的 10 子类 SeaS checkpoint, 每个 sks 块 (每子类 4 个 token) 用同一组
ref 正常图 + 对应 prompt 生成一批缺陷图, 验证"换 sks 序号 → 不同类型缺陷".

prompt 模板: "a ob1 with sks{4i+1} sks{4i+2} sks{4i+3} sks{4i+4}"
ref: 从 train/good 等距取 10 张 (SeaS 要求 ref 目录 >= batch_size=10).

输出: outputs/subtype_gen/{cat}/{i:02d}_{type}/{image|mask}/*.png + meta.json
用法:
  python src/closedloop_v3/gen_per_sks.py --gpu 2 [--per_block 30]
"""
import argparse, json, os, shutil, subprocess, sys
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from generate_mb_guided import build_seas_cmd, make_temp_config, guidance_from_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--category', default='pipe_fryum')
    p.add_argument('--seas_dir', default='/data/chenjiawen/SeaS')
    p.add_argument('--seas_python', default='/home/chenjiawen/anaconda3/envs/seas/bin/python')
    p.add_argument('--gen_ckpt', default='/data/chenjiawen/SeaS/outputs/checkpoints/pipe_fryum_subtype/generation-checkpoint')
    p.add_argument('--rmp_ckpt', default='/data/chenjiawen/SeaS/outputs/checkpoints/pipe_fryum_subtype/mask-checkpoint/rmp')
    p.add_argument('--report', default=os.path.join(ROOT, 'outputs', 'visa_seas_subtype', '{cat}_subtype_report.json'))
    p.add_argument('--ref_dir', default=os.path.join(ROOT, 'outputs', 'visa_datasets', '{cat}', 'train', 'good'))
    p.add_argument('--dataset_dir', default=os.path.join(ROOT, 'outputs', 'visa_datasets'),
                   help='用于推断 guidance_scale (VisA → 2)')
    p.add_argument('--out_dir', default=os.path.join(ROOT, 'outputs', 'subtype_gen'))
    p.add_argument('--per_block', type=int, default=30)
    p.add_argument('--noise_step', type=int, default=1500)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--seed_start', type=int, default=100)
    p.add_argument('--keep_refs', action='store_true', help='保留临时 ref 目录')
    args = p.parse_args()

    cat = args.category
    guidance = guidance_from_dataset(args.dataset_dir)
    with open(args.report.format(cat=cat)) as f:
        report = json.load(f)
    types = sorted(report['subtypes'].keys())  # 与 SeaS sorted(iterdir) 一致

    sd_path = os.path.join(args.seas_dir, 'model_hub', 'stable-diffusion-v1-4')
    src_refs = sorted(glob(os.path.join(args.ref_dir.format(cat=cat), '*.png')))
    if len(src_refs) < 10:
        raise SystemExit(f'ref 不足 10 张: {len(src_refs)}')
    # 等距取 10 张, 所有块共用 → 仅 prompt 不同
    picks = [src_refs[int(i * (len(src_refs) - 1) / 9)] for i in range(10)]

    summary = {}
    for i, typ in enumerate(types):
        sks = [f'sks{i*4 + k}' for k in (1, 2, 3, 4)]
        prompt = f'a ob1 with {" ".join(sks)}'
        # 临时 ref 目录 (SeaS einsum bug 需要 >= batch_size=10)
        ref_dir = os.path.abspath(os.path.join(args.out_dir, cat, f'_refs_tmp'))
        os.makedirs(ref_dir, exist_ok=True)
        for rep, src in enumerate(picks):
            shutil.copy2(src, os.path.join(ref_dir, f'ref_{rep:02d}.png'))

        out = os.path.abspath(os.path.join(args.out_dir, cat, f'{i:02d}_{typ}'))
        config_path = make_temp_config(args.seas_dir, args.noise_step, f'sks{i}',
                                       guidance=guidance)
        cmd = build_seas_cmd(
            args.seas_python, args.seas_dir, out, ref_dir, prompt,
            args.gen_ckpt, args.rmp_ckpt, sd_path, args.gpu,
            args.per_block, config_path, seed_start=args.seed_start,
        )
        print(f'[block {i:02d}] {typ}: {prompt}')
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=args.seas_dir, timeout=3600)
            if res.returncode != 0:
                print(f'  [ERROR] {res.stderr[-500:]}')
                summary[typ] = {'ok': False, 'err': res.stderr[-300:]}
                continue
        except subprocess.TimeoutExpired:
            print(f'  [TIMEOUT]')
            summary[typ] = {'ok': False, 'err': 'timeout'}
            continue

        n_img = len(glob(os.path.join(out, 'image', '*.png')))
        n_mask = len(glob(os.path.join(out, 'mask', '*.png')))
        with open(os.path.join(out, 'meta.json'), 'w') as f:
            json.dump({'block': i, 'type': typ, 'sks': sks, 'prompt': prompt,
                       'refs': [os.path.basename(x) for x in picks],
                       'noise_step': args.noise_step, 'guidance': guidance,
                       'n_img': n_img}, f, indent=2)
        summary[typ] = {'ok': True, 'sks': sks, 'n_img': n_img, 'n_mask': n_mask}
        print(f'  -> {n_img} imgs / {n_mask} masks')
        if not args.keep_refs:
            shutil.rmtree(ref_dir, ignore_errors=True)

    out_json = os.path.join(args.out_dir, cat, 'generation_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'完成 → {out_json}')


if __name__ == '__main__':
    main()
