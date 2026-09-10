#!/usr/bin/env python3
"""
build_candidate_pool.py — Selection Study Phase A
==================================================
为 selection study 构建一个"未过滤"的 SeaS 候选池, 逐候选保存全部筛选特征:
  M(x), Gap, PB, DRAEM image score, mask_area, amap 统计 (max/topk/region),
  shape (num_components/max_comp/compactness/aspect) 等 → candidate.csv。

对每个 train 高-M 正常参考图:
  - BN-style (正常 prompt, 官方 1500~1800 noise 阶梯)   → 边界正常候选
  - BD-style (异常 prompt, 同 noise 阶梯)   → 缺陷候选
注意: add_noise_step 必须 >= 1500 (SeaS 官方默认), 更低会产出大片暗色块。
每个 (ref, kind, noise) 生成一个 SeaS batch (num_variants 张), 逐张评分并保留
image/ + mask/ 到 pool (供后续 selector 引用训练)。**不做任何接受/拒绝过滤** ——
这是与闭环生成的关键区别: 闭环只留接受样本, 本研究需要全谱候选。

用法:
  python src/selection/build_candidate_pool.py \
    --category pipe_fryum \
    --high_m_json outputs/features_visa/pipe_fryum/missing_boundary/high_m_regions.json \
    --oracle_dir  outputs/features_visa/pipe_fryum/missing_boundary/mx_oracle \
    --dataset_dir outputs/visa_datasets \
    --output_dir  outputs/selection_pool \
    --gpus 0,6,7 --score_gpu 7
"""
import argparse, os, sys, json, shutil, subprocess, time, csv, threading
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mx_oracle import MxOracle
from generate_mb_guided import build_seas_cmd, make_temp_config, guidance_from_dataset

AREA_THR = 128          # mask > 128 视为缺陷像素
AMAP_THR = 0.5          # amap 二值阈值 (与 oracle.amap_cov 一致)
TOPK_FRAC = 0.01        # top 1% 像素


def resolve_image(dataset_dir, category, img_name):
    """兼容 'train/good/380.png' (VisA/MVTec 新格式) 与 'train_good_380' (旧)."""
    if os.path.exists(img_name):
        return img_name
    p = os.path.join(dataset_dir, category, img_name)
    if os.path.exists(p):
        return p
    if img_name.startswith('train/good/'):
        return os.path.join(dataset_dir, category, 'train', 'good',
                            img_name.split('/')[-1])
    if img_name.startswith('test/'):
        parts = img_name.split('/')
        return os.path.join(dataset_dir, category, 'test', parts[1], parts[2])
    if img_name.startswith('train_good_'):
        return os.path.join(dataset_dir, category, 'train', 'good',
                            f'{img_name.split("_")[-1]}.png')
    return None


def mask_amap_features(mask_path, amap_np, target=256):
    """SeaS mask + DRAEM amap 的统计特征.

    mask 与 amap 都缩放到 target×target 对齐.
    返回 dict: mask_area, mask_area_bin(0.05), num_components, max_comp_frac,
      compactness, aspect_ratio, amap_max, topk01, region_mean, region_max, bbox_frac.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    mask = cv2.resize(mask, (target, target), interpolation=cv2.INTER_NEAREST)
    mb = (mask > AREA_THR)
    area = float(mb.mean())
    h, w = mb.shape
    if amap_np.shape != mb.shape:
        amap_np = cv2.resize(amap_np, (target, target), interpolation=cv2.INTER_LINEAR)

    flat = amap_np.ravel()
    amap_max = float(flat.max())
    k = max(1, int(TOPK_FRAC * flat.size))
    topk01 = float(np.sort(flat)[-k:].mean())

    if area <= 0:
        return {'mask_area': area, 'mask_area_bin': float(area >= 0.05),
                'num_components': 0, 'max_comp_frac': 0.0, 'compactness': 0.0,
                'aspect_ratio': 0.0, 'bbox_frac': 0.0,
                'amap_max': amap_max, 'topk01': topk01,
                'region_mean': 0.0, 'region_max': 0.0}

    # connected components
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mb.astype(np.uint8), 8)
    comp_areas = stats[1:, cv2.CC_STAT_AREA]
    num_components = int(n - 1)
    max_comp_frac = float(comp_areas.max() / (h * w))
    # 最大连通区域 compactness: 4πA / P²
    max_i = int(np.argmax(comp_areas)) + 1
    cm = (labels == max_i)
    contours, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = float(cv2.arcLength(contours[0], True)) if contours else 1.0
    A_max = comp_areas[max_i - 1]
    compactness = float(4.0 * np.pi * A_max / (perim * perim + 1e-6))
    # 最大连通区域 aspect ratio + bbox 占比
    x, y, bw, bh = stats[max_i, :4]
    aspect_ratio = float(min(bw, bh) / max(bw, bh, 1))
    bbox_frac = float((bw * bh) / (h * w))

    return {'mask_area': area, 'mask_area_bin': float(area >= 0.05),
            'num_components': num_components, 'max_comp_frac': max_comp_frac,
            'compactness': compactness, 'aspect_ratio': aspect_ratio,
            'bbox_frac': bbox_frac,
            'amap_max': amap_max, 'topk01': topk01,
            'region_mean': float(amap_np[mb].mean()),
            'region_max': float(amap_np[mb].max())}


class PoolBuilder:
    def __init__(self, oracle, args):
        self.oracle = oracle
        self.args = args
        # GPU round-robin: 每个 SeaS 进程占 ~13GB, 同一 GPU 同时最多 1 个进程,
        # 否则 2×13GB > 24GB 显存 OOM (见 selection_pool OOM 诊断)。
        self._gpu_lock = threading.Lock()
        self._gpu_counter = [0]
        self._gpu_locks = {g: threading.Lock() for g in args.gpus}

    def _next_gpu(self):
        with self._gpu_lock:
            g = self.args.gpus[self._gpu_counter[0] % len(self.args.gpus)]
            self._gpu_counter[0] += 1
            return g

    def _gen_batch(self, kind, prompt, noise, seed, ref_dir):
        """调 SeaS 生成一批候选, 返回 (image_paths, mask_paths) + batch_dir."""
        gpu_id = self._next_gpu()
        config_path = make_temp_config(self.args.seas_dir, noise, f'pool_{kind}_{gpu_id}',
                                       guidance=self.args.guidance)
        num = max(self.args.num_variants, 10)
        cmd = build_seas_cmd(
            self.args.seas_python, self.args.seas_dir, '', ref_dir, prompt,
            self.args.gen_ckpt, self.args.rmp_ckpt, self.args.sd_path,
            gpu_id, num, config_path, seed_start=seed,
        )
        batch_dir = os.path.abspath(os.path.join(
            self.args.output_dir, self.args.category, 'pool',
            f'{kind}_{noise}_{seed}'))
        os.makedirs(batch_dir, exist_ok=True)
        # build_seas_cmd 的 output_dir 是 SeaS 输出根; SeaS_infer 会在其下建 image/ mask/
        # 直接用 batch_dir 作为 output_dir → image/batch_dir/image, mask/batch_dir/mask
        cmd[cmd.index('--output_dir') + 1] = batch_dir
        # 不用 capture_output=True: Python 3.8 下 SeaS 的子进程会继承 pipe 写端,
        # 导致 communicate() 永远等不到 EOF 而挂死 (见 selection_pool_build 挂起记录)。
        # 改走 per-batch 日志文件: subprocess.run 只等直接子进程退出, 文件 fd 被孙进程
        # 继承也无所谓。若 SeaS 崩溃, 错误见 <batch_dir>/gen.log。
        log_path = os.path.join(batch_dir, 'gen.log')
        # 同一 GPU 同时只跑一个 SeaS (锁在 subprocess 上) —— 避免 2×~13GB 显存 OOM。
        with self._gpu_locks[gpu_id]:
            try:
                with open(log_path, 'w') as log_f:
                    result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                            cwd=self.args.seas_dir, timeout=600)
                if result.returncode != 0:
                    tail = ''
                    try:
                        with open(log_path) as log_f:
                            tail = log_f.read()[-250:]
                    except OSError:
                        pass
                    print(f"  [SeaS ERROR] {kind} noise={noise}: {tail}")
                    return [], [], batch_dir
            except subprocess.TimeoutExpired:
                print(f"  [TIMEOUT] {kind} noise={noise} (log: {log_path})")
                return [], [], batch_dir
        return (sorted(glob(os.path.join(batch_dir, 'image', '*.png'))),
                sorted(glob(os.path.join(batch_dir, 'mask', '*.png'))), batch_dir)

    def _score_candidate(self, img_path, mask_path, ref_path, kind, noise, seed):
        """逐候选评分 + 特征 → row dict."""
        try:
            r = self.oracle.score_full(img_path)
        except Exception as e:
            print(f"  [score ERR] {img_path}: {e}")
            return None
        amap_np = r.pop('amap_np')
        try:
            f = mask_amap_features(mask_path, amap_np)
        except Exception as e:
            print(f"  [mask ERR] {mask_path}: {e}")
            return None
        row = {'img_path': img_path, 'mask_path': mask_path, 'ref_path': ref_path,
               'kind': kind, 'noise': noise, 'seed': seed,
               'M': r['M'], 'gap': r['gap'], 'pb': r['pb'],
               'score': r['score'], 'amap_cov': r['amap_cov']}
        row.update(f)
        return row

    def gen_ref(self, sample, idx):
        """Phase A: 只做 SeaS 生成, 不评分.

        返回该 ref 的全部候选 dict 列表 {img_path, mask_path, ref_path, kind,
        noise, seed}。与评分 (Phase B) 分离, 避免"并发 fork SeaS 子进程"与
        "并发 numpy/BLAS/torch 计算"在同一多线程进程内互相死锁
        (见 selection_pool 两次挂起诊断)。此阶段父进程只做 subprocess/IO,
        无任何数值计算。
        """
        src_path = resolve_image(self.args.dataset_dir, self.args.category, sample['img_name'])
        if src_path is None:
            print(f"  [skip] cannot resolve {sample['img_name']}")
            return []
        ref_dir = os.path.abspath(os.path.join(
            self.args.output_dir, self.args.category, '_refs_tmp',
            f'{idx}'))
        os.makedirs(ref_dir, exist_ok=True)
        for rep in range(10):
            shutil.copy2(src_path, os.path.join(ref_dir, f'ref_{rep:02d}.png'))

        jobs = []
        for noise in self.args.bn_noise:
            jobs.append(('bn', 'a ob1', noise))
        for noise in self.args.bd_noise:
            jobs.append(('bd', 'a ob1 with sks1 sks2 sks3 sks4', noise))

        cands = []
        seed = self.args.seed_offset + idx * 137
        for kind, prompt, noise in jobs:
            seed += 10
            imgs, masks, batch_dir = self._gen_batch(
                kind, prompt, noise, seed, ref_dir)
            for i, img in enumerate(imgs):
                if i >= len(masks):
                    continue
                cands.append({'img_path': img, 'mask_path': masks[i],
                              'ref_path': src_path, 'kind': kind,
                              'noise': noise, 'seed': seed})
        shutil.rmtree(ref_dir, ignore_errors=True)
        return cands


def main():
    parser = argparse.ArgumentParser(description='Build selection-study candidate pool')
    parser.add_argument('--category', default='pipe_fryum')
    parser.add_argument('--high_m_json', required=True)
    parser.add_argument('--oracle_dir', required=True)
    parser.add_argument('--dataset_dir', default='/data/chenjiawen/MissingBoundary/outputs/visa_datasets')
    parser.add_argument('--output_dir', default='/data/chenjiawen/MissingBoundary/outputs/selection_pool')
    parser.add_argument('--seas_dir', default='/data/chenjiawen/SeaS')
    parser.add_argument('--seas_python', default='/home/chenjiawen/anaconda3/envs/seas/bin/python')
    parser.add_argument('--draem_ckpt_dir', default='/data/chenjiawen/MissingBoundary/outputs/checkpoints/visa')
    parser.add_argument('--draem_base_name', default='DRAEM_test_0.0001_200_bs8')
    parser.add_argument('--draem_dir', default='/data/chenjiawen/DRAEM')
    parser.add_argument('--gpus', default='0,6,7')
    parser.add_argument('--score_gpu', type=int, default=7)
    parser.add_argument('--num_refs', type=int, default=20)
    parser.add_argument('--num_variants', type=int, default=10)
    # BN/BD 都用 SeaS 官方验证的 add_noise_step=1500 起步 (seas.yaml 默认)。
    # add_noise_step < 1500 产大片暗色块 (实测 300~1200 全黑, 连 g2 的 bd_1200 也是
    # 坏的, 见 selection_pool_g2 bn/bd 质量诊断)。BN/BD 的区别只靠 prompt
    # (正常 vs sks 异常), noise 区间相同。
    parser.add_argument('--bn_noise', default='1500,1650,1800')
    parser.add_argument('--bd_noise', default='1500,1650,1800')
    parser.add_argument('--max_workers', type=int, default=3)
    # 重生成实验: 平移每张参考图的推理 seed (默认 0 = 与历史池逐位一致).
    # seed = seed_offset + idx*137, 每个 (参考图,noise) job 再 +10.
    parser.add_argument('--seed_offset', type=int, default=0)
    args = parser.parse_args()

    args.bn_noise = [int(x) for x in args.bn_noise.split(',')]
    args.bd_noise = [int(x) for x in args.bd_noise.split(',')]
    args.gpus = [int(g) for g in args.gpus.split(',')]
    args.gen_ckpt = os.path.join(args.seas_dir, 'outputs', 'checkpoints',
                                 args.category, 'generation-checkpoint')
    args.rmp_ckpt = os.path.join(args.seas_dir, 'outputs', 'checkpoints',
                                 args.category, 'mask-checkpoint', 'rmp')
    args.sd_path = os.path.join(args.seas_dir, 'model_hub', 'stable-diffusion-v1-4')
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    # SeaS load_args 从 config 读取 guidance_scale (覆盖 CLI), 故按数据集推断并写入 temp config.
    # MVTec AD=8, VisA=2, MVTec 3D AD=5 (见 generate_mb_guided.guidance_from_dataset).
    args.guidance = guidance_from_dataset(args.dataset_dir)
    print(f"guidance_scale={args.guidance} (dataset_dir={args.dataset_dir})")

    with open(args.high_m_json) as f:
        high_m = json.load(f)
    refs = high_m['boundary_normal_samples'][:args.num_refs]
    print(f"refs: {len(refs)} (train high-M normal), "
          f"bn_noise={args.bn_noise} bd_noise={args.bd_noise}, gpus={args.gpus}")

    # oracle 延迟到 Phase A 之后加载: 生成期间父进程不持有 CUDA/BLAS 上下文,
    # fork SeaS 子进程才安全。多线程并发 fork + 共享主机显存争用已反复导致
    # 死锁/OOM (见 selection_pool 挂起与 OOM 诊断), 故 Phase A 走单进程顺序。
    builder = PoolBuilder(None, args)
    start = time.time()

    # Phase A: 顺序生成 (一次只 fork 一个 SeaS; round-robin 选卡单线程天然错开)
    print(f"\n[Phase A] 顺序生成 {len(refs)} refs × 6 batches, gpus={args.gpus} ...")
    candidates = []
    for idx, sample in enumerate(refs):
        candidates.extend(builder.gen_ref(sample, idx))
        print(f"  ref {idx+1}/{len(refs)}: 累计 {len(candidates)} candidates "
              f"({time.time()-start:.0f}s)")
    print(f"[Phase A] {len(candidates)} candidates 生成完毕, {time.time()-start:.0f}s")

    # 加载 oracle (生成完成后再持有 CUDA 上下文)
    print(f"加载 oracle: {args.oracle_dir}")
    builder.oracle = MxOracle.load(args.oracle_dir, draem_dir=args.draem_dir,
                                   checkpoint_dir=args.draem_ckpt_dir,
                                   base_model_name=args.draem_base_name,
                                   category=args.category,
                                   device=f'cuda:{args.score_gpu}')

    # Phase B: 顺序评分 (只计算, 无 fork)
    print(f"[Phase B] 逐候选评分 (oracle cuda:{args.score_gpu}) ...")
    rows = []
    for i, c in enumerate(candidates):
        row = builder._score_candidate(c['img_path'], c['mask_path'],
                                       c['ref_path'], c['kind'],
                                       c['noise'], c['seed'])
        if row is not None:
            rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(candidates)} ({time.time()-start:.0f}s)")
    print(f"[Phase B] {len(rows)} rows, {time.time()-start:.0f}s")

    out_cat = os.path.join(args.output_dir, args.category)
    os.makedirs(out_cat, exist_ok=True)
    csv_path = os.path.join(out_cat, 'candidate.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"\n候选池完成: {len(rows)} candidates, {time.time()-start:.0f}s")
    print(f"  csv: {csv_path}")
    if rows:
        import numpy as np
        arr = np.array([r['mask_area'] for r in rows])
        print(f"  mask_area: min={arr.min():.5f} med={np.median(arr):.5f} "
              f"P90={np.percentile(arr,90):.5f} max={arr.max():.5f}")
        print(f"  mask_area>0.05: {(arr >= 0.05).sum()} / {len(arr)}")
        print(f"  mask_area>=0.002: {(arr >= 0.002).sum()} / {len(arr)}")


if __name__ == '__main__':
    main()
