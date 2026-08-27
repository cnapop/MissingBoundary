#!/usr/bin/env python3
"""
M(x) Closed-Loop SeaS Generation (v3)
======================================
让 M(x) 真正驱动生成 —— 生成 → 评分 → 接受/拒绝 → 自适应参数:

对每个高-M 参考图 (BN=top-M train normal, BD=top-M train normal —— 与 BN 同源,
  prompt 决定内容; 不再使用 test 高-M 缺陷图, 严格 MVTec 协议 train-only):
  1. 用该图单独作 ref (ref 目录复制 10 份规避 SeaS einsum bug)
  2. 沿 noise 阶梯逐级生成 batch (每级 num_variants 张, 变 seed)
  3. 逐张评分 (mx_oracle): M(x), DRAEM score, RMP mask 覆盖率
  4. 接受条件:
       BN: M(x) >= m_accept_bn  AND mask_cov < rmp_thr_bn  (高-M 且内容正常)
       BD: M(x) >= m_accept_bd  AND mask_cov > rmp_thr_bd  (高-M 且缺陷存在)
                 AND score 落在 blind band (难检测)
  5. 配额未满 → 继续阶梯; 产量过低 → 降级 M_accept (P90→P85→P80)
  6. 记录每候选实测 M/gap/pb/score/mask_cov/noise/seed 与接受/拒绝原因,
     输出 generation_closedloop_summary.json (接受漏斗 = M(x) 驱动证据)

环境: 编排器用 DRAEM python (sklearn+torch+model_unet); SeaS 用 subprocess 调 seas python.
"""
import argparse, os, sys, json, shutil, subprocess, time, threading
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mx_oracle import MxOracle
from generate_mb_guided import build_seas_cmd, make_temp_config

_SCORE_LOCK = threading.Lock()


def resolve_image(dataset_dir, category, img_name):
    """兼容新格式 (train/good/203.png, test/contamination/006.png) 与旧格式 (train_good_203)."""
    if os.path.exists(img_name):
        return img_name
    if img_name.startswith('train/good/'):
        return os.path.join(dataset_dir, category, 'train', 'good', img_name.split('/')[-1])
    if img_name.startswith('test/'):
        parts = img_name.split('/')  # ['test', 'defect', 'idx.png']
        return os.path.join(dataset_dir, category, 'test', parts[1], parts[2])
    if img_name.startswith('train_good_'):
        idx = img_name.split('_')[-1]
        return os.path.join(dataset_dir, category, 'train', 'good', f'{idx}.png')
    if img_name.startswith('test_'):
        parts = img_name.split('_')
        idx = parts[-1]
        defect = '_'.join(parts[1:-1])
        return os.path.join(dataset_dir, category, 'test', defect, f'{idx}.png')
    return None


def read_mask_coverage(mask_path):
    """RMP mask 覆盖率 (mask>128 的像素占比)."""
    try:
        a = np.array(Image.open(mask_path).convert('L'))
        return float((a > 128).mean())
    except Exception:
        return None


class ClosedLoopWorker:
    def __init__(self, oracle, args):
        self.oracle = oracle
        self.args = args

    # ---------------------------------------------------------
    def _gen_batch(self, kind, sample, prompt, noise_step, gpu_id, seed_start,
                   ref_dir, gen_dir):
        """调 SeaS 生成一批候选到独立 gen_dir, 返回 (image_paths, mask_paths)."""
        config_path = make_temp_config(self.args.seas_dir, noise_step, f'{kind}_{gpu_id}')
        num = max(self.args.num_variants, 10)  # SeaS 需要 >= batch_size(10)
        cmd = build_seas_cmd(
            self.args.seas_python, self.args.seas_dir, gen_dir, ref_dir, prompt,
            self.args.gen_ckpt, self.args.rmp_ckpt, self.args.sd_path,
            gpu_id, num, config_path, seed_start=seed_start,
        )
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    cwd=self.args.seas_dir, timeout=1800)
            if result.returncode != 0:
                print(f"  [SeaS ERROR] {sample['img_name']} noise={noise_step}: "
                      f"{result.stderr[-200:]}")
                return [], []
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {sample['img_name']} noise={noise_step}")
            return [], []
        return (sorted(glob(os.path.join(gen_dir, 'image', '*.png'))),
                sorted(glob(os.path.join(gen_dir, 'mask', '*.png'))))

    # ---------------------------------------------------------
    def _evaluate_candidate(self, img_path, mask_path, kind, m_thr):
        """返回 (accept, record). record 含实测指标与拒绝原因."""
        with _SCORE_LOCK:
            r = self.oracle.score(img_path)
        mask_cov = read_mask_coverage(mask_path) if mask_path else None
        rec = {'img': img_path, 'img_name': os.path.basename(img_path),
               'mask_path': mask_path,
               'noise': None, 'seed': None, 'score': r['score'], 'gap': r['gap'],
               'pb': r['pb'], 'M': r['M'], 'amap_cov': r['amap_cov'],
               'mask_cov': mask_cov,
               'accept': False, 'reason': None}

        lo, hi = self.args.score_band
        if mask_cov is None:
            rec['reason'] = 'no_mask'
        elif kind == 'boundary_normal':
            if r['M'] < m_thr:
                rec['reason'] = 'M_below_thr'
            elif mask_cov >= self.args.rmp_thr_bn:
                rec['reason'] = 'mask_present'
            else:
                rec['accept'] = True
        else:  # blind_defect
            if r['M'] < m_thr:
                rec['reason'] = 'M_below_thr'
            elif mask_cov <= self.args.rmp_thr_bd:
                rec['reason'] = 'no_defect'
            elif not (lo <= r['score'] <= hi):
                rec['reason'] = 'score_out_of_band'
            else:
                rec['accept'] = True
        return rec['accept'], rec

    # ---------------------------------------------------------
    def run_one(self, kind, sample, prompt, gpu_id, idx):
        """单个参考图的全闭环: 阶梯生成 → 评分 → 接受/拒绝 → 自适应."""
        src_path = resolve_image(self.args.dataset_dir, self.args.category, sample['img_name'])
        if src_path is None:
            print(f"  [skip] cannot resolve {sample['img_name']}")
            return None

        quota = self.args.quota_per_ref
        m_thr = (self.oracle.state['m_accept_bn'] if kind == 'boundary_normal'
                 else self.oracle.state['m_accept_bd'])
        # 降级档: P{percentile} -> P{percentile-5} -> P{percentile-10}
        relax_thrs = (self.oracle.state['bn_relax'] if kind == 'boundary_normal'
                      else self.oracle.state['bd_relax'])

        # ref 目录: 同一高-M 图复制 10 份 (SeaS batch 需求)
        ref_dir = os.path.abspath(os.path.join(
            self.args.output_dir, self.args.category,
            f'_{kind}_refs_tmp_{abs(hash(sample["img_name"])) % 100000}'))
        os.makedirs(ref_dir, exist_ok=True)
        for rep in range(10):
            shutil.copy2(src_path, os.path.join(ref_dir, f'ref_{rep:02d}.png'))

        sample_out = os.path.abspath(os.path.join(
            self.args.output_dir, self.args.category, kind,
            f'sample_{os.path.basename(src_path)[:-4]}'))
        os.makedirs(sample_out, exist_ok=True)

        ladder = (self.args.bn_noise_ladder if kind == 'boundary_normal'
                  else self.args.bd_noise_ladder)
        accepted = []
        records = []
        attempts = 0
        relax_idx = 0
        seed = idx * 137
        saved = 0

        def _persist(rec):
            """把接受样本立即复制为 sample_out/acc_XX.png (+ mask)."""
            nonlocal saved
            if saved >= quota:
                return
            src = rec['img']
            if not os.path.exists(src):
                return
            dst = os.path.join(sample_out, f"acc_{saved:02d}.png")
            shutil.copy2(src, dst)
            if rec['mask_path'] and os.path.exists(rec['mask_path']):
                shutil.copy2(rec['mask_path'],
                             os.path.join(sample_out, f"acc_{saved:02d}_mask.png"))
            rec['saved_as'] = os.path.basename(dst)
            saved += 1

        while len(accepted) < quota and attempts < self.args.max_attempts:
            attempts += 1
            for noise in ladder:
                if len(accepted) >= quota:
                    break
                # 每批独立临时目录, 避免 SeaS 覆写, 也方便评分后即清
                gen_dir = os.path.join(sample_out, f'.tmp_{noise}_{seed}')
                os.makedirs(gen_dir, exist_ok=True)
                cand_imgs, cand_masks = self._gen_batch(
                    kind, sample, prompt, noise, gpu_id, seed, ref_dir, gen_dir)
                seed += 10
                for i, img in enumerate(cand_imgs):
                    mask = cand_masks[i] if i < len(cand_masks) else None
                    acc, rec = self._evaluate_candidate(img, mask, kind, m_thr)
                    rec['noise'] = noise
                    rec['seed'] = seed - 10
                    rec['sample'] = sample['img_name']
                    records.append(rec)
                    if acc:
                        accepted.append(rec)
                        _persist(rec)
                if not self.args.keep_candidates:
                    shutil.rmtree(gen_dir, ignore_errors=True)

            # 产量过低 → 降级 M_accept, 用剩余尝试继续
            if len(accepted) < quota and relax_idx < len(relax_thrs) - 1:
                relax_idx += 1
                m_thr = relax_thrs[relax_idx]
                print(f"  [relax] {sample['img_name']} {kind}: "
                      f"accepted {len(accepted)}/{quota}, "
                      f"M_thr -> {m_thr:.4f}")

        meta = {
            'kind': kind, 'source_img': sample['img_name'],
            'source_path': src_path, 'source_M': sample.get('M', None),
            'source_score': sample.get('score', None),
            'prompt': prompt, 'quota': quota, 'saved': saved,
            'm_accept_final': m_thr, 'gpu': gpu_id,
            'noise_ladder': ladder, 'attempts': attempts,
            'accepted': accepted[:quota],
            'output_dir': sample_out,
        }
        with open(os.path.join(sample_out, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        shutil.rmtree(ref_dir, ignore_errors=True)
        n_img = len(glob(os.path.join(sample_out, 'acc_[0-9][0-9].png')))
        print(f"  [OK] {kind} {sample['img_name']} (src_M={sample.get('M', 0):.3f}) "
              f"accepted {n_img}/{quota} (evaluated {len(records)}) on GPU{gpu_id}")
        return {'kind': kind, 'sample': sample['img_name'], 'records': records,
                'accepted': meta['accepted'], 'saved': saved, 'm_accept_final': m_thr}


def summarize(results, output_path, category):
    """汇总接受漏斗, 输出 JSON."""
    n_cand = n_acc = 0
    reasons = {}
    acc_M, acc_score, acc_mask = [], [], []
    per_ref = []
    for r in results:
        if r is None:
            continue
        n_cand += len(r['records'])
        n_acc += r['saved']
        for rec in r['records']:
            if not rec['accept']:
                reasons[rec['reason']] = reasons.get(rec['reason'], 0) + 1
            else:
                acc_M.append(rec['M']); acc_score.append(rec['score'])
                if rec['mask_cov'] is not None:
                    acc_mask.append(rec['mask_cov'])
        per_ref.append({'sample': r['sample'], 'kind': r['kind'],
                        'evaluated': len(r['records']), 'accepted': r['saved'],
                        'm_accept_final': r['m_accept_final']})

    summary = {
        'category': category,
        'total_candidates': n_cand,
        'total_accepted': n_acc,
        'rejected_by': reasons,
        'accepted_M_mean': float(np.mean(acc_M)) if acc_M else None,
        'accepted_M_std': float(np.std(acc_M)) if acc_M else None,
        'accepted_score_mean': float(np.mean(acc_score)) if acc_score else None,
        'accepted_mask_cov_mean': float(np.mean(acc_mask)) if acc_mask else None,
        'per_ref': per_ref,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print("=" * 60)
    print(f"接受漏斗: {n_cand} 候选 → {n_acc} 接受")
    for k, v in reasons.items():
        print(f"  拒绝原因 {k}: {v}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='M(x) closed-loop SeaS generation (v3)')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--high_m_json', type=str, required=True)
    parser.add_argument('--oracle_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='/data/chenjiawen/Datasets/MVTec-AD')
    parser.add_argument('--seas_dir', type=str, default='/data/chenjiawen/SeaS')
    parser.add_argument('--seas_python', type=str,
                        default='/home/chenjiawen/anaconda3/envs/seas/bin/python')
    parser.add_argument('--draem_ckpt_dir', type=str, default='/data/chenjiawen/DRAEM/checkpoints')
    parser.add_argument('--draem_base_name', type=str,
                        default='DRAEM_test_0.0001_700_bs8')
    parser.add_argument('--draem_dir', type=str, default='/data/chenjiawen/DRAEM')
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--score_gpu', type=int, default=7,
                        help='GPU for DRAEM scoring (separate from generation GPUs)')
    parser.add_argument('--num_variants', type=int, default=10,
                        help='Candidates per noise level per ref (>=10 for SeaS batch)')
    parser.add_argument('--num_bn', type=int, default=20)
    parser.add_argument('--num_bd', type=int, default=10)
    parser.add_argument('--quota_per_ref', type=int, default=8,
                        help='Target accepted samples per ref')
    parser.add_argument('--max_attempts', type=int, default=6,
                        help='Max ladder passes per ref before giving up')
    parser.add_argument('--bn_noise_ladder', type=str, default='300,500,700')
    parser.add_argument('--bd_noise_ladder', type=str, default='1200,1500,1800')
    parser.add_argument('--rmp_thr_bn', type=float, default=0.05,
                        help='BN max allowed mask coverage (normal content)')
    parser.add_argument('--rmp_thr_bd', type=float, default=0.05,
                        help='BD min required mask coverage (defect present)')
    parser.add_argument('--score_band', type=str, default='0.5,0.995',
                        help='BD blind score band (lo,hi)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--keep_candidates', action='store_true',
                        help='诊断用: 保留 SeaS 临时 batch (image/ mask/), 不清理')
    parser.add_argument('--gen_ckpt', type=str, default=None,
                        help='覆盖 generation-checkpoint 路径 (默认 seas_dir/outputs/checkpoints/{cat})')
    parser.add_argument('--rmp_ckpt', type=str, default=None,
                        help='覆盖 mask-checkpoint/rmp 路径')
    args = parser.parse_args()

    args.bn_noise_ladder = [int(x) for x in args.bn_noise_ladder.split(',')]
    args.bd_noise_ladder = [int(x) for x in args.bd_noise_ladder.split(',')]
    lo, hi = args.score_band.split(',')
    args.score_band = (float(lo), float(hi))
    args.gen_ckpt = args.gen_ckpt or os.path.join(
        args.seas_dir, 'outputs', 'checkpoints', args.category, 'generation-checkpoint')
    args.rmp_ckpt = args.rmp_ckpt or os.path.join(
        args.seas_dir, 'outputs', 'checkpoints', args.category, 'mask-checkpoint', 'rmp')
    args.sd_path = os.path.join(args.seas_dir, 'model_hub', 'stable-diffusion-v1-4')

    with open(args.high_m_json) as f:
        high_m = json.load(f)
    bn_normals = high_m.get('boundary_normal_samples', [])
    # BN 与 BD 都用高-M train 正常图作 ref: 同一 M 目标区域, prompt 决定内容
    # (BN=normal prompt → 高-M 正常变体; BD=anomaly prompt → 高-M 缺陷变体)
    bn_samples = bn_normals[:args.num_bn]
    # BD 参考图与 BN 同源 (train 高-M 正常图): 不再回退到 test 的 blind_defect_samples,
    # 保证生成阶段不接触任何 test 信息 (test 统计泄漏已从 oracle 与闭环中移除).
    bd_samples = bn_normals[args.num_bn:args.num_bn + args.num_bd] if len(bn_normals) > args.num_bn \
        else bn_normals[:args.num_bd]

    print(f"加载 oracle: {args.oracle_dir}")
    oracle = MxOracle.load(args.oracle_dir, draem_dir=args.draem_dir,
                           checkpoint_dir=args.draem_ckpt_dir,
                           base_model_name=args.draem_base_name,
                           category=args.category, device=f'cuda:{args.score_gpu}')
    print(f"  m_accept_bn={oracle.state['m_accept_bn']:.4f} "
          f"m_accept_bd={oracle.state['m_accept_bd']:.4f}")

    gpu_list = [int(g) for g in args.gpus.split(',')]
    os.makedirs(os.path.join(args.output_dir, args.category), exist_ok=True)

    jobs = []
    for i, s in enumerate(bn_samples):
        jobs.append(('boundary_normal', s, 'a ob1', i))
    for i, s in enumerate(bd_samples):
        jobs.append(('blind_defect', s, 'a ob1 with sks1 sks2 sks3 sks4', len(bn_samples) + i))
    print(f"Total jobs: {len(jobs)} (BN={len(bn_samples)}, BD={len(bd_samples)}), "
          f"GPUs={gpu_list}, score_gpu={args.score_gpu}")

    worker = ClosedLoopWorker(oracle, args)

    def job_fn(idx_job):
        kind, sample, prompt, idx = idx_job
        gpu = gpu_list[idx % len(gpu_list)]
        return worker.run_one(kind, sample, prompt, gpu, idx)

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(gpu_list)) as ex:
        futures = [ex.submit(job_fn, j) for j in jobs]
        for fut in futures:
            results.append(fut.result())
    elapsed = time.time() - start

    summary_path = os.path.join(args.output_dir, args.category, 'generation_closedloop_summary.json')
    summarize(results, summary_path, args.category)
    print(f"\n闭环生成完成: {elapsed:.1f}s, 漏斗摘要: {summary_path}")


if __name__ == '__main__':
    main()
