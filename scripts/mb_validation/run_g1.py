#!/usr/bin/env python3
"""
run_g1.py — G1: baseline seed 方差 (fryum / pipe_fryum)
========================================================
回答: fryum 上的 S8 +0.1903 是否超过 plain baseline 自身的随机种子波动?

设计 (见 /home/chenjiawen/workspace/实验方案.md §四/§五):
  - 每类别 × {baseline, S8} × seeds {0,1,2} = 12 runs。
  - baseline: DRAEM 原生配方 (bs8, lr1e-4, 200ep, DTD 合成, num_workers=16)
  - S8:       fix2 配方 (bs4, 空注入之外与 cross_visa S8 完全一致: random init,
              lr1e-4, 200ep, mirror-weight 0.5, num-workers 4)
  - 两条路径都设 seed 并开 cuDNN 确定性 (train_draem_single.py --seed),
    否则 "seed 方差" 里会混入算法非确定性, 与 fix2 的 std 不可比。

产物 (方案 §三):
  experiments/mb_validation/G1_seed_variance/
    ├── config.json                      全局配置
    ├── {cat}/{method}_seed{s}/run_config.json   单 run 元数据 (§二十五)
    ├── {cat}/{method}_seed{s}/train.log
    ├── {cat}/{method}_seed{s}/checkpoint/
    └── {cat}/{method}_seed{s}/eval.json         4 个指标
  汇总: run_g1_summary.py 读全部 eval.json -> mean±std

用法:
  {PY} scripts/mb_validation/run_g1.py --phase all --gpus 2,3
  {PY} scripts/mb_validation/run_g1.py --phase eval      # 只补评估
"""
import argparse, json, math, os, subprocess, sys, time

ROOT = '/data/chenjiawen/MissingBoundary'
PY = '/home/chenjiawen/anaconda3/envs/DRAEM/bin/python'
DATASETS = os.path.join(ROOT, 'outputs', 'visa_datasets')
DTD = '/data/chenjiawen/DRAEM/datasets/dtd/images'
DRAEM_REPO = '/data/chenjiawen/DRAEM'
EXP_ROOT = os.path.join(ROOT, 'experiments', 'mb_validation', 'G1_seed_variance')

BASE_NAME = 'DRAEM_test_0.0001_200_bs8'
EPOCHS = 200
LR = 1e-4
MIRROR_W = 0.5
BS_FIX2 = 4
BS_DRAEM = 8
NUM_WORKERS_FIX2 = 4
MIN_FREE_MIB = 20000   # 只挑基本空闲的卡 (24564 MiB 卡上占用 <4.5G)。
                       # 单 run 只占 12626 MiB, 但阈值若贴着它设, 就会出现
                       # "卡上已有一个 run (free ~13G) 仍判为可启动" → 两张挤一张卡。
SEEDS = [0, 1, 2]
CATS = ['fryum', 'pipe_fryum']

# S8 数据源: fryum 来自 cross_visa (selection_g3_fryum),
# pipe_fryum 来自最早的 selection 研究 (outputs/selection)。
SEL_DIR = {
    'fryum': os.path.join(ROOT, 'outputs', 'selection_g3_fryum', 'S8_m_soft'),
    'pipe_fryum': os.path.join(ROOT, 'outputs', 'selection', 'S8_m_soft'),
}
# baseline 在 VisA 上的历史参照值 (pixel AUC), 仅用于记录, 不参与判定
BASELINE_REF = {'fryum': 0.6898, 'pipe_fryum': 0.6982}
S8_REF = {'fryum': 0.8801, 'pipe_fryum': 0.5836}


def rundir(cat, method, seed):
    return os.path.join(EXP_ROOT, cat, f'{method}_seed{seed}')


def eval_path(cat, method, seed):
    return os.path.join(rundir(cat, method, seed), 'eval.json')


def is_active(cat, method, seed):
    """该 run 是否已有外部训练进程在跑 (cmdline 含它的 checkpoint 目录)。

    runner 被杀后重启时, 早先启动的 run 仍在训练; 若不加这道判断, runner 会
    在别的卡上重复启动同一 run, 两个进程覆写同一份 checkpoint → 结果作废。
    """
    d = os.path.join(rundir(cat, method, seed), 'checkpoint')
    r = subprocess.run(['pgrep', '-f', d], capture_output=True, text=True)
    return bool(r.stdout.strip())


def gpu_free_mib(gpu):
    """该 GPU 当前空闲显存 (MiB); 查不到返回 None。"""
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
             '-i', str(gpu)], capture_output=True, text=True, timeout=15)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def train_complete(cat, method, seed):
    """train.log 是否已打印到最后一个 epoch —— 区分"训完待评估"与"没训/训一半"。

    进程退出时 stdio 缓冲会 flush, 所以进程结束后该判断是可靠的。缺这一步的话,
    runner 会把"已训完但还没评估"的 run 误判为"未训练"而重训 200 epoch。
    """
    p = os.path.join(rundir(cat, method, seed), 'train.log')
    if not os.path.exists(p):
        return False
    txt = open(p, errors='ignore').read()
    if method == 'baseline':
        return f'Epoch: {EPOCHS - 1}' in txt        # DRAEM 原生: "Epoch: 199"
    return f'epoch={EPOCHS - 1}/{EPOCHS}' in txt    # fix2: "epoch=199/200"


def count_png(d, pattern='*.png'):
    import glob
    return len(glob.glob(os.path.join(d, pattern)))


def train_cmd(cat, method, seed, gpu, out):
    ckpt = os.path.join(out, 'checkpoint')
    if method == 'baseline':
        return [PY, os.path.join(ROOT, 'src', 'closedloop_v3', 'train_draem_single.py'),
                '--obj_name', cat, '--bs', str(BS_DRAEM), '--lr', str(LR),
                '--epochs', str(EPOCHS), '--gpu_id', str(gpu),
                '--data_path', DATASETS, '--anomaly_source_path', DTD,
                '--checkpoint_path', ckpt, '--log_path', out,
                '--seed', str(seed)]
    sel = SEL_DIR[cat]
    return [PY, os.path.join(ROOT, 'src', 'fix2', 'train_draem_fix2.py'),
            '--manifest', os.path.join(sel, 'bd_manifest.csv'),
            '--normal-data-dir', os.path.join(sel, 'train_good_plus_bn'),
            '--anomaly-source-path', DTD,
            '--init-random', '--base-name', BASE_NAME, '--category', cat,
            '--output-dir', ckpt,
            '--epochs', str(EPOCHS), '--lr', str(LR),
            '--mirror-weight', str(MIRROR_W), '--batch-size', str(BS_FIX2),
            '--num-workers', str(NUM_WORKERS_FIX2),
            '--draem-repo', DRAEM_REPO, '--device', f'cuda:{gpu}',
            '--seed', str(seed)]


def eval_cmd(cat, gpu, out):
    ckpt = os.path.join(out, 'checkpoint')
    return [PY, os.path.join(ROOT, 'src', 'selection', 'eval_study.py'),
            '--gpu_id', str(gpu), '--base_model_name', BASE_NAME,
            '--data_path', DATASETS, '--checkpoint_path', ckpt]


def write_run_config(cat, method, seed, gpu, out):
    sel = SEL_DIR[cat] if method == 'S8' else None
    n_bn = count_png(os.path.join(sel, 'train_good_plus_bn'), 'mb_bn_*.png') if sel else 0
    n_bd = 0
    if sel:
        with open(os.path.join(sel, 'bd_manifest.csv')) as f:
            n_bd = sum(1 for _ in f) - 1
    n_real = count_png(os.path.join(DATASETS, cat, 'train', 'good'))
    cfg = {
        'experiment_name': f'G1_{cat}_{method}_seed{seed}',
        'category': cat,
        'method': method,
        'seed': seed,
        'gpu': gpu,
        'dataset': 'VisA',
        'train_test_split': 'VisA 官方 split (train/good 200ep 训练, test 评估)',
        'image_resolution': 256,
        'model_architecture': 'DRAEM (ReconstructiveSubNetwork + DiscriminativeSubNetwork)',
        'optimizer': 'Adam',
        'learning_rate': LR,
        'batch_size': BS_DRAEM if method == 'baseline' else BS_FIX2,
        'loss': 'DRAEM: L2 + SSIM + Focal',
        'total_training_epochs': EPOCHS,
        'steps_per_epoch': (math.ceil(n_real / BS_DRAEM) if method == 'baseline'
                            else max(math.ceil((n_real + n_bn) / BS_FIX2),
                                     math.ceil(n_bd / BS_FIX2))),
        'augmentation': 'DTD 合成异常 (DRAEM 原生)',
        'init_method': 'random' if method == 'S8' else 'random (DRAEM weights_init)',
        'deterministic': True,
        'num_real_samples': n_real,
        'num_synthetic_samples': n_bn + n_bd,
        'num_synthetic_bn': n_bn,
        'num_synthetic_bd': n_bd,
        'selection_method': 'S8_m_soft (M(x) soft)' if method == 'S8' else None,
        'selection_source': sel,
        'generator': 'SeaS (mask-checkpoint/rmp, SD v1.4)',
        'generator_seed': 'idx*137 + 10*k (pool 默认, seed_offset=0)',
        'guidance_scale': 2,
        'mirror_weight': MIRROR_W if method == 'S8' else None,
        'num_workers': NUM_WORKERS_FIX2 if method == 'S8' else 16,
        'checkpoint': os.path.join(out, 'checkpoint'),
        'reference_reported': {
            'baseline_pixel_auc': BASELINE_REF[cat],
            'S8_pixel_auc_historical': S8_REF[cat],
        },
    }
    json.dump(cfg, open(os.path.join(out, 'run_config.json'), 'w'), indent=2,
              ensure_ascii=False)
    return cfg


def run_train(cat, method, seed, gpu):
    out = rundir(cat, method, seed)
    os.makedirs(os.path.join(out, 'checkpoint'), exist_ok=True)
    cfg = write_run_config(cat, method, seed, gpu, out)
    log = os.path.join(out, 'train.log')
    cmd = train_cmd(cat, method, seed, gpu, out)
    t0 = time.time()
    with open(log, 'w') as f:
        f.write('CMD: ' + ' '.join(cmd) + '\n\n')
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    dt = time.time() - t0
    cfg['training_time_s'] = round(dt)
    cfg['train_rc'] = rc
    json.dump(cfg, open(os.path.join(out, 'run_config.json'), 'w'), indent=2,
              ensure_ascii=False)
    return rc == 0, dt


def run_eval(cat, gpu, out, timeout=1800):
    env = dict(os.environ)
    env['MB_CAT'] = cat
    env['PYTHONPATH'] = DRAEM_REPO + os.pathsep + env.get('PYTHONPATH', '')
    r = subprocess.run(eval_cmd(cat, gpu, out), capture_output=True, text=True,
                       cwd=DRAEM_REPO, env=env, timeout=timeout)
    metric = None
    for line in r.stdout.splitlines():
        if line.strip().startswith('METRIC:'):
            metric = json.loads(line.strip()[7:])
    if metric is None:
        return None, (r.stderr or r.stdout)[-500:]
    metric['category'] = cat
    metric['checkpoint'] = os.path.join(out, 'checkpoint')
    json.dump(metric, open(os.path.join(out, 'eval.json'), 'w'), indent=2)
    return metric, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='all', choices=['train', 'eval', 'all'])
    ap.add_argument('--gpus', default='2,3')
    ap.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    ap.add_argument('--categories', default=','.join(CATS))
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    cats = [c.strip() for c in args.categories.split(',')]

    os.makedirs(EXP_ROOT, exist_ok=True)
    global_cfg = {
        'experiment': 'G1_seed_variance',
        'question': 'fryum 的 S8 +0.1903 是否超过 plain baseline 自身种子波动?',
        'plan_ref': '/home/chenjiawen/workspace/实验方案.md §四/§五',
        'categories': cats, 'seeds': seeds, 'gpus': gpus,
        'epochs': EPOCHS, 'lr': LR,
        'batch_size_baseline': BS_DRAEM, 'batch_size_fix2': BS_FIX2,
        'mirror_weight': MIRROR_W, 'deterministic': True,
        'note': ('baseline 与 S8 的 steps_per_epoch 不同 (原生配方 vs fix2 配方); '
                 'G1 只比较同一方法内的 seed 波动, 跨方法的 steps 对齐由 G2 的 '
                 'B0\' 臂承担。'),
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json.dump(global_cfg, open(os.path.join(EXP_ROOT, 'config.json'), 'w'),
              indent=2, ensure_ascii=False)

    # 任务队列: baseline 先跑 (短, ~0.5h) 以便尽早看到 baseline 方差
    jobs = []
    for method in ('baseline', 'S8'):
        for cat in cats:
            for seed in seeds:
                jobs.append((cat, method, seed))

    if args.phase in ('train', 'all'):
        running = {}      # gpu -> (job, proc, f, t0)
        eval_failed = set()
        t_start = time.time()
        print(f'G1: {len(jobs)} runs 总, GPUs={gpus} '
              f'(每轮重算状态, 外部在跑的 run 会被接管而非重复启动)', flush=True)
        while True:
            # 每轮重新计算 todo, 而不是启动时算一次:
            #  - is_active  的 run (外部进程在训) 跳过, 等它自己跑完;
            #  - train_complete 的 run 交给下面的评估分支, 绝不重训。
            todo = [j for j in jobs if not os.path.exists(eval_path(*j))]
            if not todo and not running:
                break

            for gpu in gpus:
                if gpu in running or not todo:
                    continue
                # 显存闸门: 一个 DRAEM run 占 12.6G, 24G 卡上放两个直接 CUDA OOM。
                # 需要它是因为 GPU 池里可能包含"外面正跑着我自己的 run"或别的组的卡
                # —— 此时该 GPU 虽不在 running 里, 但显存已被占满。
                free = gpu_free_mib(gpu)
                if free is not None and free < MIN_FREE_MIB:
                    continue
                job = next((c for c in todo
                            if not is_active(*c) and not train_complete(*c)), None)
                if job is None:
                    continue
                cat, method, seed = job
                out = rundir(*job)
                os.makedirs(os.path.join(out, 'checkpoint'), exist_ok=True)
                log = os.path.join(out, 'train.log')
                cmd = train_cmd(cat, method, seed, gpu, out)
                write_run_config(cat, method, seed, gpu, out)
                f = open(log, 'w')
                f.write('CMD: ' + ' '.join(cmd) + '\n\n')
                f.flush()
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                        cwd=ROOT, start_new_session=True)
                running[gpu] = (job, proc, f, time.time())
                print(f'  [start] {cat}/{method}/seed{seed} gpu={gpu} pid={proc.pid} '
                      f'({time.time()-t_start:.0f}s)', flush=True)

            time.sleep(20)

            for gpu in list(running):
                job, proc, f, t0 = running[gpu]
                if proc.poll() is None:
                    continue
                f.close()
                cat, method, seed = job
                out = rundir(*job)
                dt = time.time() - t0
                cfg = json.load(open(os.path.join(out, 'run_config.json')))
                cfg['training_time_s'] = round(dt)
                cfg['train_rc'] = proc.returncode
                json.dump(cfg, open(os.path.join(out, 'run_config.json'), 'w'),
                          indent=2, ensure_ascii=False)
                status = 'ok' if proc.returncode == 0 else f'RC={proc.returncode}'
                print(f'  [done]  {cat}/{method}/seed{seed} gpu={gpu} {status} '
                      f'{dt/60:.1f}min', flush=True)
                running.pop(gpu)

            # 评估: 训完但缺 eval.json 的 run —— 既含 runner 自己跑的, 也含
            # runner 重启前就已启动、现已跑完的 (G1 首次启动的两个 baseline)。
            busy = {j for j, _, _, _ in running.values()}
            for job in jobs:
                if job in busy or os.path.exists(eval_path(*job)):
                    continue
                cat, method, seed = job
                out = rundir(*job)
                rec = os.path.join(out, 'checkpoint', f'{BASE_NAME}_{cat}_.pckl')
                if not os.path.exists(rec) or is_active(*job):
                    continue
                if not train_complete(*job) or job in eval_failed:
                    continue
                gpu = gpus[0]
                metric, err = run_eval(cat, gpu, out)
                if metric:
                    print(f"  [eval]  {cat}/{method}/seed{seed} "
                          f"pix={metric['pixel_auc']:.4f} "
                          f"img={metric['image_auc']:.4f}", flush=True)
                else:
                    eval_failed.add(job)
                    print(f'  [eval]  {cat}/{method}/seed{seed} FAILED: {err}',
                          flush=True)

    print(f'\n产物: {EXP_ROOT}', flush=True)


if __name__ == '__main__':
    main()
