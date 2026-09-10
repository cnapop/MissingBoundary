#!/usr/bin/env python3
"""
run_cross_visa.py — VisA 全类交叉验证编排器
=============================================
在 VisA 多个类别上顺序/并行执行 MissingBoundary 闭环管线:
  prep → baseline → features → synthetic → mx → seas_gen → seas_mask
      → pool → select → fix2 → eval → baseline_eval

设计要点 (吸取 g3 教训):
  - 幂等: 每个 stage 用"产物 done 标记"判定是否完成, 完成即跳过, 永不重跑.
  - 崩溃安全: stage 以 setsid 分离子进程运行 + pidfile 记录; 编排器进程死了,
    子进程继续跑并写 done 标记; 重启编排器自动续传.
  - GPU 分配: 全局分配器, 按 free-mem 阈值整卡独占, self.held 防止重复分配.
  - 并行: 每个类别一个线程, 共享分配器.

用法:
  {PY} run_cross_visa.py --categories capsules,pcb1,candle --selectors S0_random,S2_m_only,S4_m_score,S8_m_soft
  {PY} run_cross_visa.py --status        # 只打印各 stage 进度, 不启动任何东西

由 cron 监控 /tmp/monitor_cross.py 重启 (进程死亡且未完成时).
"""
import argparse, os, sys, json, csv, glob, time, subprocess, shutil, socket, threading

ROOT = '/data/chenjiawen/MissingBoundary'
SEAS = '/data/chenjiawen/SeaS'
VISA_ROOT = '/data/chenjiawen/Datasets/VisA'
SPLIT_CSV = os.path.join(VISA_ROOT, 'split_csv', '1cls.csv')
PY_DRAEM = '/home/chenjiawen/anaconda3/envs/DRAEM/bin/python'
PY_SEAS = '/home/chenjiawen/anaconda3/envs/seas/bin/python'
ACTIVATE = 'source /home/chenjiawen/anaconda3/bin/activate seas'
DTD = '/data/chenjiawen/DRAEM/datasets/dtd/images'
DRAEM_REPO = '/data/chenjiawen/DRAEM'
BASE = 'DRAEM_test_0.0001_200_bs8'
CKPT_DIR = os.path.join(ROOT, 'outputs', 'checkpoints', 'visa')
FEAT_DIR = os.path.join(ROOT, 'outputs', 'features_visa')
DATASETS = os.path.join(ROOT, 'outputs', 'visa_datasets')
VISA_SEAS = os.path.join(ROOT, 'outputs', 'visa_seas')
POOL_ROOT = os.path.join(ROOT, 'outputs', 'selection_pool_g3')
LOGS = os.path.join(ROOT, 'outputs', 'cross_visa', 'logs')
BASELINE_OUT = os.path.join(ROOT, 'outputs', 'cross_visa', 'baseline')
MARKS = os.path.join(ROOT, 'outputs', 'cross_visa', 'marks')
SUPERVISE_DONE_JSON = None  # 见 fix2 stage

# baseline 用 mark 文件标记"真正完成" (DRAEM 每 epoch 覆盖保存 ckpt,
# 文件存在 ≠ 训练完成 — 崩溃恢复时绝不能用 epoch-0 的模型去跑 features).
BASELINE_MARK = True

N_GPU = 8
FIX2_GPU_GB = 22.0     # fix2 2槽/卡 需要 ≥22GB 空闲
HEAVY_GPU_GB = 15.0    # baseline/seas/pool 需要 ≥15GB 空闲
LIGHT_GPU_GB = 8.0     # features/synthetic/eval 需要 ≥8GB 空闲

# 每个 stage 需要的 GPU 数
STAGE_NGPU = {
    'prep': 0, 'baseline': 1, 'features': 1, 'synthetic': 1, 'mx': 0,
    'seas_gen': 1, 'seas_mask': 1, 'pool': 4, 'select': 0, 'fix2': 4,
    'eval': 1, 'baseline_eval': 1,
}
# 每个 GPU stage 的最小空闲显存 (GB)
STAGE_MIN_GB = {
    'baseline': HEAVY_GPU_GB, 'features': LIGHT_GPU_GB, 'synthetic': LIGHT_GPU_GB,
    'seas_gen': HEAVY_GPU_GB, 'seas_mask': HEAVY_GPU_GB, 'pool': HEAVY_GPU_GB,
    'fix2': FIX2_GPU_GB, 'eval': LIGHT_GPU_GB, 'baseline_eval': LIGHT_GPU_GB,
}
STAGE_ORDER = ['prep', 'baseline', 'features', 'synthetic', 'mx',
               'seas_gen', 'seas_mask', 'pool', 'select', 'fix2', 'eval', 'baseline_eval']

# 已有资产可跳过的 stage (仅 candle/pipe_fryum)
# candle: visa_datasets/visa_seas 已转换, baseline ckpt 已有, SeaS ckpt 已有
#   -> 跳过 prep(会自动检测) baseline seas_gen seas_mask
# 通用: 各 stage 自身有 done 检测, 这里仅用于提示


class Allocator:
    """整卡独占分配器。锁保护, self.held 记录已分给本进程的 GPU。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.held = {}   # tag -> set(gpu)

    def _free_mem(self, gpu):
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode()
            for line in out.strip().splitlines():
                i, free = line.split(',')
                if int(i) == gpu:
                    return float(free)
        except Exception:
            pass
        return 0.0

    def acquire(self, n, tag, min_gb):
        """阻塞直到拿到 n 张空闲 GPU。返回 gpu 列表。"""
        while True:
            with self.lock:
                have = set()
                for g in range(N_GPU):
                    if any(g in s for s in self.held.values()):
                        continue
                    if self._free_mem(g) >= min_gb:
                        have.add(g)
                if len(have) >= n:
                    picked = sorted(have)[:n]
                    self.held[tag] = set(picked)
                    return picked
            time.sleep(60)

    def release(self, tag):
        with self.lock:
            self.held.pop(tag, None)


alloc = Allocator()


def _env(**kw):
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env.update(kw)
    return env


def log(cat, stage, msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {cat}:{stage} {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, 'driver.log'), 'a') as f:
        f.write(line + '\n')


def _pidfile(cat, stage):
    return os.path.join(LOGS, f'{cat}.{stage}.pid')


def _launch_and_wait(cat, stage, cmd, env, stdout_path):
    """分离启动 + 等待, 返回 (done_ok, rc)。pidfile 防双重启动。"""
    pidf = _pidfile(cat, stage)
    # 已有存活实例?
    if os.path.exists(pidf):
        try:
            old = int(open(pidf).read().strip())
            os.kill(old, 0)
            log(cat, stage, f'skip: 已有存活实例 pid={old}')
            while True:
                try:
                    os.kill(old, 0)
                except OSError:
                    break
                time.sleep(30)
            return True, 0
        except (OSError, ValueError):
            pass  # 死 pid, 重新启动
    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
    with open(stdout_path, 'a') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             env=env, cwd=ROOT, start_new_session=True)
    with open(pidf, 'w') as f:
        f.write(str(p.pid))
    log(cat, stage, f'launched pid={p.pid}: {" ".join(cmd[:3])} ...')
    rc = p.wait()
    return rc == 0, rc


def _row_count(csv_path):
    try:
        with open(csv_path) as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return 0


# ---------------- done 标记 ----------------
def done_prep(cat):
    g = os.path.join(DATASETS, cat, 'train', 'good', '000.png')
    gt = os.path.join(DATASETS, cat, 'ground_truth', 'anomaly')
    return os.path.exists(g) and glob.glob(os.path.join(gt, '*_mask.png'))


def _baseline_ckpt_ok(cat):
    for suffix in ('_.pckl', '__seg.pckl'):
        f = os.path.join(CKPT_DIR, f'{BASE}_{cat}{suffix}')
        if not (os.path.exists(f) and os.path.getsize(f) > 1e7):
            return False
    return True


def _baseline_complete(cat):
    """基线真正完成 = ckpt 存在 + 训练日志含 'Epoch: 199' (200 轮跑完)."""
    if not _baseline_ckpt_ok(cat):
        return False
    try:
        txt = open(os.path.join(LOGS, f'{cat}.baseline.log')).read()
        return 'Epoch: 199' in txt
    except OSError:
        return False


def _mark(cat, stage):
    return os.path.join(MARKS, f'{cat}.{stage}')


def _touch_mark(cat, stage):
    os.makedirs(MARKS, exist_ok=True)
    with open(_mark(cat, stage), 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S'))


def done_baseline(cat):
    """baseline 完成 = mark 文件存在 (ckpt 每 epoch 覆盖保存, 不能当完成标记)."""
    return os.path.exists(_mark(cat, 'baseline'))


def done_features(cat):
    return os.path.exists(os.path.join(FEAT_DIR, cat, 'train_good', 'scores.csv'))


def done_synthetic(cat):
    return os.path.exists(os.path.join(FEAT_DIR, cat, 'train_good', 'synthetic_scores.csv'))


def done_mx(cat):
    return os.path.exists(os.path.join(FEAT_DIR, cat, 'missing_boundary', 'high_m_regions.json'))


def done_seas_gen(cat):
    return os.path.exists(os.path.join(SEAS, 'outputs', 'checkpoints', cat,
                                       'generation-checkpoint', 'model_index.json'))


def done_seas_mask(cat):
    return os.path.exists(os.path.join(SEAS, 'outputs', 'checkpoints', cat,
                                       'mask-checkpoint', 'rmp'))


def done_pool(cat):
    return _row_count(os.path.join(POOL_ROOT, cat, 'candidate.csv')) >= 1000


def done_select(cat, selectors):
    for name in selectors:
        if not os.path.exists(os.path.join(SEL_ROOT(cat), name, 'bd_manifest.csv')):
            return False
    return True


def done_fix2(cat, selectors):
    for name in selectors:
        p = os.path.join(SEL_ROOT(cat), '_logs', f'{name}.train.log')
        try:
            if '[fix2] done' not in open(p).read():
                return False
        except OSError:
            return False
    return True


def done_eval(cat, selectors):
    p = os.path.join(SEL_ROOT(cat), 'selection_results.json')
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
        return all(s in d and d[s] for s in selectors)
    except (ValueError, KeyError):
        return False


def done_baseline_eval(cat):
    return os.path.exists(os.path.join(BASELINE_OUT, f'{cat}.json'))


def SEL_ROOT(cat):
    return os.path.join(ROOT, 'outputs', f'selection_g3_{cat}')


# ---------------- stage 执行 ----------------
def run_prep(cat):
    """VisA→MVTec/SeaS 转换 + ground_truth 桥接 (eval 的 loader 需要)."""
    cmd = [PY_DRAEM, 'src/closedloop_v3/prepare_visa.py',
           '--visa_root', VISA_ROOT, '--split_csv', SPLIT_CSV, '--category', cat,
           '--out_datasets', DATASETS, '--out_seas', VISA_SEAS]
    ok, rc = _launch_and_wait(cat, 'prep', cmd, _env(), os.path.join(LOGS, f'{cat}.prep.log'))
    if not ok:
        return False
    gt = os.path.join(DATASETS, cat, 'ground_truth', 'anomaly')
    os.makedirs(gt, exist_ok=True)
    # 桥接 eval 用 GT mask 后, 删除 test/anomaly 内联 _mask.png (prepare_visa 因 SeaS
    # 掩码对齐而内联写入). DRAEM MVTec 测试 loader globs test/*/*.png, 会把内联 mask
    # 当作测试图 → 其 mask 路径 *_mask_mask.png 缺失 → cv2.resize 崩溃 (capsules 实测
    # 2026-09-02). ground_truth 副本已桥接, 删除安全 (eval 前无任何阶段读 test/anomaly).
    inline = glob.glob(os.path.join(DATASETS, cat, 'test', 'anomaly', '*_mask.png'))
    for m in inline:
        shutil.copy2(m, os.path.join(gt, os.path.basename(m)))
        os.remove(m)
    return done_prep(cat)


def run_baseline(cat, gpu):
    cmd = [PY_DRAEM, 'src/closedloop_v3/train_draem_single.py',
           '--obj_name', cat, '--bs', '8', '--lr', '0.0001', '--epochs', '200',
           '--gpu_id', str(gpu), '--data_path', DATASETS,
           '--anomaly_source_path', DTD, '--checkpoint_path', CKPT_DIR,
           '--log_path', os.path.join(ROOT, 'outputs', 'logs', 'visa')]
    ok, rc = _launch_and_wait(cat, 'baseline', cmd, _env(), os.path.join(LOGS, f'{cat}.baseline.log'))
    if ok and _baseline_complete(cat):
        _touch_mark(cat, 'baseline')
        return True
    return False


def run_features(cat, gpu):
    # 注: --max_train_images 传大值(100000)禁抽样 → 每类用全量训练集建特征/
    # M(x) oracle (与 pipe_fryum 的"450=全量"语义一致; 默认 200/209 会截断).
    cmd = [PY_DRAEM, 'src/extract_features.py',
           '--category', cat, '--data_path', DATASETS, '--checkpoint_path', CKPT_DIR,
           '--base_model_name', BASE, '--output_dir', FEAT_DIR,
           '--gpu_id', str(gpu), '--max_train_images', '100000']
    ok, rc = _launch_and_wait(cat, 'features', cmd, _env(), os.path.join(LOGS, f'{cat}.features.log'))
    return ok and done_features(cat)


def run_synthetic(cat, gpu):
    cmd = [PY_DRAEM, 'src/extract_synthetic_scores.py',
           '--category', cat, '--data_dir', DATASETS, '--anomaly-source-path', DTD,
           '--checkpoint_path', CKPT_DIR, '--base_model_name', BASE,
           '--feature_dir', FEAT_DIR, '--gpu_id', str(gpu)]
    ok, rc = _launch_and_wait(cat, 'synthetic', cmd, _env(), os.path.join(LOGS, f'{cat}.synthetic.log'))
    return ok and done_synthetic(cat)


def run_mx(cat):
    cmd = [PY_DRAEM, 'src/compute_missing_boundary.py',
           '--category', cat, '--output_dir', FEAT_DIR, '--max_train_images', '100000']
    ok, rc = _launch_and_wait(cat, 'mx', cmd, _env(), os.path.join(LOGS, f'{cat}.mx.log'))
    return ok and done_mx(cat)


def _accel_yaml(gpu):
    p = f'/tmp/cross_accel_gpu{gpu}.yaml'
    if not os.path.exists(p):
        content = ("compute_environment: LOCAL_MACHINE\ndebug: false\n"
                   "distributed_type: 'NO'\ndowncast_bf16: 'no'\n"
                   f"gpu_ids: '{gpu}'\nmachine_rank: 0\n"
                   "main_training_function: main\nmixed_precision: 'no'\n"
                   "num_machines: 1\nnum_processes: 1\nrdzv_backend: static\n"
                   "same_network: true\ntpu_env: []\ntpu_use_cluster: false\n"
                   "tpu_use_sudo: false\nuse_cpu: false\n")
        with open(p, 'w') as f:
            f.write(content)
    return p


def run_seas_gen(cat, gpu):
    inst = os.path.join(VISA_SEAS, cat, 'instance')
    cmd = (f"cd {SEAS} && {ACTIVATE} && ACCELERATE_FORCE_NUM_PROCESSES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
           f"accelerate launch --config_file {_accel_yaml(gpu)} examples/SeaS_main.py "
           f"--output_dir=outputs/checkpoints/{cat} "
           f"--instance_data_dir={inst} "
           f"--mask_dir={os.path.join(VISA_SEAS, cat, 'mask')} "
           f"--normal_data_dir={os.path.join(VISA_SEAS, cat, 'normal')} "
           f"--gen_train_steps=800 --checkpointing_steps=800")
    ok, rc = _launch_and_wait(cat, 'seas_gen', ['bash', '-c', cmd], _env(),
                              os.path.join(LOGS, f'{cat}.seas_gen.log'))
    return ok and done_seas_gen(cat)


def run_seas_mask(cat, gpu):
    inst = os.path.join(VISA_SEAS, cat, 'instance')
    cmd = (f"cd {SEAS} && {ACTIVATE} && ACCELERATE_FORCE_NUM_PROCESSES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
           f"accelerate launch --config_file {_accel_yaml(gpu)} examples/SeaS_mask_main.py "
           f"--output_dir=outputs/checkpoints/{cat} "
           f"--instance_data_dir={inst} "
           f"--mask_dir={os.path.join(VISA_SEAS, cat, 'mask')} "
           f"--normal_data_dir={os.path.join(VISA_SEAS, cat, 'normal')} "
           f"--seas_trained_model_path=outputs/checkpoints/{cat}/generation-checkpoint "
           f"--mask_train_steps=800 --checkpointing_steps=800")
    ok, rc = _launch_and_wait(cat, 'seas_mask', ['bash', '-c', cmd], _env(),
                              os.path.join(LOGS, f'{cat}.seas_mask.log'))
    return ok and done_seas_mask(cat)


def run_pool(cat, gpus):
    gs = ','.join(str(g) for g in gpus)
    cmd = [PY_DRAEM, 'src/selection/build_candidate_pool.py',
           '--category', cat,
           '--high_m_json', os.path.join(FEAT_DIR, cat, 'missing_boundary', 'high_m_regions.json'),
           '--oracle_dir', os.path.join(FEAT_DIR, cat, 'missing_boundary', 'mx_oracle'),
           '--dataset_dir', DATASETS, '--output_dir', POOL_ROOT,
           '--gpus', gs, '--score_gpu', str(gpus[-1]),
           '--num_refs', '20', '--num_variants', '10']
    ok, rc = _launch_and_wait(cat, 'pool', cmd, _env(), os.path.join(LOGS, f'{cat}.pool.log'))
    return ok and done_pool(cat)


def run_select(cat, selectors):
    env = _env(MB_CAT=cat, MB_POOL_CSV=os.path.join(POOL_ROOT, cat, 'candidate.csv'),
               MB_SEL_ROOT=SEL_ROOT(cat), MB_SELECTORS=','.join(selectors))
    cmd = [PY_DRAEM, 'src/selection/run_study.py', '--phase', 'select']
    ok, rc = _launch_and_wait(cat, 'select', cmd, env, os.path.join(LOGS, f'{cat}.select.log'))
    return ok and done_select(cat, selectors)


def run_fix2(cat, selectors, gpus):
    unfinished = [s for s in selectors if not _fix2_one_done(cat, s)]
    if not unfinished:
        return True
    # 1 模型/卡铺开 (fix2 真实瓶颈是 num_workers=0 的 CPU 串行增强, GPU 空转;
    # 铺卡让每模型独占 SM, 再配合 MB_NUM_WORKERS=4 并行增强 → 每步 ~0.15s).
    cap = {str(g): 1 for g in gpus}
    env = _env(MB_CAT=cat, MB_SEL_ROOT=SEL_ROOT(cat),
               MB_SELECTORS=','.join(unfinished), MB_CAPACITY=json.dumps(cap),
               MB_NUM_WORKERS='4')
    cmd = [PY_DRAEM, 'src/selection/supervise_train.py']
    ok, rc = _launch_and_wait(cat, 'fix2', cmd, env, os.path.join(LOGS, f'{cat}.fix2.log'))
    return ok and done_fix2(cat, selectors)


def _fix2_one_done(cat, name):
    p = os.path.join(SEL_ROOT(cat), '_logs', f'{name}.train.log')
    try:
        return '[fix2] done' in open(p).read()
    except OSError:
        return False


def run_eval(cat, selectors, gpu):
    env = _env(MB_CAT=cat, MB_SEL_ROOT=SEL_ROOT(cat),
               MB_SELECTORS=','.join(selectors), MB_GPUS=str(gpu))
    cmd = [PY_DRAEM, 'src/selection/run_study.py', '--phase', 'eval']
    ok, rc = _launch_and_wait(cat, 'eval', cmd, env, os.path.join(LOGS, f'{cat}.eval.log'))
    return ok and done_eval(cat, selectors)


def run_baseline_eval(cat, gpu):
    env = _env(MB_CAT=cat, PYTHONPATH=DRAEM_REPO + os.pathsep + env_or('PYTHONPATH', ''))
    cmd = [PY_DRAEM, 'src/selection/eval_study.py', '--gpu_id', str(gpu),
           '--base_model_name', BASE, '--data_path', DATASETS, '--checkpoint_path', CKPT_DIR]
    out = os.path.join(LOGS, f'{cat}.baseline_eval.log')
    ok, rc = _launch_and_wait(cat, 'baseline_eval', cmd, env, out)
    if not (ok and done_baseline_eval(cat)):
        # 从日志解析 METRIC
        try:
            txt = open(out).read()
            for line in txt.splitlines():
                if line.startswith('METRIC:'):
                    metric = json.loads(line[7:])
                    metric['category'] = cat
                    os.makedirs(BASELINE_OUT, exist_ok=True)
                    json.dump(metric, open(os.path.join(BASELINE_OUT, f'{cat}.json'), 'w'), indent=2)
                    return True
        except Exception as e:
            log(cat, 'baseline_eval', f'parse failed: {e}')
        return False
    return True


def env_or(k, d):
    return os.environ.get(k, d)


# ---------------- 驱动 ----------------
RUNNERS = {
    'prep': lambda cat, gpus: run_prep(cat),
    'baseline': lambda cat, gpus: run_baseline(cat, gpus[0]),
    'features': lambda cat, gpus: run_features(cat, gpus[0]),
    'synthetic': lambda cat, gpus: run_synthetic(cat, gpus[0]),
    'mx': lambda cat, gpus: run_mx(cat),
    'seas_gen': lambda cat, gpus: run_seas_gen(cat, gpus[0]),
    'seas_mask': lambda cat, gpus: run_seas_mask(cat, gpus[0]),
    'pool': lambda cat, gpus: run_pool(cat, gpus),
    'select': lambda cat, gpus: run_select(cat, selectors_for(cat)),
    'fix2': lambda cat, gpus: run_fix2(cat, selectors_for(cat), gpus),
    'eval': lambda cat, gpus: run_eval(cat, selectors_for(cat), gpus[0]),
    'baseline_eval': lambda cat, gpus: run_baseline_eval(cat, gpus[0]),
}

SELECTORS = None


def selectors_for(cat):
    return SELECTORS


def stage_done(cat, stage):
    fns = {
        'prep': done_prep, 'baseline': done_baseline, 'features': done_features,
        'synthetic': done_synthetic, 'mx': done_mx, 'seas_gen': done_seas_gen,
        'seas_mask': done_seas_mask, 'pool': done_pool,
        'select': lambda c: done_select(c, SELECTORS),
        'fix2': lambda c: done_fix2(c, SELECTORS),
        'eval': lambda c: done_eval(c, SELECTORS),
        'baseline_eval': done_baseline_eval,
    }
    return fns[stage](cat)


def run_category(cat):
    for stage in STAGE_ORDER:
        if stage_done(cat, stage):
            continue
        tag = f'{cat}:{stage}'
        n = STAGE_NGPU[stage]
        if n == 0:
            log(cat, stage, 'start (CPU)')
            ok = RUNNERS[stage](cat, [])
            if not ok:
                log(cat, stage, 'FAILED')
                return cat, stage, 'failed'
            continue
        min_gb = STAGE_MIN_GB[stage]
        gpus = alloc.acquire(n, tag, min_gb)
        log(cat, stage, f'start gpu={gpus}')
        try:
            ok = RUNNERS[stage](cat, gpus)
        finally:
            alloc.release(tag)
        if not ok:
            log(cat, stage, f'FAILED (gpu={gpus})')
            return cat, stage, 'failed'
    log(cat, 'ALL', 'complete')
    return cat, None, 'done'


def main():
    global SELECTORS
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', type=str, default='capsules,pcb1,candle')
    ap.add_argument('--selectors', type=str,
                    default='S0_random,S2_m_only,S4_m_score,S8_m_soft')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--max-parallel-fix2', type=int, default=4,
                    help='最多同时几个类别的 fix2 占用 GPU (每个 fix2 占 2 卡)')
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(',') if c.strip()]
    SELECTORS = [s.strip() for s in args.selectors.split(',') if s.strip()]

    if args.status:
        for cat in cats:
            line = [cat]
            for stage in STAGE_ORDER:
                d = stage_done(cat, stage)
                line.append(f'{stage}:{"OK" if d else ".."}')
            print(' '.join(line))
        return

    os.makedirs(LOGS, exist_ok=True)
    log('driver', 'start', f'categories={cats} selectors={SELECTORS} '
                           f'host={socket.gethostname()} pid={os.getpid()}')
    results = {}
    with threading.BoundedSemaphore(args.max_parallel_fix2):
        pass  # 简化: 不在这里做 fix2 并行上限, 由分配器隐式控制

    # 串行预热环境检查
    for cat in cats:
        if not os.path.exists(os.path.join(DATASETS, cat, 'train', 'good', '000.png')):
            pass  # prep 会建

    threads = []
    for cat in cats:
        t = threading.Thread(target=lambda c=cat: results.update({c: run_category(c)}))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    log('driver', 'end', json.dumps(results, ensure_ascii=False))
    for cat, stage, status in results.values():
        if status != 'done':
            print(f'[WARN] {cat} 停在 {stage}: {status}')


if __name__ == '__main__':
    main()
