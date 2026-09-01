#!/usr/bin/env python3
"""supervise_candle.py — candle 8 模型 (4 L2 + 4 L1) 统一容量感知训练监督器.

GPU 4/6/7 空闲, 每卡 2 槽 (fix2 ~10.7GB, 2×10.7=21.4GB < 24.5GB).
任务 = L2 (run_study) + L1 (run_endpoint_l1) × S0/S5/S8/S9.
"""
import os, sys, time, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'selection'))
import run_study
import run_endpoint_l1

CAPACITY = {4: 2, 6: 2, 7: 2}
POLL_S = 60
NAMES = ['S0_random', 'S5_m_topk', 'S8_m_soft', 'S9_m_pareto']
LOG_DIR = os.path.join(ROOT, 'outputs', '_candle_train_logs')
os.makedirs(LOG_DIR, exist_ok=True)


def build_tasks():
    tasks = []
    for cond, train_fn, root in [('L2', run_study.train_cmd, run_study.SEL_ROOT),
                                 ('L1', run_endpoint_l1.train_cmd, run_endpoint_l1.SEL_L1_ROOT)]:
        for name in NAMES:
            ckpt = os.path.join(root, name, 'checkpoints', 'fix2')
            tasks.append({'key': f'{cond}_{name}', 'cond': cond, 'name': name,
                          'train': train_fn, 'ckpt': ckpt})
    return tasks


def main():
    tasks = build_tasks()
    print(f'tasks: {[t["key"] for t in tasks]}')
    gpu_slots = dict(CAPACITY)
    running = {}
    done, failed = [], []
    start = time.time()
    tasks = list(tasks)

    def launch(t, gpu):
        ret = t['train'](t['name'], gpu)
        cmd = ret[0] if isinstance(ret, tuple) else ret   # run_study:(cmd,ckpt) / l1:cmd
        log = os.path.join(LOG_DIR, f"{t['key']}.train.log")
        # 清理崩溃残留 checkpoint
        for p in ('.pckl', '_seg.pckl'):
            f = os.path.join(t['ckpt'], f'{run_study.BASE}_{run_study.CAT}_{p}')
            if os.path.exists(f):
                os.remove(f)
        with open(log, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    cwd=ROOT, start_new_session=True)
        running[t['key']] = (proc, gpu, t)
        gpu_slots[gpu] -= 1
        print(f'  [{t["key"]}] gpu={gpu} pid={proc.pid} 槽余{gpu_slots[gpu]} '
              f'({time.time()-start:.0f}s)', flush=True)

    # 首轮填满
    for t in list(tasks):
        for g in sorted(gpu_slots):
            if gpu_slots[g] > 0:
                tasks.remove(t); launch(t, g); break
    print(f'首轮 {len(running)}/8, 队列剩 {len(tasks)}', flush=True)

    while tasks or running:
        time.sleep(POLL_S)
        for key in list(running):
            proc, gpu, t = running[key]
            rc = proc.poll()
            if rc is None:
                continue
            gpu_slots[gpu] += 1
            running.pop(key)
            if rc == 0:
                done.append(key)
                print(f'  [done] {key} ({time.time()-start:.0f}s)', flush=True)
            else:
                failed.append(key)
                try:
                    tail = open(os.path.join(LOG_DIR, f'{key}.train.log')).read()[-400:]
                except OSError:
                    tail = '(no log)'
                print(f'  [FAIL] {key} rc={rc} -> {tail}', flush=True)
            if tasks:
                nxt = tasks.pop(0)
                if gpu_slots[gpu] > 0:
                    launch(nxt, gpu)
                else:
                    for g in sorted(gpu_slots):
                        if gpu_slots[g] > 0:
                            launch(nxt, g); break

    with open(os.path.join(LOG_DIR, 'supervise_done.json'), 'w') as f:
        json.dump({'done': done, 'failed': failed, 'wall_s': round(time.time() - start)}, f, indent=2)
    print(f'\n全部完成: {len(done)} done / {len(failed)} failed, {time.time()-start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
