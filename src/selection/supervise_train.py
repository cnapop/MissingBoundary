#!/usr/bin/env python3
"""
supervise_train.py — Selection Study 训练监督器
=================================================
在 GPU 容量受限 (每卡 ~24GB, fix2 每模型 ~10.7GB VRAM) 下, 把 10 个 selector
的 fix2 训练排队并逐个启动:
  - capacity = {gpu: 最多同时几个 fix2 进程}
  - 某进程退出 → 释放该 GPU 槽位 → 启动队列里下一个
  - 全部完成 → 写 done 标记并退出

历史教训 (selection_pool 研究):
  naive round-robin 把 3 个模型放到 GPU 0/2 → 3×~11GB + 其他用户占用 → OOM,
  5/10 训练崩溃。这里按真实显存容量 (每模型 10.7GB) 计算每卡可容纳数:
    GPU0 已有 2.8GB (他人) → 1 槽
    GPU1 空闲             → 2 槽
    GPU2 已有 5.8GB (他人) → 1 槽
    GPU6 已有 0.8GB       → 2 槽
    GPU7 空闲             → 2 槽
  (GPUs 3/4/5 被他人高占用, 不碰)

用法:
  /home/chenjiawen/anaconda3/envs/DRAEM/bin/python src/selection/supervise_train.py
"""
import os, sys, time, json, subprocess, signal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'selection'))
from run_study import SELECTOR_NAMES, train_cmd, SEL_ROOT, CAT, BASE

# 每卡最多并发 fix2 数 (实测每模型 ~10.7GB, 24.5GB 卡留 ~2GB 余量)
import json as _json
DEFAULT_CAP = {0: 1, 1: 2, 2: 1, 6: 2, 7: 2}
CAPACITY = _json.loads(os.environ.get('MB_CAPACITY', _json.dumps(DEFAULT_CAP)))
POLL_S = 60
LOG_DIR = os.path.join(SEL_ROOT, '_logs')


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    queue = list(SELECTOR_NAMES)
    running = {}          # name -> (proc, gpu)
    gpu_slots = {g: CAPACITY[g] for g in CAPACITY}
    done, failed = [], []
    start = time.time()

    def launch(name, gpu):
        cmd, ckpt = train_cmd(name, gpu)
        log = os.path.join(LOG_DIR, f'{name}.train.log')
        # 崩溃残留的 epoch-0 checkpoint 由 --init-random 覆盖, 无害; 先清掉保证干净
        for p in ('_seg.pckl', '.pckl'):
            f = os.path.join(ckpt, f'{BASE}_{CAT}_{p}')
            if os.path.exists(f):
                os.remove(f)
        with open(log, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    cwd=ROOT, start_new_session=True)
        running[name] = (proc, gpu)
        gpu_slots[gpu] -= 1
        print(f'  [{name}] gpu={gpu} pid={proc.pid} 槽余 {gpu_slots[gpu]} '
              f'({time.time()-start:.0f}s)', flush=True)

    # 首轮填满空闲槽
    for name in list(queue):
        for g in sorted(gpu_slots):
            if gpu_slots[g] > 0:
                queue.remove(name)
                launch(name, g)
                break

    print(f'首轮启动 {len(running)}/{len(SELECTOR_NAMES)}, 队列剩 {len(queue)}', flush=True)

    while queue or running:
        time.sleep(POLL_S)
        for name in list(running):
            proc, gpu = running[name]
            rc = proc.poll()
            if rc is None:
                continue
            gpu_slots[gpu] += 1
            running.pop(name)
            if rc == 0:
                done.append(name)
                print(f'  [done] {name} gpu={gpu} rc=0 ({time.time()-start:.0f}s)', flush=True)
            else:
                failed.append(name)
                print(f'  [FAIL] {name} gpu={gpu} rc={rc} '
                      f'-> tail: {_tail(os.path.join(LOG_DIR, name + ".train.log"))}', flush=True)
            # 释放槽位 → 补一个队列项
            if queue:
                nxt = queue.pop(0)
                # 优先用刚释放的 GPU (若还有槽)
                if gpu_slots[gpu] > 0:
                    launch(nxt, gpu)
                else:
                    for g in sorted(gpu_slots):
                        if gpu_slots[g] > 0:
                            launch(nxt, g)
                            break

    with open(os.path.join(SEL_ROOT, '_logs', 'supervise_done.json'), 'w') as f:
        json.dump({'done': done, 'failed': failed,
                   'wall_s': round(time.time() - start)}, f)
    print(f'\n全部完成: {len(done)} done / {len(failed)} failed, '
          f'{time.time()-start:.0f}s', flush=True)


def _tail(path, n=400):
    try:
        with open(path) as f:
            return f.read()[-n:].strip()
    except OSError:
        return '(no log)'


if __name__ == '__main__':
    main()
