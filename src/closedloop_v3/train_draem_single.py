#!/usr/bin/env python3
"""
单类别 DRAEM 训练入口 (供 VisA 等非 MVTec 类别)
==================================================
train_DRAEM.py 的 obj 列表只有 MVTec 15 类, 本脚本直接用 --obj_name 指定类别,
复用其 train_on_device。checkpoint 命名与 oracle 约定对齐:
  DRAEM_test_{lr}_{epochs}_bs{bs}_{obj_name}_.pckl
用法:
  python src/train_draem_single.py --obj_name pipe_fryum --bs 8 --lr 0.0001 \
      --epochs 200 --gpu_id 0 \
      --data_path outputs/visa_datasets \
      --anomaly_source_path /data/chenjiawen/DRAEM/datasets/dtd/images \
      --checkpoint_path outputs/checkpoints/visa \
      --log_path outputs/logs
"""
import argparse, os, random, sys, torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../DRAEM'))
from train_DRAEM import train_on_device


def set_seed(seed, deterministic=True):
    """固定全部随机源 (random / numpy / torch / cuda)。

    DRAEM 侧原本没有任何 seed 控制, 三个"同配置"重跑会因 shuffle=True +
    num_workers=16 自然发散, 无法记录也无法复现 → G1 (baseline seed 方差)
    无法执行。注意 DRAEM 数据管线的随机源是 numpy
    (data_loader.py 的 np.random.choice + rand_perlin_2d_np), 只设 torch 不够。

    deterministic 与 fix2 (train_draem_fix2.py) 保持一致: cuDNN 默认可能选
    非确定性卷积算法, 微小浮点差会经 200 epoch 放大成不同模型, 使"seed 方差"
    混入算法非确定性 → 两种方法的 std 不可比。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    print(f"[seed] seed={seed} deterministic={deterministic} "
          f"(random/numpy/torch/cuda 全部固定)", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Single-category DRAEM training')
    parser.add_argument('--obj_name', type=str, required=True)
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--anomaly_source_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=42,
                        help='训练种子 (random/numpy/torch/cuda 四源, 默认 42)')
    parser.add_argument('--no-deterministic', action='store_true',
                        help='关闭 cuDNN 确定性 (默认开启, 与 fix2 对齐)')
    args = parser.parse_args()

    set_seed(args.seed, deterministic=not args.no_deterministic)

    with torch.cuda.device(args.gpu_id):
        train_on_device([args.obj_name], args)


if __name__ == '__main__':
    main()
