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
import argparse, os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../DRAEM'))
from train_DRAEM import train_on_device


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
    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        train_on_device([args.obj_name], args)


if __name__ == '__main__':
    main()
