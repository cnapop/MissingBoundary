#!/usr/bin/env python3
"""
Synthetic Anomaly Score Extraction (train-only, for protocol-clean KDE-PB)
==========================================================================
修复 test 统计泄漏: 原 mx_oracle 用 test 缺陷图的 score 拟合 kde_anom (PB 的异常侧
分布), 任何 test 统计参与训练数据构造都违反严格 MVTec 协议. 本脚本改用 **训练集正常图
的 Perlin×DTD 合成异常** —— 即 baseline DRAEM 在自身训练时见过的那类异常 —— 打分,
产出 train-only 的异常 score 分布, 供 build_and_save 替换原 test 缺陷拟合.

与特征提取使用同一 baseline DRAEM (同 checkpoint + base_model_name + category),
保证 score 分布与 train_good/scores.csv (kde_normal) 同源可比.

输出:
  {feature_dir}/{category}/train_good/synthetic_scores.csv
    img_name,anomaly_score   (img_name 仅可读性, 构建时按值数组拟合 KDE)

用法 (VisA pipe_fryum):
  python src/extract_synthetic_scores.py \
    --category pipe_fryum \
    --data_dir outputs/visa_datasets \
    --checkpoint_path outputs/checkpoints/visa \
    --base_model_name DRAEM_test_0.0001_200_bs8 \
    --feature_dir outputs/features_visa \
    --gpu_id <GPU>
"""
import argparse, os, sys, csv
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../DRAEM')
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from data_loader import MVTecDRAEMTrainDataset


def main():
    p = argparse.ArgumentParser(description='Score DRAEM synthetic anomalies (train-only)')
    p.add_argument('--category', required=True)
    p.add_argument('--data_dir', default='/data/chenjiawen/MissingBoundary/outputs/visa_datasets',
                   help='含 {category}/train/good/*.png 的根目录')
    p.add_argument('--anomaly-source-path', default='/data/chenjiawen/DRAEM/datasets/dtd/images')
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--base_model_name', required=True)
    p.add_argument('--feature_dir', required=True,
                   help='写入 synthetic_scores.csv 的特征根目录 (与 extract_features 的 output_dir 相同)')
    p.add_argument('--n_epochs', type=int, default=8,
                   help='遍历数据集轮数 (每轮 ~50% 样本注入异常)')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    device = f'cuda:{args.gpu_id}'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(device)  # 与 fix2 一致, SSIM/loss 侧若有隐式 .cuda() 落到正确卡

    run = f"{args.base_model_name}_{args.category}_"
    rec = ReconstructiveSubNetwork(3, 3)
    rec.load_state_dict(torch.load(os.path.join(args.checkpoint_path, run + '.pckl'),
                                   map_location=device))
    rec.to(device); rec.eval()
    seg = DiscriminativeSubNetwork(6, 2)
    seg.load_state_dict(torch.load(os.path.join(args.checkpoint_path, run + '_seg.pckl'),
                                   map_location=device))
    seg.to(device); seg.eval()

    dataset = MVTecDRAEMTrainDataset(
        os.path.join(args.data_dir, args.category, 'train', 'good'),
        args.anomaly_source_path,
        resize_shape=[256, 256],
    )
    print(f"[synthetic] normal images={len(dataset.image_paths)} "
          f"dtd sources={len(dataset.anomaly_source_paths)} gpu={args.gpu_id}")

    scores = []
    src_idx = []
    with torch.no_grad():
        for epoch in range(args.n_epochs):
            for _ in range(len(dataset)):
                s = dataset.__getitem__(0)  # __getitem__ 内部随机采样 normal+source
                if s['has_anomaly'][0] != 1.0:
                    continue
                x = torch.tensor(s['augmented_image']).unsqueeze(0).to(device)
                g = rec(x)
                j = torch.cat([g.detach(), x], dim=1)
                o = seg(j)
                sm = torch.softmax(o, dim=1)
                amap_avg = F.avg_pool2d(sm[:, 1:, :, :], 21, stride=1, padding=21 // 2)
                scores.append(float(amap_avg.max().item()))
                src_idx.append(int(s['idx']))
    scores = np.array(scores)
    print(f"[synthetic] has_anomaly samples={len(scores)} "
          f"score range [{scores.min():.4f}, {scores.max():.4f}] "
          f"mean={scores.mean():.4f} median={np.median(scores):.4f}")

    out_dir = os.path.join(args.feature_dir, args.category, 'train_good')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'synthetic_scores.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['img_name', 'anomaly_score'])
        for k, (sc, ix) in enumerate(zip(scores, src_idx)):
            w.writerow([f'synthetic/{k:04d}.png', f'{sc:.6f}'])
    print(f"[synthetic] saved {len(scores)} rows -> {csv_path}")


if __name__ == '__main__':
    main()
