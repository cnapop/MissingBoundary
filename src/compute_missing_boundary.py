#!/usr/bin/env python3
"""
Missing Boundary Score Computation (v3)
=========================================
基于固定归一化的 M(x) Oracle 计算 (切到 DBD 第二代 KDE-PB 定义):

  M(x) = alpha * Gap_norm(x) + beta * PB(x)
    Gap_norm: DRAEM b3/b4/b6 特征 → KNN(k) 到 normal bank 距离,
              固定锚定 normal 自距离分布 (P1/P99, 保存为常数)
    PB:       KDE-PB, PB(s) = 1 - 2|P(normal|score) - 0.5| (bandwidth=0.1)

同时导出 MxOracle bundle 到 missing_boundary/mx_oracle/, 供生成闭环
对每张生成样本用同一套固定定义评分 (消除 train/test 分集归一化不可比).

输出:
  missing_boundary/missing_boundary.csv   逐图像 M 与分量
  missing_boundary/high_m_regions.json    高-M 参考图 (BN=top train, BD=top test)
  missing_boundary/pb_scores.csv / gap_scores.csv
  missing_boundary/mx_oracle/oracle.pkl   M(x) oracle bundle
"""
import argparse, os, sys, json, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mx_oracle import MxOracle


def main():
    parser = argparse.ArgumentParser(description='Compute Missing Boundary scores (v3, KDE-PB)')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--bandwidth', type=float, default=0.1)
    parser.add_argument('--percentile', type=int, default=90)
    parser.add_argument('--max_train_images', type=int, default=209,
                        help='Train images to build normal bank (MVTec 209, VisA ~450)')
    args = parser.parse_args()

    feature_dir = args.output_dir
    category = args.category
    mb_out = os.path.join(feature_dir, category, 'missing_boundary')
    oracle_dir = os.path.join(mb_out, 'mx_oracle')
    os.makedirs(mb_out, exist_ok=True)

    # 1. 构建 oracle (固定定义, 重算所有 train/test 的 M)
    print("=" * 60)
    print(f"Phase 1 (v3): 构建 M(x) Oracle (KDE-PB, percentile=P{args.percentile})")
    print("=" * 60)
    oracle, image_data = MxOracle.build_and_save(
        feature_dir, category, oracle_dir,
        alpha=args.alpha, beta=args.beta, k=args.k,
        bandwidth=args.bandwidth, percentile=args.percentile,
        max_train_images=args.max_train_images,
    )

    test_results = sorted(image_data['test'], key=lambda r: r['M'], reverse=True)
    train_results = sorted(image_data['train'], key=lambda r: r['M'], reverse=True)

    print("\nTop-10 test by M(x):")
    for r in test_results[:10]:
        print(f"    {r['img_name']}: score={r['score']:.4f}, PB={r['pb']:.4f}, "
              f"Gap={r['gap']:.4f}, M={r['M']:.4f}")
    print("\nTop-10 train by M(x):")
    for r in train_results[:10]:
        print(f"    {r['img_name']}: score={r['score']:.4f}, PB={r['pb']:.4f}, "
              f"Gap={r['gap']:.4f}, M={r['M']:.4f}")

    # 2. missing_boundary.csv
    csv_path = os.path.join(mb_out, 'missing_boundary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'img_name', 'anomaly_score', 'pb', 'gap_mean', 'gap_max', 'M', 'split'])
        writer.writeheader()
        for r in test_results:
            writer.writerow({'img_name': r['img_name'], 'anomaly_score': r['score'],
                             'pb': r['pb'], 'gap_mean': r['gap'], 'gap_max': r['gap_max'],
                             'M': r['M'], 'split': 'test'})
        for r in train_results:
            writer.writerow({'img_name': r['img_name'], 'anomaly_score': r['score'],
                             'pb': r['pb'], 'gap_mean': r['gap'], 'gap_max': r['gap_max'],
                             'M': r['M'], 'split': 'train_good'})

    # 3. pb / gap CSV
    pb_csv = os.path.join(mb_out, 'pb_scores.csv')
    with open(pb_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'pb_mean', 'split'])
        for r in test_results:
            writer.writerow([r['img_name'], r['pb'], 'test'])
        for r in train_results:
            writer.writerow([r['img_name'], r['pb'], 'train_good'])
    gap_csv = os.path.join(mb_out, 'gap_scores.csv')
    with open(gap_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'gap_mean', 'split'])
        for r in test_results:
            writer.writerow([r['img_name'], r['gap'], 'test'])
        for r in train_results:
            writer.writerow([r['img_name'], r['gap'], 'train_good'])

    # 4. 高-M 参考图 (用 oracle 的固定阈值, 保证 ref 与闭环同一把尺子)
    #    协议修正: m_accept_bd 现在与 m_accept_bn 同为 train M 的 P{percentile}.
    #    blind_defect 列表仅为**分析**输出 (test 中超过 train 分位 M 的缺陷图),
    #    生成闭环 (generate_mb_closedloop.py) 已不再读取它 —— 生成不接触任何 test 信息.
    train_threshold = oracle.state['m_accept_bn']    # P{percentile} train
    test_threshold = oracle.state['m_accept_bd']     # P{percentile} train (与 BN 同尺)
    test_high_m = [r for r in test_results if r['M'] >= test_threshold]
    train_high_m = [r for r in train_results if r['M'] >= train_threshold]
    boundary_normal = train_high_m                     # 高-M train 正常图 (BN 与 BD 参考图均出自此)
    blind_defect = [r for r in test_high_m if r['score'] >= 0.5]  # 分析: test 高-M 缺陷图

    print(f"\n  Test threshold (P{args.percentile}, train 锚定): {test_threshold:.4f}")
    print(f"  Train threshold (P{args.percentile}): {train_threshold:.4f}")
    print(f"  Boundary Normal (train high-M, 兼作 BD 参考图源): {len(boundary_normal)}")
    print(f"  Blind Defect (分析输出: test high-M, score>=0.5): {len(blind_defect)}")

    high_m_data = {
        'test_threshold': float(test_threshold),
        'train_threshold': float(train_threshold),
        'num_high_m_test': len(test_high_m),
        'num_high_m_train': len(train_high_m),
        'num_boundary_normal': len(boundary_normal),
        'num_blind_defect': len(blind_defect),
        'alpha': args.alpha, 'beta': args.beta, 'k_neighbors': args.k,
        'mb_version': 'v3_kde_pb',
        'boundary_normal_samples': [
            {'img_name': r['img_name'], 'M': r['M'], 'score': r['score']}
            for r in boundary_normal],
        'blind_defect_samples': [
            {'img_name': r['img_name'], 'M': r['M'], 'score': r['score']}
            for r in blind_defect],
    }
    json_path = os.path.join(mb_out, 'high_m_regions.json')
    with open(json_path, 'w') as f:
        json.dump(high_m_data, f, indent=2)

    print(f"\nAll results saved to: {mb_out}")
    print(f"  missing_boundary.csv, high_m_regions.json, mx_oracle/oracle.pkl")
    return {'test': test_results, 'train_good': train_results}, high_m_data


if __name__ == '__main__':
    main()
