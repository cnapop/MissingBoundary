#!/usr/bin/env python3
"""
评估 sks 块 → 缺陷类型对应
============================
冻结 ResNet50 (ImageNet) 提取特征:
  - 每个训练子类的图像特征均值 → 质心原型
  - 每个 sks 块生成图 → 与各质心余弦最近邻分类
输出:
  - 10×10 混淆矩阵热图 (块 i → 预测子类)
  - 每块 top-1 命中率 / 全块准确率
  - 块内 vs 块间余弦相似度 (区分度)
  - 每块 montage (前 12 张)
用法:
  python src/closedloop_v3/eval_per_sks.py [--gen_dir outputs/subtype_gen/pipe_fryum]
"""
import argparse, json, os
from glob import glob
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRANS = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_feature_model(device):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    return model


@torch.no_grad()
def embed(model, paths, device, bs=32):
    feats = []
    for i in range(0, len(paths), bs):
        batch = [TRANS(Image.open(p).convert('RGB')) for p in paths[i:i + bs]]
        x = torch.stack(batch).to(device)
        f = model(x).cpu().numpy()
        feats.append(f)
    feats = np.concatenate(feats, 0)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return feats


def montage(paths, title, out_path, n=12):
    n = min(n, len(paths))
    paths = paths[:n]
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, p in zip(axes, paths):
        ax.imshow(Image.open(p))
    axes[0].set_title(title, fontsize=9)
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gen_dir', default=os.path.join(ROOT, 'outputs', 'subtype_gen', 'pipe_fryum'))
    p.add_argument('--instance_dir', default=os.path.join(ROOT, 'outputs', 'visa_seas_subtype', 'pipe_fryum', 'instance'))
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    types = sorted([d for d in os.listdir(args.instance_dir)
                    if os.path.isdir(os.path.join(args.instance_dir, d))])
    print(f'{len(types)} 子类: {types}')

    device = args.device
    model = load_feature_model(device)

    # ---- 子类质心原型 ----
    centroids, proto_counts = {}, {}
    for t in types:
        paths = sorted(glob(os.path.join(args.instance_dir, t, '*.png')))
        if not paths:
            continue
        f = embed(model, paths, device)
        centroids[t] = f.mean(0)
        centroids[t] = centroids[t] / (np.linalg.norm(centroids[t]) + 1e-12)
        proto_counts[t] = len(paths)
    type_list = list(centroids.keys())
    C = np.stack([centroids[t] for t in type_list])  # [K, 2048]

    # ---- 每个 sks 块生成图 ----
    block_dirs = sorted([d for d in os.listdir(args.gen_dir)
                         if os.path.isdir(os.path.join(args.gen_dir, d)) and d[0].isdigit()])
    cm = np.zeros((len(block_dirs), len(type_list)), dtype=int)
    report = {'types': type_list, 'blocks': {}, 'within_sim': {}, 'cross_sim': {}}
    all_block_feats = []

    for b, bdir in enumerate(block_dirs):
        img_paths = sorted(glob(os.path.join(args.gen_dir, bdir, 'image', '*.png')))
        if not img_paths:
            print(f'[block {bdir}] 无生成图')
            continue
        feats = embed(model, img_paths, device)
        sims = feats @ C.T  # [N, K]
        pred = sims.argmax(1)
        for pidx in pred:
            cm[b, pidx] += 1
        acc = (pred == np.array([b] * len(pred))).mean()  # 期望块 i → 子类 i
        report['blocks'][bdir] = {
            'n': len(img_paths),
            'top1_subtype': type_list[int(np.bincount(pred, minlength=len(type_list)).argmax())],
            'acc_expected': float(acc),
            'pred_dist': {type_list[k]: int(v) for k, v in enumerate(np.bincount(pred, minlength=len(type_list)))},
        }
        all_block_feats.append(feats)
        montage(img_paths, f'{bdir}  (acc={acc:.2f})',
                os.path.join(args.gen_dir, f'{bdir}_montage.png'))
        print(f'[block {bdir}] n={len(img_paths)} 期望子类命中={acc:.2f} '
              f'top1={type_list[int(np.bincount(pred, minlength=len(type_list)).argmax())]}')

    # ---- 块内 vs 块间余弦相似度 ----
    within, cross = [], []
    for a in range(len(all_block_feats)):
        fa = all_block_feats[a]
        within.append((fa @ fa.T).mean())
    for a in range(len(all_block_feats)):
        for b in range(a + 1, len(all_block_feats)):
            cross.append((all_block_feats[a] @ all_block_feats[b].T).mean())
    report['within_sim'] = {'mean': float(np.mean(within)) if within else None,
                            'per_block': [float(x) for x in within]}
    report['cross_sim'] = {'mean': float(np.mean(cross)) if cross else None}
    print(f'\n块内余弦相似度均值: {report["within_sim"]["mean"]:.4f} '
          f'(越高 → 同块生成越自洽)')
    print(f'块间余弦相似度均值: {report["cross_sim"]["mean"]:.4f} '
          f'(越低 → 块间差异越大)')

    # ---- 混淆矩阵热图 ----
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(type_list)), [t[:18] for t in type_list], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(block_dirs)), [os.path.basename(b)[:26] for b in block_dirs], fontsize=8)
    ax.set_xlabel('预测子类 (最近质心)')
    ax.set_ylabel('sks 块')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = os.path.join(args.gen_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=130)
    plt.close(fig)

    # ---- 真实训练图自检: 同特征同质心, 判断特征本身能否区分子类 ----
    self_cm = np.zeros((len(type_list), len(type_list)), dtype=int)
    report['selfcheck_acc'] = {}
    print('\n[真实训练图自检] 用训练图特征 vs 训练图质心 (评估特征区分度)')
    for i, t in enumerate(type_list):
        paths = sorted(glob(os.path.join(args.instance_dir, t, '*.png')))
        if not paths:
            continue
        feats = embed(model, paths, device)
        pred = (feats @ C.T).argmax(1)
        for pidx in pred:
            self_cm[i, pidx] += 1
        acc = (pred == i).mean()
        report['selfcheck_acc'][t] = float(acc)
        print(f'  [{i:02d}] {t}: 自检命中={acc:.2f} ({len(paths)} 图)')
    self_diag = np.trace(self_cm) / (self_cm.sum() + 1e-12)
    report['selfcheck_diag_acc'] = float(self_diag)
    print(f'\n真实图自检总体对角命中率: {self_diag:.3f} '
          f'(若低 → ImageNet 特征无法区分子类, 生成命中低不归咎于 sks)')

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(self_cm, cmap='Greens')
    ax.set_xticks(range(len(type_list)), [t[:18] for t in type_list], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(type_list)), [t[:18] for t in type_list], fontsize=8)
    ax.set_xlabel('预测子类')
    ax.set_ylabel('真实子类 (训练图)')
    for i in range(self_cm.shape[0]):
        for j in range(self_cm.shape[1]):
            ax.text(j, i, self_cm[i, j], ha='center', va='center', fontsize=8,
                    color='white' if self_cm[i, j] > self_cm.max() / 2 else 'black')
    plt.colorbar(im)
    plt.tight_layout()
    self_cm_path = os.path.join(args.gen_dir, 'selfcheck_confusion_matrix.png')
    plt.savefig(self_cm_path, dpi=130)
    plt.close(fig)

    diag = np.trace(cm) / (cm.sum() + 1e-12)
    report['overall_diag_acc'] = float(diag)
    report['confusion_matrix'] = cm.tolist()
    out_json = os.path.join(args.gen_dir, 'eval_report.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n总体对角命中率: {diag:.3f}')
    print(f'→ {out_json}')
    print(f'→ {cm_path}')
    print(f'→ {os.path.join(args.gen_dir, "*_montage.png")}')


if __name__ == '__main__':
    main()
