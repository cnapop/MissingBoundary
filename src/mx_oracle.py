#!/usr/bin/env python3
"""
M(x) Oracle — 固定归一化的 Missing Boundary 评分器
=====================================================
把 M(x) 定义固化为一个可复用、可保存的评分器, 让"参考图选择"与"生成样本
接受/拒绝"使用完全相同的定义, 消除 train/test 分集归一化带来的不可比问题。

定义 (与 DBD 第二代一致):
  - Gap(x):   DRAEM b3/b4/b6 多尺度特征 → 32x32 融合像素向量,
              KNN(k=10) 到 normal bank 的平均距离.
              固定归一化: gap_norm = (d - P1_train) / (P99_train - P1_train),
              锚定 normal 自距离分布, 保存 P1/P99 常数.
  - PB(x):    KDE-PB, 图像级 anomaly score → P(normal|score),
              PB = 1 - 2|P(normal|score) - 0.5| (bandwidth=0.1).
  - M(x)      = alpha * Gap_norm + beta * PB.

用法:
  1) 构建 (phase1 内调用): 基于已提取的 train/test 特征 + 分数, 无需 DRAEM 模型.
     MxOracle.build_and_save(feature_dir, category, output_dir, ...)
  2) 运行时评分 (生成闭环内调用): 加载 DRAEM 模型, 对新生成的图逐张评分.
     oracle = MxOracle.load(oracle_dir, draem_ckpt_dir=..., base_model_name=..., ...)
     r = oracle.score('/path/to/generated.png')   # {'score','gap','pb','M'}
"""
import os, pickle, sys
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity

DEFAULT_TARGET = (32, 32)      # 融合特征分辨率
DEFAULT_K = 10                 # KNN 邻居数
DEFAULT_BANDWIDTH = 0.1        # KDE 带宽
BANK_SAMPLE = 5000             # normal bank 像素采样数
GAP_CLIP = 5.0                 # gap_norm 截断上限


def fuse_feature_dict(feat_dict, target=DEFAULT_TARGET):
    """把 b3/b4/b6 特征插值到 target 后按通道拼接, 返回 (H*W, C) 像素向量."""
    resized = []
    for name in ['b3', 'b4', 'b6']:
        ft = feat_dict[name]
        if ft.shape[1] != target[0] or ft.shape[2] != target[1]:
            ft = F.interpolate(torch.tensor(ft).unsqueeze(0), size=target,
                               mode='bilinear', align_corners=False).squeeze(0).numpy()
        resized.append(ft)
    fused = np.concatenate(resized, axis=0)
    C, H, W = fused.shape
    return fused.reshape(C, H * W).transpose(1, 0)


def load_feature_pixels(feat_dir, max_images=None):
    """加载目录下所有 *_feats.npy, 返回 (N_pixels, C) 像素向量."""
    paths = sorted(glob(os.path.join(feat_dir, '*_feats.npy')))
    if max_images is not None:
        paths = paths[:max_images]
    vecs = []
    for p in paths:
        vecs.append(fuse_feature_dict(np.load(p, allow_pickle=True).item()))
    return np.concatenate(vecs, axis=0) if vecs else np.empty((0, 0))


def load_scores(csv_path):
    """加载 scores.csv → {img_name: anomaly_score}."""
    import csv
    scores = {}
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                scores[row['img_name']] = float(row['anomaly_score'])
    return scores


def kde_pb_from_scores(score, kde_normal, kde_anom):
    """KDE-PB: PB(s) = 1 - 2|P(normal|s) - 0.5|, log-sum-exp 稳定."""
    s = np.array([[float(score)]])
    log_pn = kde_normal.score_samples(s)
    if kde_anom is not None:
        log_pa = kde_anom.score_samples(s)
        m = np.maximum(log_pn, log_pa)
        p_norm = np.exp(log_pn - m) / (np.exp(log_pn - m) + np.exp(log_pa - m) + 1e-8)
    else:
        p_norm = np.exp(log_pn) / (np.exp(log_pn) + 1e-8)
    return float(np.clip(1.0 - 2.0 * np.abs(p_norm - 0.5), 0.0, 1.0))


class MxOracle:
    """固定归一化 M(x) 评分器.

    state 字段:
      bank, knn_pkl, gap_p1, gap_p99, kde_normal_pkl, kde_anom_pkl,
      alpha, beta, k, bandwidth,
      m_accept_bn, m_accept_bd, bn_relax, bd_relax, normal_score_max
      (kde_anom 拟合自训练集正常图的 Perlin×DTD 合成异常 score;
       m_accept_bd / bd_relax 锚定 train M 分位 —— 与 BN 同一把 M 尺子,
       任何 test 统计都不参与训练数据构造, 满足严格 MVTec 协议)
    """

    def __init__(self, state=None, draem_dir=None, checkpoint_dir=None,
                 base_model_name=None, category=None, device='cuda:0'):
        self.state = state or {}
        self.device = device
        self._rec = None
        self._seg = None
        self._knn = None
        self._kde_normal = None
        self._kde_anom = None
        self._draem_dir = draem_dir
        self._ckpt_dir = checkpoint_dir
        self._base_name = base_model_name
        self._category = category

    # -------------------------------------------------------------
    # 懒加载
    # -------------------------------------------------------------
    def _load_models(self):
        if self._rec is not None:
            return
        sys.path.insert(0, self._draem_dir or '/data/chenjiawen/DRAEM')
        from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
        run = f"{self._base_name}_{self._category}_"
        ckpt = self._ckpt_dir or '/data/chenjiawen/DRAEM/checkpoints'
        rec = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        rec.load_state_dict(torch.load(os.path.join(ckpt, run + '.pckl'), map_location=self.device))
        rec.to(self.device); rec.eval()
        seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2, out_features=True)
        seg.load_state_dict(torch.load(os.path.join(ckpt, run + '_seg.pckl'), map_location=self.device))
        seg.to(self.device); seg.eval()
        seg.out_features = True
        self._rec, self._seg = rec, seg

    def _load_knn(self):
        if self._knn is None:
            bank = self.state['bank']
            # n_jobs=1: kneighbors 的 joblib/loky 进程池会与并发 fork SeaS 子进程的
            # 其他线程死锁 (见 selection_pool 挂起诊断)。bank 仅 ~100 向量, 单线程足够快。
            self._knn = NearestNeighbors(n_neighbors=self.state.get('k', DEFAULT_K), n_jobs=1)
            self._knn.fit(bank)

    def _load_kde(self):
        if self._kde_normal is None:
            self._kde_normal = self.state['kde_normal_pkl']
            self._kde_anom = self.state.get('kde_anom_pkl')

    # -------------------------------------------------------------
    # DRAEM 特征 + 分数提取 (对新图)
    # -------------------------------------------------------------
    def _extract(self, image_path):
        """返回 (fused_pixel_vec (1024,C), image_score, amap_np (H,W))."""
        self._load_models()
        img = Image.open(image_path).convert('RGB').resize((256, 256), Image.BILINEAR)
        x = torch.tensor(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            rec = self._rec(x)
            joined = torch.cat([rec.detach(), x], dim=1)
            out, b2, b3, b4, b5, b6 = self._seg(joined)
            sm = torch.softmax(out, dim=1)
            amap = sm[0, 1:, :, :]                       # (1,H,W)
            amap_avg = F.avg_pool2d(amap, 21, stride=1, padding=21 // 2)
            score = amap_avg.max().item()
            amap_np = amap[0, 0].cpu().numpy()           # (H,W) 原始概率图
            feats = {}
            for name, f in [('b3', b3), ('b4', b4), ('b6', b6)]:
                feats[name] = F.interpolate(f.detach(), size=DEFAULT_TARGET,
                                            mode='bilinear', align_corners=False)[0].cpu().numpy()
        return fuse_feature_dict(feats), score, amap_np

    # -------------------------------------------------------------
    # 单张评分
    # -------------------------------------------------------------
    def _score_from_vec(self, vec, score, amap_np, amap_thr=0.5):
        """从已提取特征计算 M(x) 分量 (供 score / score_full 共用)."""
        self._load_knn()
        self._load_kde()
        d, _ = self._knn.kneighbors(vec)
        d_mean = d.mean(axis=1)
        p1, p99 = self.state['gap_p1'], self.state['gap_p99']
        gap_pix = np.clip((d_mean - p1) / (p99 - p1 + 1e-8), 0.0, GAP_CLIP)
        gap = float(gap_pix.mean())
        pb = kde_pb_from_scores(score, self._kde_normal, self._kde_anom)
        alpha = self.state.get('alpha', 0.5)
        beta = self.state.get('beta', 0.5)
        M = alpha * gap + beta * pb
        amap_cov = float((amap_np > amap_thr).mean())
        return {'score': score, 'gap': gap, 'pb': pb, 'M': M, 'amap_cov': amap_cov}

    def score(self, image_path, amap_thr=0.5):
        """对新图计算 {score, gap, pb, M, amap_cov}.

        amap_cov: DRAEM 像素异常图概率 > amap_thr 的像素占比 (缺陷存在信号).
        """
        vec, score, amap_np = self._extract(image_path)
        return self._score_from_vec(vec, score, amap_np, amap_thr)

    def score_full(self, image_path, amap_thr=0.5):
        """score() + amap: 额外返回 DRAEM 像素异常图 (供 selection study 提特征).

        返回 {score, gap, pb, M, amap_cov, amap_np (H,W float)}.
        """
        vec, score, amap_np = self._extract(image_path)
        r = self._score_from_vec(vec, score, amap_np, amap_thr)
        r['amap_np'] = amap_np
        return r

    def score_many(self, paths):
        return [self.score(p) for p in paths]

    # -------------------------------------------------------------
    # 保存 / 加载
    # -------------------------------------------------------------
    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'oracle.pkl'), 'wb') as f:
            pickle.dump(self.state, f)

    @classmethod
    def load(cls, oracle_dir, **runtime_kwargs):
        with open(os.path.join(oracle_dir, 'oracle.pkl'), 'rb') as f:
            state = pickle.load(f)
        return cls(state=state, **runtime_kwargs)

    @staticmethod
    def _percentile_thresholds(M_vals, percentile=90):
        levels = [percentile - 10, percentile - 5, percentile]
        qs = np.percentile(M_vals, levels)
        return {f'p{percentile}': float(qs[2]),
                f'p{percentile - 5}': float(qs[1]),
                f'p{percentile - 10}': float(qs[0])}

    # -------------------------------------------------------------
    # 构建 (基于已提取特征 + 分数, 无需 DRAEM 模型)
    # -------------------------------------------------------------
    @classmethod
    def build_and_save(cls, feature_dir, category, out_dir,
                       alpha=0.5, beta=0.5, k=DEFAULT_K, bandwidth=DEFAULT_BANDWIDTH,
                       bank_sample=BANK_SAMPLE, max_train_images=209, seed=42,
                       percentile=90):
        train_feat_dir = os.path.join(feature_dir, category, 'train_good', 'features')
        test_feat_dir = os.path.join(feature_dir, category, 'test', 'features')
        train_score_csv = os.path.join(feature_dir, category, 'train_good', 'scores.csv')
        test_score_csv = os.path.join(feature_dir, category, 'test', 'scores.csv')
        synthetic_score_csv = os.path.join(feature_dir, category, 'train_good', 'synthetic_scores.csv')

        print("=" * 60)
        print("MxOracle.build_and_save — 固定归一化 M(x) 评分器")
        print("=" * 60)

        # 1. Normal bank + KNN
        print("[1/6] 构建 normal bank (KNN k={}) ...".format(k))
        train_pix = load_feature_pixels(train_feat_dir, max_images=max_train_images)
        print("  train pixels:", train_pix.shape)
        rng = np.random.RandomState(seed)
        idx = rng.choice(train_pix.shape[0], min(bank_sample, train_pix.shape[0]), replace=False)
        bank = train_pix[idx]
        knn = NearestNeighbors(n_neighbors=min(k, bank.shape[0]), n_jobs=1)
        knn.fit(bank)

        # 2. 固定 gap 归一化常数 (锚定 normal 自距离分布)
        print("[2/6] 计算固定 gap 归一化常数 (P1/P99 over normal self-distance) ...")
        d_train, _ = knn.kneighbors(train_pix)
        d_train_mean = d_train.mean(axis=1)
        gap_p1, gap_p99 = np.percentile(d_train_mean, [1, 99])
        print(f"  gap_p1={gap_p1:.4f}  gap_p99={gap_p99:.4f}")

        def gap_norm(d_mean):
            return np.clip((d_mean - gap_p1) / (gap_p99 - gap_p1 + 1e-8), 0.0, GAP_CLIP)

        # 3. KDE-PB: 正常/异常 score 分布
        #    异常侧 KDE 用训练集正常图的 Perlin×DTD 合成异常 score (train-only),
        #    不再使用 test 缺陷图 —— 消除 test 统计对训练数据构造的污染 (严格 MVTec 协议).
        print("[3/6] 拟合 KDE-PB (bandwidth={}, 异常侧=合成异常 train-only) ...".format(bandwidth))
        train_scores = load_scores(train_score_csv)
        test_scores = load_scores(test_score_csv)
        synthetic_scores = load_scores(synthetic_score_csv)
        normal_scores = np.array(list(train_scores.values())).reshape(-1, 1)
        anom_scores = (np.array(list(synthetic_scores.values())).reshape(-1, 1)
                       if synthetic_scores else None)
        kde_normal = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_normal.fit(normal_scores)
        kde_anom = None
        if anom_scores is not None and len(anom_scores) > 5:
            kde_anom = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde_anom.fit(anom_scores)
        n_anom = len(anom_scores) if anom_scores is not None else 0
        if anom_scores is not None:
            print(f"  normal n={len(normal_scores)}, synthetic-anomaly n={n_anom}"
                  f"  (score range: normal [{normal_scores.min():.3f},{normal_scores.max():.3f}],"
                  f" anom [{anom_scores.min():.3f},{anom_scores.max():.3f}])")
        else:
            print(f"  normal n={len(normal_scores)}, synthetic-anomaly n=0 (退化, PB≈0)")

        # 4. 用同一固定定义重算所有 train/test 的 M
        print("[4/6] 重算 train/test 的 M (固定归一化) ...")

        def image_Ms(pix_vectors, names, scores_dict):
            """逐图像 (gap, gap_max, pb, M, score)."""
            d, _ = knn.kneighbors(pix_vectors)
            raw = d.mean(axis=1)
            gaps = gap_norm(raw)
            out = []
            start = 0
            for name in names:
                end = start + 1024
                seg = gaps[start:end]
                gap = float(seg.mean())
                sc = scores_dict.get(name, 0.5)
                pb = kde_pb_from_scores(sc, kde_normal, kde_anom)
                out.append({'img_name': name, 'score': sc, 'gap': gap,
                            'gap_max': float(seg.max()), 'pb': pb,
                            'M': alpha * gap + beta * pb})
                start = end
            return out

        # train images
        train_paths = sorted(glob(os.path.join(train_feat_dir, '*_feats.npy')))
        train_pix_all = np.concatenate([fuse_feature_dict(np.load(p, allow_pickle=True).item())
                                        for p in train_paths], axis=0)
        # 注意: names 需与 scores.csv 的 img_name 对齐 (train/good/203.png)
        train_names = []
        for p in train_paths:
            stem = os.path.basename(p).replace('_feats.npy', '')
            parts = stem.split('_')
            idx = parts[-1]
            train_names.append(f"train/good/{idx}.png")
        train_data = image_Ms(train_pix_all, train_names, train_scores)

        test_paths = sorted(glob(os.path.join(test_feat_dir, '*_feats.npy')))
        test_pix_all = np.concatenate([fuse_feature_dict(np.load(p, allow_pickle=True).item())
                                       for p in test_paths], axis=0)
        test_names = []
        for p in test_paths:
            stem = os.path.basename(p).replace('_feats.npy', '')
            # stem 形如 test_broken_large_000 → test/broken_large/000.png
            parts = stem.split('_')
            idx = parts[-1]
            defect = '_'.join(parts[1:-1])
            test_names.append(f"test/{defect}/{idx}.png")
        test_data = image_Ms(test_pix_all, test_names, test_scores)

        train_Ms = np.array([d['M'] for d in train_data])
        test_Ms = np.array([d['M'] for d in test_data])
        train_sc = [d['score'] for d in train_data]
        test_sc = [d['score'] for d in test_data]

        print(f"  train M: [{train_Ms.min():.4f}, {train_Ms.max():.4f}] mean={train_Ms.mean():.4f}")
        print(f"  test  M: [{test_Ms.min():.4f}, {test_Ms.max():.4f}] mean={test_Ms.mean():.4f}")

        # 5. 阈值 — m_accept_bd 与 m_accept_bn 锚定同一把 M 尺子 (train 分位),
        #    BD 的判别完全交给 mask_cov>rmp_thr_bd + score∈band 两个门 (见生成闭环).
        print("[5/6] 导出阈值 (BD/BN 共用 train M 分位) ...")
        bn_thr = cls._percentile_thresholds(train_Ms, percentile)
        bd_thr = bn_thr
        normal_score_max = float(np.array(train_sc).max())

        # 6. 组装 state 并保存
        print("[6/6] 保存 oracle ...")
        state = {
            'bank': bank,
            'knn_pkl': knn,
            'gap_p1': gap_p1, 'gap_p99': gap_p99,
            'kde_normal_pkl': kde_normal, 'kde_anom_pkl': kde_anom,
            'alpha': alpha, 'beta': beta, 'k': k, 'bandwidth': bandwidth,
            'category': category, 'percentile': percentile,
            'm_accept_bn': bn_thr[f'p{percentile}'],
            'm_accept_bd': bd_thr[f'p{percentile}'],
            'bn_relax': [bn_thr[f'p{percentile}'], bn_thr[f'p{percentile - 5}'],
                         bn_thr[f'p{percentile - 10}']],
            'bd_relax': [bd_thr[f'p{percentile}'], bd_thr[f'p{percentile - 5}'],
                         bd_thr[f'p{percentile - 10}']],
            'normal_score_max': normal_score_max,
            'n_train': int(len(train_Ms)), 'n_test': int(len(test_Ms)),
        }
        oracle = cls(state=state)
        oracle.save(out_dir)

        # 附一份 JSON 摘要便于人工查看
        import json
        summary = {k: v for k, v in state.items()
                   if not isinstance(v, (np.ndarray, KernelDensity, NearestNeighbors))}
        summary['knn'] = 'pickled'
        for arr_name in ['bank']:
            summary[arr_name] = list(state[arr_name].shape)
        with open(os.path.join(out_dir, 'oracle_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print("  oracle 保存至:", out_dir)
        print("  m_accept_bn (train P90)={:.4f}  m_accept_bd (train P90, 与 BN 同尺)={:.4f}".format(
            bn_thr['p90'], bd_thr['p90']))
        return oracle, {'train': train_data, 'test': test_data}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build M(x) oracle from extracted features')
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=DEFAULT_K)
    parser.add_argument('--bandwidth', type=float, default=DEFAULT_BANDWIDTH)
    parser.add_argument('--max_train_images', type=int, default=209)
    args = parser.parse_args()
    MxOracle.build_and_save(
        args.feature_dir, args.category, args.output_dir,
        alpha=args.alpha, beta=args.beta, k=args.k, bandwidth=args.bandwidth,
        max_train_images=args.max_train_images,
    )
