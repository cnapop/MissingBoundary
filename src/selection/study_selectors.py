#!/usr/bin/env python3
"""
selectors.py — Selection Study Phase B: S0-S9
==============================================
从 candidate.csv 候选池中, 用 10 种筛选策略各选出 K 个 BN (boundary normal,
进 train/good) 与 K 个 BD (blind defect, 进 MirrorEM 双流)。

统一框架 (numpy, 无 pandas 依赖):
  - bn_cand = 内容正常候选 (mask_area < AREA_DEF)
  - bd_cand = 缺陷候选 (mask_area >= AREA_DEF)
  每个 selector 在相应池里按自己的证据/评分排序, 取 Top-K (或随机)。

证据定义 (越高越像缺陷):
  E1   = mask_area
  E4   = z(mask_area) + z(score)                    # 面积 + 检测器响应
  E5   = z(mask_area) + z(topk01) + z(amap_max)     # + 局部 Top-K/Max
  E6   = z(region_mean) + z(region_max)             # 区域内部证据
  E7   = z(area)+z(region_mean)+z(topk01)+z(amap_max)+z(max_comp_frac)+z(compactness)

返回: 每 selector → (bn_idx, bd_idx)  (池内行索引列表, 各长度 K)。
"""
import numpy as np

AREA_DEF = 0.002        # 缺陷存在门 (与 clean fix2 rmp_thr_bd 对齐)
AREA_SMALL = 0.01       # 小缺陷判定: mask_area < 1%
M_ACCEPT_BN = 0.357     # train M P90 (clean oracle); 门不过时回退排序
M_ACCEPT_BD = 0.357


class Pool:
    """candidate.csv 的内存表示: {col: np.array}, 支持 filter/z/sort/sample."""

    def __init__(self, rows):
        self.cols = list(rows[0].keys()) if rows else []
        self.n = len(rows)
        self.data = {}
        for c in self.cols:
            try:
                self.data[c] = np.array([float(r[c]) for r in rows])
            except (TypeError, ValueError):
                self.data[c] = np.array([r[c] for r in rows])
        self.float_cols = {c for c in self.cols if self.data[c].dtype.kind == 'f'}

    def z(self, col, idx=None):
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        v = self.data[col][idx].astype(float)
        sd = v.std()
        return (v - v.mean()) / (sd + 1e-9)


def _topk(values, K, ascending=False):
    """返回 Top-K 的下标 (稳定)."""
    order = np.argsort(values, kind='stable')
    if not ascending:
        order = order[::-1]
    return order[:K]


def _evidence(pool, idx, name):
    z = pool.z
    i = np.asarray(idx)
    if name == 'area':
        return z('mask_area', i)
    if name == 'area_score':
        return z('mask_area', i) + z('score', i)
    if name == 'topk':
        return z('mask_area', i) + z('topk01', i) + z('amap_max', i)
    if name == 'region':
        return z('region_mean', i) + z('region_max', i)
    if name == 'multi':
        return (z('mask_area', i) + z('region_mean', i) + z('topk01', i)
                + z('amap_max', i) + z('max_comp_frac', i) + z('compactness', i))
    raise ValueError(name)


def _split(pool):
    area = pool.data['mask_area']
    bn_idx = np.where(area < AREA_DEF)[0]
    bd_idx = np.where(area >= AREA_DEF)[0]
    return bn_idx, bd_idx


def _rng(seed):
    return np.random.RandomState(seed)


def select_random(pool, K, seed=42):
    bn, bd = _split(pool)
    r = _rng(seed)
    bn = r.choice(bn, size=min(K, len(bn)), replace=False)
    r = _rng(seed + 1)
    bd = r.choice(bd, size=min(K, len(bd)), replace=False)
    return bn.tolist(), bd.tolist()


def select_area(pool, K, **kw):
    bn, bd = _split(pool)
    bn = bn[_topk(pool.data['mask_area'][bn], K, ascending=True)]
    bd = bd[_topk(pool.data['mask_area'][bd], K, ascending=False)]
    return bn.tolist(), bd.tolist()


def select_m_only(pool, K, **kw):
    bn, bd = _split(pool)
    bn = bn[_topk(pool.data['M'][bn], K, ascending=False)]
    bd = bd[_topk(pool.data['M'][bd], K, ascending=False)]
    return bn.tolist(), bd.tolist()


def _gate_fill(pool, cand, K, gate):
    M = pool.data['M'][cand]
    g = cand[M >= gate]
    if len(g) >= K:
        return g[_topk(pool.data['M'][g], K, ascending=False)]
    return cand[_topk(M, K, ascending=False)]


def select_m_area(pool, K, **kw):
    bn, bd = _split(pool)
    bn = _gate_fill(pool, bn, K, M_ACCEPT_BN)
    bd = _gate_fill(pool, bd, K, M_ACCEPT_BD)
    return bn.tolist(), bd.tolist()


def _bn_by_m_and_neg_ev(pool, cand, K, ev):
    s = pool.z('M', cand) - _evidence(pool, cand, ev)
    return cand[_topk(s, K, ascending=False)]


def _bd_by_m_and_ev(pool, cand, K, ev):
    s = pool.z('M', cand) + _evidence(pool, cand, ev)
    return cand[_topk(s, K, ascending=False)]


def select_m_score(pool, K, **kw):
    bn, bd = _split(pool)
    return _bn_by_m_and_neg_ev(pool, bn, K, 'area_score'), _bd_by_m_and_ev(pool, bd, K, 'area_score')


def select_m_topk(pool, K, **kw):
    bn, bd = _split(pool)
    return _bn_by_m_and_neg_ev(pool, bn, K, 'topk'), _bd_by_m_and_ev(pool, bd, K, 'topk')


def select_m_region(pool, K, **kw):
    bn, bd = _split(pool)
    return _bn_by_m_and_neg_ev(pool, bn, K, 'region'), _bd_by_m_and_ev(pool, bd, K, 'region')


def select_m_multi(pool, K, **kw):
    bn, bd = _split(pool)
    return _bn_by_m_and_neg_ev(pool, bn, K, 'multi'), _bd_by_m_and_ev(pool, bd, K, 'multi')


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def select_m_soft(pool, K, **kw):
    bn, bd = _split(pool)
    s_all = pool.z('M', np.arange(pool.n)) + _evidence(pool, np.arange(pool.n), 'multi')
    P = _sigmoid(s_all)
    bnP = P[bn]
    bdP = P[bd]
    bn_hard = bn[_topk(bnP, K, ascending=True)]          # 最低 P_D → 最"正常"
    bn_rest = bn[_topk(bnP, K * 2, ascending=True)]
    # 若 hard (<0.2) 不足, 从全池按 P_D 升序补
    bn_sel = np.concatenate([bn_hard, bn_rest]) if len(bn_hard) < K else bn_hard
    bn_sel = np.unique(bn_sel)[:K]
    if len(bn_sel) < K:
        bn_sel = np.concatenate([bn_sel, bn[~np.isin(bn, bn_sel)]])[:K]
    bd_hard = bd[_topk(bdP, K, ascending=False)]
    bd_sel = bd_hard if len(bd_hard) >= K else np.concatenate([bd_hard, bd])[:K]
    return bn_sel.tolist(), bd_sel.tolist()


def _pareto(points):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return []
    keep = np.ones(n, bool)
    for i in range(n):
        dominated = (pts[:, 0] >= pts[i, 0]) & (pts[:, 1] >= pts[i, 1]) & (
            (pts[:, 0] > pts[i, 0]) | (pts[:, 1] > pts[i, 1]))
        if dominated.any():
            keep[i] = False
    return np.where(keep)[0]


def select_m_pareto(pool, K, **kw):
    bn, bd = _split(pool)
    ev_bd = _evidence(pool, bd, 'multi')
    bd_pts = np.stack([pool.z('M', bd), ev_bd], axis=1)
    f_idx = _pareto(bd_pts)
    bd_f = bd[f_idx]
    bd_f = bd_f[_topk(pool.data['M'][bd_f], K, ascending=False)]
    if len(bd_f) < K:
        bd_f = bd[_topk(pool.data['M'][bd], K, ascending=False)]
    ev_bn = _evidence(pool, bn, 'multi')
    bn_pts = np.stack([pool.z('M', bn), -ev_bn], axis=1)
    f_idx = _pareto(bn_pts)
    bn_f = bn[f_idx]
    bn_f = bn_f[_topk(pool.data['M'][bn_f], K, ascending=False)]
    if len(bn_f) < K:
        bn_f = bn[_topk(pool.data['M'][bn], K, ascending=False)]
    return bn_f.tolist(), bd_f.tolist()


SELECTORS = {
    'S0_random': select_random,
    'S1_area': select_area,
    'S2_m_only': select_m_only,
    'S3_m_area': select_m_area,
    'S4_m_score': select_m_score,
    'S5_m_topk': select_m_topk,
    'S6_m_region': select_m_region,
    'S7_m_multi': select_m_multi,
    'S8_m_soft': select_m_soft,
    'S9_m_pareto': select_m_pareto,
}

SELECTOR_NAMES = list(SELECTORS.keys())


def run_all(pool, K=100, seed=42):
    """所有 selector → {name: (bn_idx, bd_idx)}."""
    return {name: fn(pool, K, seed=seed) for name, fn in SELECTORS.items()}


def screening_stats(pool, results):
    """Level-1 统计: {name: {...}}."""
    n = pool.n
    stats = {}
    for name, (bn, bd) in results.items():
        bn, bd = np.asarray(bn), np.asarray(bd)
        bd_area = pool.data['mask_area'][bd]
        stats[name] = {
            'n_bn': int(len(bn)), 'n_bd': int(len(bd)),
            'accept_rate': round(float((len(bn) + len(bd)) / n), 4),
            'bd_small_frac': round(float((bd_area < AREA_SMALL).mean()), 4),
            'bd_median_area': round(float(np.median(bd_area)), 5),
            'bn_median_M': round(float(np.median(pool.data['M'][bn])), 4),
            'bd_median_M': round(float(np.median(pool.data['M'][bd])), 4),
            'bd_median_region': round(float(np.median(pool.data['region_mean'][bd])), 4),
            'bd_median_score': round(float(np.median(pool.data['score'][bd])), 4),
            'bd_median_amap_max': round(float(np.median(pool.data['amap_max'][bd])), 4),
        }
    return stats


def set_overlap(results):
    """两两 selector BD 集合 Jaccard (用行索引)."""
    names = list(results.keys())
    sets = {nm: set(bd) for nm, (bn, bd) in results.items()}
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sets[names[i]], sets[names[j]]
            out[f'{names[i]}/{names[j]}'] = round(len(a & b) / max(len(a | b), 1), 3)
    return out
