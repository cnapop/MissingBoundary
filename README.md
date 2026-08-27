# Missing Boundary Guided Generation for DRAEM

面向工业异常检测 (Industrial Anomaly Detection) 的 **Missing Boundary (缺失边界)** 引导生成方法。

**核心思想**: 异常检测模型最难处理的样本位于正常/异常决策边界附近——"缺失边界"区域。我们用 `M(x)` 度量一个样本离正常分布的"缺失程度"，并驱动生成模型 (SeaS) 产出真正位于边界的样本 (Boundary Normal, BN) 与难以检测的缺陷 (Blind Defect, BD)，用它们增强训练，提升检测器 (DRAEM)。

```
M(x) = α · Gap(x) + β · PB(x)
```

- `Gap(x)`: 样本特征到正常分布的密度距离 (k-NN, 固定 P1/P99 锚定归一化)
- `PB(x)`: 概率边界概率 (KDE-based, bandwidth=0.1)
- 默认 `α = β = 0.5`

**闭环流水线**: `特征提取 → 计算 M(x)/构建 oracle → SeaS 生成候选 → 逐张 M(x) 评分/接受/拒绝 → 数据增强 (BN→train/good, BD→双流) → DRAEM 重训 → 评估`。

> 检测器: [DRAEM](https://github.com/VitjanZ/DRAEM) ｜ 生成器: SeaS (Stable-Diffusion 1.4 微调)

---

## 目录结构

```
MissingBoundary/
├── README.md                      # 本文件: 实验总览 + 运行指南
├── config.sh                      # 中心配置 (conda 环境 / 路径 / 实验参数)
├── run_pipeline.sh                # 实验1: v1/v2 引导生成管线 (phase0-4)
├── docs/
│   ├── SUMMARY.md                 # 完整实验记录 (v1→v3→fix2→色差)
│   ├── NEXT_STEPS.md              # 续跑入口 + 全部命令/坑位
│   ├── EXPERIMENT_color_shift.md  # 实验4: 色差剂量-响应消融方案
│   ├── SELECTION_STUDY.md         # 实验5: 筛选策略研究方案
│   └── analysis_findings.md       # 早期生成质量分析
├── src/                           # 共享核心 (phase0-4 管线脚本)
│   ├── extract_features.py        #   Phase 0: DRAEM 特征 + 分数提取
│   ├── compute_missing_boundary.py#   Phase 1: M(x) 计算 + oracle 导出
│   ├── mx_oracle.py               #   M(x) 固定归一化评分器
│   ├── extract_synthetic_scores.py#   (协议干净版) train-only 合成异常打分
│   ├── generate_samples.py        #   (v1) SeaS 批量生成
│   ├── generate_mb_guided.py      #   (v2) M(x) 引导生成 + 共享工具
│   ├── prepare_and_retrain.py     #   Phase 3: 数据增强 + DRAEM 重训
│   ├── evaluate.py                #   Phase 4: 评估对比
│   ├── generate_report.py         #   报告生成
│   ├── visualize_mb.py            #   M(x)/Gap/PB 可视化
│   └── pipeline/results/          #   实验1 结果产物
├── src/closedloop_v3/             # 实验2: v3 M(x) 闭环生成
│   ├── generate_mb_closedloop.py  #   闭环: 生成→评分→接受/拒绝→自适应
│   ├── prepare_visa.py            #   VisA → MVTec+SeaS 布局转换
│   ├── train_draem_single.py      #   单类别 DRAEM 训练 (非 MVTec 类)
│   └── results/                   #   各类别闭环接受漏斗
├── src/fix2/                      # 实验3: fix2 双流训练
│   ├── train_draem_fix2.py        #   MirrorEM 双流训练 (BD→双流)
│   ├── fix2_prepare.py            #   BN→train/good + BD manifest 组装
│   └── results/                   #   manifest 样例 + 结果
├── src/color_shift/               # 实验4: 色差消融
│   ├── color_align_generated.py   #   Reinhard LAB 颜色校准
│   ├── diag_seg_floor.py          #   背景地板 / 分离比诊断
│   └── results/                   #   color_shift_report / seg_floor
└── src/selection/                 # 实验5: 筛选策略研究
    ├── build_candidate_pool.py    #   候选池构建 (1200 候选)
    ├── run_study.py               #   --phase select|train|eval
    ├── study_selectors.py         #   S0-S9 十种筛选策略
    ├── eval_study.py              #   策略评估
    ├── supervise_train.py         #   训练监督器 (GPU 容量感知)
    └── results/                   #   screening_stats.json
```

> 数据集、训练权重/检查点、生成图、特征缓存 **不纳入版本控制** (见 `.gitignore`)。各实验 `results/` 目录只保留小体积结果产物 (JSON/CSV/结果表)。

---

## 环境依赖

| 依赖 | 说明 |
|---|---|
| Conda env `DRAEM` | 编排器/评分/训练环境，需 `torch + sklearn + model_unet` |
| Conda env `seas` | SeaS 生成环境 (无 sklearn，仅子进程调用) |
| [DRAEM](https://github.com/VitjanZ/DRAEM) | 检测器，`/path/to/DRAEM` 需含 `model_unet.py`、`train_DRAEM.py`、`data_loader.py`、`datasets/dtd` |
| [SeaS](https://github.com/cnapop/SeaS-Replication) | 生成器，含训练好的 generation/mask checkpoint |
| 数据集 | [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)、[VisA](https://github.com/amazon-science/spot-diff) |
| GPU | 8×RTX4090 (0-7); 生成与评分分 GPU 并行 |

**路径配置**: 所有脚本的绝对路径默认值按本机 `/data/chenjiawen/...` 与 `/home/chenjiawen/...` 写死。换机器请修改 [config.sh](config.sh) 与各脚本 argparse 默认值。

---

## 实验总览

| # | 实验 | 核心脚本 | 一句话结论 |
|---|------|----------|-----------|
| 1 | v1/v2 引导生成管线 | `run_pipeline.sh` (phase0-4) | 管线跑通; v2 实测生成样本 M(x) 漂移, 锚定失效 → 触发 v3 闭环 |
| 2 | v3 M(x) 闭环生成 | `src/closedloop_v3/` | M(x) 成为生成运行时判据; bottle/pipe_fryum 接受漏斗可审计 |
| 3 | fix2 双流训练 | `src/fix2/` | BD→双流正确消费; bottle 通过, pipe_fryum 图像升像素降; 色差是 pixel 退化主因 |
| 4 | 色差剂量-响应消融 | `src/color_shift/` | L1 (半量对齐) 恢复 ~99% pixel 损失, 是实际最优 |
| 5 | 筛选策略研究 (S0-S9) | `src/selection/` | 池内 M↔mask 面积反相关 (corr≈−0.763), M 排序选到小缺陷 |

---

## 实验 1: v1/v2 引导生成管线 (phase0-4)

**目的**: 端到端跑通"特征 → M(x) → 生成 → 重训 → 评估"，是 v3 的前身 (open-loop: 生成样本的 M(x) 从不检查)。

**完整闭环**:
```bash
source config.sh
bash run_pipeline.sh full --category bottle --gpu 0
```

**分步执行** (任一步可单独跑):
```bash
# Phase 0: DRAEM 特征 + 分数提取
bash run_pipeline.sh phase0_extract --category bottle --gpu 0

# Phase 1: Missing Boundary 计算 (PB + Gap + M, 导出 oracle)
bash run_pipeline.sh phase1_mb --category bottle

# Phase 2: SeaS 生成 (v2 open-loop 或 v3 闭环, 见实验 2)
bash run_pipeline.sh phase2_generate --category bottle --gpu 0 --num-bn 100 --num-bd 100

# Phase 3: 数据增强 + DRAEM 重训
bash run_pipeline.sh phase3_retrain --category bottle --gpu 0

# Phase 4: 评估对比 (baseline vs random vs MB)
bash run_pipeline.sh phase4_evaluate --category bottle
```

**关键参数** (在 `config.sh` 中): `MB_ALPHA/MB_BETA` (M 融合权重), `MB_PERCENTILE` (高缺失阈值, 默认 90), `FEATURE_LAYERS` (多尺度特征层), `DRAEM_LR/EPOCHS/BS`。

**输出**: `outputs/features/{category}/missing_boundary/`、`outputs/generated/`、`outputs/evaluation/comparison_results.json`。早期结果见 `docs/analysis_findings.md`。

---

## 实验 2: v3 M(x) 闭环生成

**目的**: 修复 v2 缺陷——让 M(x) **驱动**生成而不是事后分组。每个高-M 参考图沿噪声阶梯生成候选，逐张用 oracle 评分并**接受/拒绝**：

```
BN:  M(x) ≥ m_accept_bn  AND mask_cov < rmp_thr_bn   (高-M 且内容正常)
BD:  M(x) ≥ m_accept_bd  AND mask_cov > rmp_thr_bd   (高-M 且缺陷存在)
              AND score 落在盲带 (难检测)
配额未满 → 继续阶梯 / 降级 M_accept (P90→P85→P80)
```

**⚠️ MVTec 协议合规 (train-only)**: 自 2026-08-24 起，管线**不使用任何 test 统计**参与生成 (合成异常 KDE、train M P90、BD 参考图=train 高-M normal)。每类别生成前须先跑合成异常打分。

**Step 0 — VisA 类别准备** (MVTec 类可跳过):
```bash
python src/closedloop_v3/prepare_visa.py --category pipe_fryum
python src/closedloop_v3/train_draem_single.py \
    --obj_name pipe_fryum --bs 8 --lr 0.0001 --epochs 200 --gpu_id 0 \
    --data_path outputs/visa_datasets \
    --anomaly_source_path /path/to/DRAEM/datasets/dtd/images \
    --checkpoint_path outputs/checkpoints/visa --log_path outputs/logs
```

**Step 1 — 合成异常打分 + M(x) oracle** (协议干净版):
```bash
python src/extract_synthetic_scores.py \
    --category pipe_fryum \
    --checkpoint_path /path/to/DRAEM/checkpoints \
    --base_model_name DRAEM_test_0.0001_200_bs8 \
    --feature_dir outputs/features_visa/pipe_fryum/train_good

python src/compute_missing_boundary.py \
    --category pipe_fryum --output_dir outputs/features_visa --max_train_images 450
# 产出: missing_boundary.csv, high_m_regions.json, mx_oracle/oracle.pkl
```

**Step 2 — M(x) 闭环生成**:
```bash
python src/closedloop_v3/generate_mb_closedloop.py \
    --category pipe_fryum \
    --high_m_json outputs/features_visa/pipe_fryum/missing_boundary/high_m_regions.json \
    --oracle_dir outputs/features_visa/pipe_fryum/missing_boundary/mx_oracle \
    --output_dir outputs/generated_v3_visa \
    --dataset_dir outputs/visa_datasets \
    --seas_dir /path/to/SeaS --seas_python /path/to/seas/bin/python \
    --draem_ckpt_dir outputs/checkpoints/visa --draem_base_name DRAEM_test_0.0001_200_bs8 \
    --draem_dir /path/to/DRAEM \
    --gpus 0,1,2,3 --score_gpu 7 \
    --num_bn 20 --num_bd 10 --num_variants 10 --quota_per_ref 8 --max_attempts 6 \
    --bn_noise_ladder 300,500,700 --bd_noise_ladder 1200,1500,1800 \
    --rmp_thr_bn 0.05 --rmp_thr_bd 0.002 --score_band 0.0,1.0
```

**输出**: `{output_dir}/{category}/generation_closedloop_summary.json` (接受漏斗 = M(x) 驱动证据)。各数据集结果副本见 `src/closedloop_v3/results/`。

**关键坑位** (详见 `docs/NEXT_STEPS.md`): ① SeaS 单张 ref 触发 einsum bug → ref 目录 ≥10 张; ② `--num_variants` ≥ SeaS batch_size; ③ 子进程调 SeaS 须 `cwd=seas_dir`; ④ 编排器必须用 DRAEM env (seas env 无 sklearn); ⑤ ≥1h 长任务用 `setsid nohup ... &` 脱离会话。

---

## 实验 3: fix2 双流训练

**目的**: 正确消费 BD 生成样本。早期 `mb_aug` 把 BD 错塞进 `train/good` 导致 Pixel AP 退化 (0.839→0.754)。正确做法 (**已验证**): **BN→train/good、BD→MirrorEM 双流** (image=ref, augmented=BD 图, mask=BD mask, focal 监督分割)。

**Step 1 — 数据准备**:
```bash
python src/fix2/fix2_prepare.py --category bottle
# 产出: train_good_plus_bn/ (209+8=217 张) + bd_manifest.csv
```

**Step 2 — 双流训练**:
```bash
python src/fix2/train_draem_fix2.py \
    --manifest outputs/fix2/bottle/bd_manifest.csv \
    --normal-data-dir outputs/fix2/bottle/train_good_plus_bn \
    --anomaly-source-path /path/to/DRAEM/datasets/dtd/images \
    --init-random --base-name DRAEM_test_0.0001_200_bs8 --category bottle \
    --output-dir outputs/fix2/bottle/checkpoints/fix2 \
    --epochs 200 --lr 0.0001 --mirror-weight 0.5 \
    --batch-size 4 --pre-batch-size 4 --draem-repo /path/to/DRAEM \
    --device cuda:7
```

**Step 3 — 评估** (DRAEM `test_DRAEM.py` 写死 `map_location='cuda:0'`，须用 `CUDA_VISIBLE_DEVICES` 把物理卡映射为 cuda:0):
```bash
CUDA_VISIBLE_DEVICES=7 python test_DRAEM.py --gpu_id 0 \
    --base_model_name DRAEM_test_0.0001_200_bs8 \
    --checkpoint_path outputs/fix2/bottle/checkpoints/fix2
```

**BN 比例消融**: 加 `--steps-per-epoch 203` 固定优化预算，只留 BN 比例一个变量 (5% vs 44%)。结论: **5% BN 是 pixel 退化主因** (AUC Pixel 0.657→0.321)。

**结果**: bottle 四指标均不劣于基线 (AP Pixel 0.8386→0.8415); pipe_fryum 图像级全升、AUC Pixel 退化 (见 `docs/SUMMARY.md §7.1/§7.4`)。Manifest 样例见 `src/fix2/results/`。

---

## 实验 4: 色差剂量-响应消融 (color shift)

**目的**: 隔离"生成图 VAE 偏色"是否为 fix2 pixel 退化主因。假设 H: 色差→重建残差→正常像素误报→背景地板抬升→pixel 降。用 Reinhard LAB 颜色传递，`t` 插值控制剂量。

**3 个水平**: L2 (自然色差, 已有 fix2_clean 结果) / L1 (`t=0.5` 半量) / L0 (`t=1.0` 全对齐)。

**Step 1 — 颜色校准**:
```bash
# L1 (半量): 保留一半色差
python src/color_shift/color_align_generated.py --category pipe_fryum \
    --level L1 --t 0.5 \
    --gen_root outputs/generated_v3_visa_clean \
    --out_root outputs/generated_v3_visa_clean_align_L1

# L0 (全对齐): 完全传递到参考图统计
python src/color_shift/color_align_generated.py --category pipe_fryum \
    --level L0 --t 1.0 \
    --gen_root outputs/generated_v3_visa_clean \
    --out_root outputs/generated_v3_visa_clean_align_L0
```

**Step 2 — 训练** (对齐后的 gen_root 跑 fix2_prepare → train_draem_fix2，命令同实验 3，仅换路径与 `--device`):
```bash
python src/fix2/fix2_prepare.py --category pipe_fryum \
    --gen_root outputs/generated_v3_visa_clean_align_L1 \
    --out_root outputs/fix2_align_L1
python src/fix2/train_draem_fix2.py \
    --manifest outputs/fix2_align_L1/pipe_fryum/bd_manifest.csv \
    --normal-data-dir outputs/fix2_align_L1/pipe_fryum/train_good_plus_bn \
    --anomaly-source-path /path/to/DRAEM/datasets/dtd/images \
    --init-random --base-name DRAEM_test_0.0001_200_bs8 --category pipe_fryum \
    --output-dir outputs/fix2_align_L1/pipe_fryum/checkpoints/fix2 \
    --epochs 200 --lr 0.0001 --mirror-weight 0.5 \
    --batch-size 4 --pre-batch-size 4 --draem-repo /path/to/DRAEM \
    --device cuda:7
```

**Step 3 — 背景地板诊断**:
```bash
python src/color_shift/diag_seg_floor.py \
    --checkpoint_path outputs/fix2_align_L1/pipe_fryum/checkpoints/fix2 \
    --category pipe_fryum --label L1 --out outputs/fix2_align_L1/pipe_fryum/seg_floor.json
```

**结果** (详见 `docs/SUMMARY.md §7.5`, 数据见 `src/color_shift/results/`):

| 指标 | baseline | L2 (自然) | **L1 (半量)** | L0 (全对齐) |
|---|---|---:|---:|---:|
| AUC Pixel | 0.6982 | 0.4913 | **0.7382** | 0.7399 |
| AP Pixel | 0.1012 | 0.0899 | **0.3076** | 0.3406 |
| AUC Image | 0.9336 | 0.9912 | **0.9594** | 0.8030 |

**判定: 支持 H** — 色差是 pixel 退化主因。半量 L1 即恢复 ~99% pixel 损失; 背景地板单调降 (0.0103→0.0035), 异常/正常分离比单调升 (8.8×→35×)。**L1 是实际最优** (pixel > baseline, image > baseline)。

---

## 实验 5: 筛选策略研究 (selection study)

**目的**: SeaS 生成大量候选，如何筛选真正值得进训练集的 BN/BD？对比 10 种筛选策略 (S0_random / S1_area / S2_m_only / S3_m_area / S4_m_score / S5_m_topk / S6_m_region / S7_m_multi / S8_m_soft / S9_m_pareto)，每策略选 100 BN + 100 BD，在原始 test 上评估。方案见 `docs/SELECTION_STUDY.md`。

**Step 1 — 构建候选池** (1200 候选 = 20 refs × 6 batches × 10 variants):
```bash
python src/selection/build_candidate_pool.py \
    --high_m_json outputs/features_visa/pipe_fryum/missing_boundary/high_m_regions.json \
    --oracle_dir outputs/features_visa/pipe_fryum/missing_boundary/mx_oracle \
    --output_dir outputs/selection_pool
```

**Step 2 — 筛选 + 组装 fix2 数据**:
```bash
python src/selection/run_study.py --phase select
```

**Step 3 — 训练** (监督器按 GPU 容量 {0:1,1:2,2:1,6:2,7:2} 排队 10 个模型, 每模型 ~10.7GB VRAM):
```bash
python src/selection/supervise_train.py
```
或手动 `python src/selection/run_study.py --phase train`。

**Step 4 — 评估对比**:
```bash
python src/selection/run_study.py --phase eval
```

**关键发现**: 池内 `M` (0.5·gap+0.5·PB) 与 BD 候选 mask 面积 **反相关 (corr≈−0.763)** —— 按 M 排序的 selector (S2/S3/S9) 会选到小且弱/近乎无缺陷的样本 (66% 小缺陷)，按面积/证据的 S1 选到大缺陷。结果统计见 `src/selection/results/screening_stats.json`。

---

## 结果摘要

| 实验 | 数据集 | 关键数字 |
|------|--------|----------|
| v3 闭环 (bottle) | MVTec | 20 候选 → 16 接受, 接受集 M=1.30 ≫ 阈值 |
| v3 闭环 (pipe_fryum) | VisA | 750 候选 → 480 接受, 接受集 M=1.26 |
| fix2 (bottle) | MVTec | AP Pixel 0.8386→**0.8415**, 四指标不劣于基线 |
| fix2 (pipe_fryum) | VisA | Image 全升 (AUC 0.934→0.991), AUC Pixel 0.698→0.491 |
| BN 比例消融 | VisA | 5% BN → pixel 崩 (AUC Pixel 0.491→0.321) |
| 色差消融 L1 | VisA | pixel 0.491→**0.738** (> baseline), image 0.991→0.959 |

详细分节 (含每次实验的设计/坑位/判定) 见 `docs/SUMMARY.md`。

---

## 常用坑位备忘

1. **DRAEM SSIM 隐式 `.cuda()`** → 训练脚本须 `torch.cuda.set_device(device)`, 否则落到被占满的 cuda:0 直接 OOM。
2. **test_DRAEM.py `map_location='cuda:0'` 写死** → 评估用 `CUDA_VISIBLE_DEVICES=<空闲卡>` 映射。
3. **VisA GT mask 软值** → 评估前须 `(gt>0.01)` 二值化, 否则 `astype(np.uint8)` 全截成 0。
4. **SeaS 配置覆盖 CLI** → 控 noise_step/gen_mask 必须用临时 config (`make_temp_config`)。
5. **长任务脱离会话** → `setsid nohup <cmd> > log 2>&1 &`, 避免会话 teardown SIGKILL。
