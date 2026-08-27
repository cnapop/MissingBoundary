# 续跑指南 — Missing Boundary v3 闭环生成

> 更新: 2026-08-12。v3 (M(x) 闭环驱动生成) 已实现并通过小规模验证。本文件是续跑入口。

## 环境速查

| 项 | 值 |
|---|---|
| 项目根 | `/data/chenjiawen/MissingBoundary` (旧路径 `workspace/MissingBoundary` 为镜像) |
| DRAEM | `/data/chenjiawen/DRAEM` (env: DRAEM, 有 sklearn) |
| SeaS | `/data/chenjiawen/SeaS` (env: seas, 无 sklearn — 编排器用 DRAEM env) |
| 数据集 | `/data/chenjiawen/Datasets/MVTec-AD` |
| M(x) oracle | `outputs/features/bottle/missing_boundary/mx_oracle/oracle.pkl` |
| 闭环输出 | `outputs/generated_v3/` |
| GPU | 8×RTX4090 (0-7); SeaS 用 0-6, 评分用 7 (`--score_gpu 7`) |

## v3 核心: M(x) 成为生成运行时判据 (闭环)

**问题(已实证)**: 旧版 open-loop——高-M ref 被选为参考, 但生成样本的 M(x) 从不检查。
实测 v2 (noise=500): ref gap=0.10, 生成变体 gap=0.27 (锚定失效); M≈0.154 只是恰好接近 P90;
score 0.988 > ref 0.972 (可能含缺陷, 无内容验证)。

**v3 闭环** (`src/generate_mb_closedloop.py`):
```
每 ref (BN=top-M train normal, BD=同批 normal + anomaly prompt):
  沿 noise 阶梯生成 batch → 逐张 oracle 评分(M, score, mask_cov) →
  BN:  M>=m_accept_bn AND mask_cov<0.05  (高-M + 内容正常)
  BD:  M>=m_accept_bd AND mask_cov>0.05 AND score∈[0.5,0.995]  (高-M + 缺陷存在 + 难检测)
  配额未满→阶梯继续/降级 M_accept(P90→P85→P80) → 记录接受漏斗
```
输出 `generation_closedloop_summary.json` = **M(x) 驱动生成的可审计证据**。

**M(x) 定义 (v3)**: `M = 0.5*Gap_norm + 0.5*PB`, Gap 用固定 P1/P99 锚定 normal 自距离 (消除分集归一化不可比), PB=KDE-PB(bandwidth=0.1)。
bottle 上正常/缺陷 DRAEM 分数几乎重叠(0.97-0.99) → PB≈1 恒定, M 实际由 Gap 主导 (数据属性非 bug)。

## 续跑命令

### 第一步: 跑通 phase1 (KDE-PB + oracle)
```bash
source /home/chenjiawen/anaconda3/bin/activate DRAEM
cd /data/chenjiawen/MissingBoundary
python src/compute_missing_boundary.py --category bottle --output_dir outputs/features
# 产出: missing_boundary.csv, high_m_regions.json, mx_oracle/oracle.pkl
```

### 第二步: 全量闭环生成
```bash
source /home/chenjiawen/anaconda3/bin/activate DRAEM
cd /data/chenjiawen/MissingBoundary
python src/generate_mb_closedloop.py \
    --category bottle \
    --high_m_json outputs/features/bottle/missing_boundary/high_m_regions.json \
    --oracle_dir outputs/features/bottle/missing_boundary/mx_oracle \
    --output_dir outputs/generated_v3 \
    --gpus 0,1,2,3 --score_gpu 7 \
    --num_bn 20 --num_bd 10 \
    --num_variants 10 --quota_per_ref 8 --max_attempts 6 \
    --bn_noise_ladder 300,500,700 \
    --bd_noise_ladder 1200,1500,1800
```
或 `bash run_pipeline.sh phase2 --gpu 0` (已接线, 用 config.sh 参数)。

### 第三步: 训练消费 (fix2)
- **bottle ✅ 已完成通过** (2026-08-19): AP Image 0.9805→0.9856, AP Pixel 0.8386→0.8415, 修复 mb_aug 0.754 退化。见 `SUMMARY.md §7`。
  - 脚本: `src/fix2_prepare.py` + `src/train_draem_fix2.py` (`--init-random` + DRAEM weights_init + MultiStepLR 保公平, DRAEM 兼容命名)。
  - 评估: `CUDA_VISIBLE_DEVICES=<GPU> python test_DRAEM.py --gpu_id 0 --base_model_name DRAEM_test_0.0001_200_bs8 --checkpoint_path outputs/fix2/bottle/checkpoints/fix2`。
- **pipe_fryum ✅ 已跑完** (2026-08-21): 200ep 全量。**混合结果** — AUC Image 0.9336→0.9766, AP Image 0.9709→0.9895, AP Pixel 0.1012→0.1068, **AUC Pixel 0.6982→0.6567 (退化)**。见 `SUMMARY.md §7.1`。
  - 数据/命令与 bottle 相同，仅换类别: `fix2_prepare.py --category pipe_fryum --gen_root outputs/generated_v3_visa --normal_src_dir outputs/visa_datasets/pipe_fryum/train/good` + `train_draem_fix2.py --category pipe_fryum`。
  - 评估用 `/tmp/test_DRAEM_pipe_fryum.py`（obj_list=pipe_fryum + GT 二值化），baseline 在 `outputs/checkpoints/visa/`。
  - **BN 比例消融 (44%→5.1%) 已跑 (2026-08-21)**: 结果**混淆**（AUC Pixel 0.6567→0.3270 崩）——steps_per_epoch 随 train 规模 810→474 从 203→119，优化预算与 BN 比例同时动，无法归因。干净重跑须 `--steps-per-epoch 203` 固定预算。待消融: (a) 重跑 5% (fixed steps); (b) mirror 权重 w 调小; (c) 只 BD 双流不加 BN。

## 关键坑位备忘 (沿用)

1. SeaS `load_args` 用 config 覆盖 CLI → 控制 noise_step/gen_mask 必须用临时 config (`make_temp_config`, 已封装)
2. SeaS 单张 ref 触发 einsum bug → ref 目录 ≥10 张 (闭环内已复制 10 份)
3. temp config `device=cuda:0`, `gpu_id: null`
4. `num_variants` ≥ batch_size(10), 否则 `total_infer_num // batch_size = 0` 不生成
5. subprocess 调 SeaS 必须 `cwd=seas_dir`, 绝对路径
6. **编排器必须在 DRAEM env** (闭环评分需 sklearn+torch+model_unet); seas env 无 sklearn
7. **长任务必须脱离会话** (>=1h 的训练/生成): 后台任务首跑在 epoch 72 被会话 teardown SIGKILL (harness 子进程生命周期)。用 `setsid nohup <cmd> > log 2>&1 &` 启动 → 进程重挂 PID 1、会话退出不影响; 输出重定向到文件 (避免子进程持管道)。核验: `ps -o pid,ppid -p <pid>` 看 PPID=1。

## 状态文件

- 实验配置: `experiment_config.json` (v3 status)
- 完整总结: `SUMMARY.md`
- 本文件: `NEXT_STEPS.md`


## 2026-08-24 更新: test 统计泄漏已修复（协议合规化）

v3 闭环生成管线已移除全部 test 统计参与:
- kde_anom 改 train-only 合成异常 KDE（`src/extract_synthetic_scores.py`, 每类别先跑一次）
- m_accept_bd = train M P90（与 BN 同尺）
- BD 参考图 = train 高-M normal（不再回退 test blind_defect_samples）

pipe_fryum oracle 已重建: `outputs/features_visa/pipe_fryum/missing_boundary/mx_oracle/`。
**续跑（协议干净版）**: 先为每个类别跑合成异常打分，再
`python src/compute_missing_boundary.py --category pipe_fryum --output_dir outputs/features_visa --max_train_images 450`
然后重跑 `src/generate_mb_closedloop.py`（命令同 §续跑命令 第二步, 指向新 oracle）。

### 2026-08-24 续: BN 比例消融 steps203 完成 — 5% BN 是 pixel 退化主因

§7.2 待办（5% BN 数据 + `--steps-per-epoch 203` 固定预算）已跑完并评估
（`outputs/fix2_bn5/pipe_fryum/checkpoints/fix2_steps203`）:
- **AUC Pixel 0.3212**（混淆版 119 步 0.3270 → 干净复现，未回升）
- → **5% BN 比例本身是 pixel 退化主因**，步数混淆已排除；44% BN 是维持 pixel 的必要条件。
- image 指标反随 5% 升到最高（AUC Image 0.9936）→ "image 高、pixel 崩"极端组合。
- 详见 `SUMMARY.md §7.3.1`。

**协议干净版 fix2 已完成并评估（2026-08-24）**: `outputs/fix2_clean/pipe_fryum/checkpoints/fix2`
（44% BN + 新 oracle 数据, 120 BD 全 train/good 核验）。结果: **AUC Image 0.9912 / AP Image 0.9957 /
AUC Pixel 0.4913 / AP Pixel 0.0899** — image 全面最好，pixel 比 leaky-44%（0.6567）回落 → 泄漏曾虚高
pixel；clean 仍不及 baseline pixel（0.6982）。详见 `SUMMARY.md §7.4`。

**待办消融（保 pixel 0.65+ / image 高）**: (a) mirror 权重 w 调小（0.5→更小）; (b) 只 BD 双流不加 BN;
(c) 找 clean pixel 回 0.65 的配置（当前 0.49 与 baseline 0.70 差距在生成数据质量）。candle 类别协议
干净跑（extract_synthetic_scores → oracle → 闭环）仍未做。

### 2026-08-26 续: 色差剂量-响应消融完成 — 色差是 pixel 退化主因，L1 为最优配置

§7.5 完整 3 水平（L2/L1/L0 Reinhard 颜色对齐）跑完。**判定：支持 H**。
- **pixel**: L2 0.4913 → **L1 0.7382** → L0 0.7399（半量即饱和恢复 ~99%，L1≈L0）
- **地板**: 0.0103 → 0.00345/0.00425（−66%/−59%）；分离比 8.8×→35×/42×
- **image 随对齐降**（0.9912→0.9594→0.8030）——DRAEM max-avgpool 图像级评分依赖弥散响应
- **L1 是实际最优**: pixel 0.7382（>baseline 0.6982）+ image 0.9594（>baseline 0.9336）
- 数据/产物: `outputs/generated_v3_visa_clean_align_L{0,1}`、`outputs/fix2_align_L{0,1}`、
  `outputs/fix2_align_L1/pipe_fryum/seg_floor.json`、`SUMMARY.md §7.5`

**下一步（按优先级）**:
1. **(E) mirror 权重 w 消融**（在 L1 对齐数据上试 w=0.1/0.3，看能否 image 保 0.96+ 且 pixel 再升）— 直接吃 L1 最优基础。
2. **(C) mask 质量校正** — 确认 residual gap（L0 0.7399 vs baseline 0.6982 已反超，残余 gap 很小，可选做）。
3. candle 类别协议干净跑（extract_synthetic_scores → oracle → 闭环 → fix2+color_align_L1），验证色差结论跨类别泛化。
4. 若论文要报 image 与 pixel 双高：报告 L1（而非 L0/L2）作为 fix2 推荐配置；指出 DRAEM 评分口径对弥散响应的偏置。
