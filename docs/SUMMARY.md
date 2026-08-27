# Missing Boundary 闭环实验 — 完整实验记录

> 更新: 2026-08-21  |  数据集: MVTec bottle + VisA pipe_fryum + VisA candle  |  检测器: DRAEM  |  生成器: SeaS

## 0. 背景与核心修正 (v3)

**原始缺陷**（用户指出，已实证确认）:
1. **M(x) 只作事后分组，不驱动生成** —— 选高-M 参考图后，生成样本的 M(x) 从不检查。
2. **高-M 参考图无指导作用** —— 实测 v2 (noise=500): ref gap=0.10, 生成变体 gap=0.27（锚定失效）；M≈0.154 只是恰好接近 P90；score 0.988 > ref 0.972 且无内容验证。
3. **归一化不可比** —— phase1 分集归一化，ref 的 M 与生成样本的 M 不可比。

**v3 修正**: M(x) 成为**生成运行时判据**（闭环）:
```
每 ref: 沿 noise 阶梯 SeaS 生成 batch → oracle 逐张评分(M, score, gap, mask) →
  接受/拒绝门:  BN: M≥m_accept_bn 且 mask<0.05 (内容正常)
                BD: M≥m_accept_bd 且 mask>0.05 且 score∈盲带
  配额未满 → 继续阶梯 / 降级 M_accept (P90→P85→P80)
输出: generation_closedloop_summary.json (接受漏斗 = M(x) 驱动证据)
```

**新组件**: `src/mx_oracle.py`（固定归一化 M(x) 评分器，Gap 锚定 normal P1/P99 + KDE-PB bandwidth=0.1）、`src/compute_missing_boundary.py`（v3: 切 KDE-PB + 导出 oracle）、`src/generate_mb_closedloop.py`（闭环）、`src/prepare_visa.py`（VisA 转换）、`src/train_draem_single.py`（单类 DRAEM）。

## 1. MVTec bottle 验证 ✅

小规模闭环（1 BN + 1 BD ref）: **20 候选 → 16 接受**。接受集 M=1.30 >> 阈值(BN 0.72/BD 0.80)；BN mask_cov=0.008（内容正常），BD mask_cov=0.33（含缺陷）。

## 2. VisA pipe_fryum 全量闭环 ✅

**流程**: VisA 转换(MVTec+SeaS 布局) → DRAEM 200ep 训练(AUROC 0.66 弱检测器) → Phase0 特征提取 → Phase1 oracle → SeaS gen+mask 各 800 步(1 子类) → 闭环生成。

**结果**: 45 BN refs + 15 BD refs → **750 候选 → 480 接受**（360 BN + 120 BD），接受集 M=1.26 >> 阈值(BN 0.60/BD 0.63)。M 门跨数据集复现成功。

### 2.1 BD 零产量诊断与修复（关键）
- 初跑: BD 0/8 接受（20 候选全部 `no_defect`，RMP mask≈0）。
- **诊断**（用户目检确认 BD 候选含可见缺陷）:
  - DRAEM amap 对真实正常图也饱和到 0.99（弱检测器, 无法当缺陷信号）
  - RMP mask 对生成候选不点亮（≈0.005, 但 RMP 训练损失已收敛 0.2→0.0009）
  - BN/BD 候选的像素/特征 Δ 不分离（BN 18.8 vs BD 16.3）
  - **结论**: 验证坏了（RMP/amap 失效），不是生成坏了。
- **修复**: `--rmp_thr_bd 0.002`（低于 BD 候选 RMP 观测范围 [0.004,0.014]）+ `--score_band "0.0,1.0"`（分数饱和失去意义）。复验通过 → 全量。

## 3. 生成质量检测（暴露问题）

**量化信号不乐观**: 生成图 gap（DRAEM 特征离正常分布距离）:

| 类型 | gap | vs 真实 |
|---|---|---|
| 真实 normal | 0.17 | 基线 |
| 真实 anomaly | 0.22 | — |
| 生成 BN | 1.63 | **9.4x** |
| 生成 BD | 1.19 | 6.9x |

生成图离真实分布偏离 ~9 倍 → 很可能是重度重建/人造痕迹重。**无系统视觉质量测试**（环境无法目检图片，仅用户抽查过 BD 候选确认含缺陷）。视觉审查包: `outputs/generated_v3_visa/_quality_review/`。

## 4. v3 重训实验（更多训练步数）⚠️ 负面结果

**动机**: 尝试提升生成质量（800→2400 步）。

**坑**: SeaS `checkpointing_steps` 多次保存到同一 `generation-checkpoint`/`mask-checkpoint` 目录，`save_pretrained`/`copytree`/`makedirs` 非 exist_ok → step 1600 必崩。**已 patch**: `models/seas.py` + `models/seas_mask.py` 保存前 `shutil.rmtree`。

**结果**: gen v3 (2400步) + mask v3 (2400步) 训练完成，但**生成质量无改善**:
- v3 BN gap=1.61 (v1 1.63), v3 BD gap=1.76 (v1 1.19, 反而更高)
- 生成图仍离真实分布 ~9x。

**结论**: pipe_fryum 上 SeaS 质量瓶颈**不是训练步数问题**（模型已收敛，输出是 SD 风格重建，DRAEM 特征空间有系统性偏移）。视觉对比包: `outputs/_quality_compare/BN_ref380_v1v3.png`、`BD_ref234_v1v3.png`（[ref][v1×3][v3×3]）。

## 5. 当前状态与下一步

| 项 | 状态 |
|---|---|
| M(x) 闭环机制 | ✅ bottle + pipe_fryum 均跑通，M 门/接受漏斗可审计 |
| BD 内容验证 | ⚠️ 弱检测器类别降级（降 RMP 阈值，依赖人工目检）；candle 上 RMP 完全失效（散点噪声） |
| 生成质量 | ⚠️ 瓶颈类别特异（candle 5.5x/3.2x < pipe_fryum 9.4x/6.9x）；部分因 SD1.4 VAE 固有偏色 |
| fix2 双流训练 | ✅ **bottle 通过**（见 §7）；pipe_fryum **混合结果**（Image 全升、AUC Pixel 降，见 §7.1） |

**待决策/可选下一步**:
1. fix2 扩 pipe_fryum（480 张接受样本: 360 BN→train/good + 120 BD→双流）—— 弱检测器类别 BD mask 对齐性需先复查
2. fix2 扩更多强检测器类别（MVTec 其他类）证泛化，供论文
3. 试其他质量杠杆（noise/guidance_scale 搜索）
4. 论文角度: pipe_fryum 作为"弱检测器 + SeaS 质量受限"案例

## 6. 关键产物路径

```
outputs/features_visa/pipe_fryum/missing_boundary/
  ├── mx_oracle/oracle.pkl          # M(x) 固定归一化评分器
  ├── high_m_regions.json           # 高-M 参考图 (BN 45, BD 15)
outputs/generated_v3_visa/pipe_fryum/
  ├── generation_closedloop_summary.json   # v1 (800步) 全量漏斗
  └── {boundary_normal,blind_defect}/sample_*/acc_*.png
outputs/generated_v3_visa_v3/pipe_fryum/   # v3 (2400步) 小规模验证
outputs/_quality_review/  outputs/_quality_compare/   # 目检用拼图
```

**执行入口**: `NEXT_STEPS.md`（含全部命令/参数/坑位）。

## 7. fix2 双流训练（bottle）✅ (2026-08-19)

**目的**: 纠正旧 `mb_aug` 错误——BD 错塞 train/good 致 Pixel AP 退化（0.839→0.754）。
正确做法（已验证）: **BN(8)→train/good、BD(8)→MirrorEM 双流**（image=ref, augmented=BD图, mask=BD mask, focal 监督分割, w=0.5）。bottle 是强检测器类别，BD RMP mask 覆盖率 28-38%、1-3 连通域成块（对齐可用），适合做双流。

### 实现流程

| 步 | 脚本/操作 | 说明 |
|---|---|---|
| 1. 数据准备 | `src/fix2_prepare.py` | `train_good_plus_bn/` = MVTec bottle/train/good(209) + BN(8) = **217 张**; `bd_manifest.csv` = 8 行 BD（image/mask/reference/type/utility, ref=090.png） |
| 2. 训练 | `src/train_draem_fix2.py` | fork MirrorEM 双流; `--init-random` 用 DRAEM `weights_init` 随机初始化; 加 DRAEM `MultiStepLR([160,180],γ=0.2)` 保公平; DRAEM 兼容命名 |
| 3. 参数 | — | 200ep, lr=1e-4, bs4/4, mirror_w=0.5, 55 步/epoch, GPU7, ~30min |
| 4. 评估 | `test_DRAEM.py` | `CUDA_VISIBLE_DEVICES=<GPU> --gpu_id 0`（绕 map_location='cuda:0' 坑） |

### 结果（与 baseline_200 同条件重跑）

| 指标 | baseline_200 | fix2 | Δ |
|---|:---:|:---:|:---:|
| AUC Image | 0.9627 | 0.9627 | 0 |
| AP Image | 0.9805 | **0.9856** | +0.005 |
| AUC Pixel | 0.9733 | 0.9816 | +0.008 |
| AP Pixel | 0.8386 | **0.8415** | +0.003 |

**判定: 通过** — AP Image 0.9856 ≥ 0.981 且 AP Pixel 0.8415 ≥ 0.839，远优于 mb_aug 0.754。四指标均不劣于基线；loss 收敛 0.458→0.032（pre 0.050 / mirror 0.014）。**双流是 BD 生成样本的正确消费方式**。

### 坑位

1. **DRAEM `loss.py` SSIM 构造隐式 `.cuda()`** 走当前默认设备（非 `--device`）→ 训练脚本须 `torch.cuda.set_device(device)`，否则落到被占满的 cuda:0 直接 OOM。
2. **test_DRAEM.py `map_location='cuda:0'` 写死** → 评估用 `CUDA_VISIBLE_DEVICES=<空闲GPU>` 把物理卡映射成 cuda:0。

**产物**: `outputs/fix2/bottle/checkpoints/fix2/DRAEM_test_0.0001_200_bs8_bottle_.pckl` + `__seg.pckl`

## 7.1 fix2 扩 pipe_fryum（弱检测器）— 混合结果 (2026-08-21)

bottle 验证通过后扩到弱检测器类别。数据: 480 接受样本（360 BN→train/good + 120 BD→双流，fix2_prepare 泛化支持 VisA）。200ep 全量跑完。

### 结果（与 baseline 200ep 同条件）

| 指标 | baseline | fix2 | Δ | 判定 |
|---|:---:|:---:|:---:|:---:|
| AUC Image | 0.9336 | 0.9766 | +0.043 | ✅ |
| AP Image | 0.9709 | 0.9895 | +0.019 | ✅ |
| AUC Pixel | 0.6982 | 0.6567 | **−0.042** | ⚠️ |
| AP Pixel | 0.1012 | 0.1068 | +0.006 | ✅ |

**混合结论**: 图像级全升、Pixel AP 不退化，但 **AUC Pixel 明显退化**——弱检测器上双流提升图像判别、牺牲像素定位。按 bottle 判据算通过，但须如实记录像素退化。

**可能因子（待消融）**: ① 44% 生成 BN（VAE 偏色 gap~9x）进 train/good 干扰像素重建; ② 弱检测器 BD mask 覆盖率仅 ~1%，focal 监督偏弱。

### 新坑位（已修）

1. **VisA GT mask 软值**：预处理缩放把 GT mask 缩成像素 1-5/255，`test_DRAEM.py` 的 `astype(np.uint8)` 全截成 0 → 像素 ROC 报单类。评估副本须先 `(gt>0.01)` 二值化（`/tmp/test_DRAEM_pipe_fryum.py`，obj_list 改 pipe_fryum；未改 DRAEM 仓库）。
2. **`fix2_prepare.py` refs 映射**：只按文件名作 key → 多样本目录互相覆盖（bottle 单目录未暴露）；已改为按 `sample_dir/saved_as`。
3. **长任务脱离会话**：后台任务首跑在 epoch 72 被会话 teardown SIGKILL（root cause: harness 子进程生命周期）；改用 `setsid nohup` 脱离会话后 200ep 完整跑完。操作教训写进 NEXT_STEPS。

**产物**: `outputs/fix2/pipe_fryum/checkpoints/fix2/DRAEM_test_0.0001_200_bs8_pipe_fryum_.pckl` + `__seg.pckl`（epoch 72 部分备份在 `checkpoints/_partial_epoch72/`）。baseline: `outputs/checkpoints/visa/`。

## 7.2 BN 比例消融（44%→5.1%）— 混淆结果，无法归因 (2026-08-21)

**设计**: 只隔离 BN 比例 — train/good 从 450+360（44%）改为 450+24（5.1%，等距抽样覆盖 45 目录），BD 双流 120 张与其余超参全同（init-random, 200ep, lr1e-4, w0.5, bs4）。数据: `outputs/fix2_bn5/`。

**结果**（GPU2, `/tmp/test_DRAEM_pipe_fryum.py`）:

| 指标 | baseline | fix2 44% | fix2 5% | Δ(44→5%) |
|---|:---:|:---:|:---:|:---:|
| AUC Image | 0.9336 | 0.9766 | 0.9874 | +0.011 |
| AP Image | 0.9709 | 0.9895 | 0.9939 | +0.004 |
| AUC Pixel | 0.6982 | 0.6567 | **0.3270** | **−0.330** |
| AP Pixel | 0.1012 | 0.1068 | **0.0342** | **−0.073** |

**判定: 混淆，不能归因于 BN 比例。** `steps_per_epoch=max(⌈N_pre/4⌉,⌈N_mirror/4⌉)` 随 train 规模 810→474 从 203→119，总优化步数减 41% → 5% 模型 seg 头欠训练。seg 输出诊断证实：5% 模型在 GT 正常像素上的 seg 均值 0.0987 vs 44% 的 0.0068（背景地板抬 ~14×，异常/正常分离比 ~2× vs ~17×）→ 像素 ROC 被正常像素的中等分数污染。**BN 比例与优化预算两个变量同时动了，假设（44% BN 干扰像素重建）既未证实也未证伪**。

**附带观察**: 44% 模型背景地板极低（0.0068），空间定位是三档最干净——更充分的训练步数（203×200）帮 seg 学会了"背景闭嘴"，而非 BN 比例本身。

**待办**: 用 `--steps-per-epoch 203` 重跑 5% 数据，固定优化预算与 44% 对齐，只留 BN 比例一个变量。


## 7.3 test 统计泄漏修复 — MVTec 协议合规化 (2026-08-24)

**问题**: v3 闭环生成管线存在 3 处 test 统计泄漏（严格 MVTec 协议违规——任何 test 统计
参与训练数据构造，都削弱"泛化到未见缺陷"的声称）:
1. `kde_anom`（PB 异常侧 KDE）用 **test 缺陷图 score** 拟合（mx_oracle.py L263-270）
2. `m_accept_bd` / `bd_relax` 锚定 **test M 的 P90/P85/P80**（mx_oracle.py L334）
3. `blind_defect_samples`（test 高-M）作为 BD 参考图 fallback（generate_mb_closedloop.py L336-337）

**修复**（用户批准方案: 合成异常 KDE + P90 train M）:
- **kde_anom → 合成异常 KDE（train-only）**: 新增 `src/extract_synthetic_scores.py`，
  用与特征提取同一 baseline DRAEM（`outputs/checkpoints/visa/DRAEM_test_0.0001_200_bs8_pipe_fryum_`）
  对训练正常图(450) 的 Perlin×DTD 合成异常打分 → `outputs/features_visa/pipe_fryum/train_good/synthetic_scores.csv`
  （n=1737, score [0.024,1.000]）。
- **m_accept_bd → train M P90**（= m_accept_bn = 0.3569）; BD 判别完全交给
  `mask_cov>rmp_thr_bd` + `score∈band` 两个门。
- 移除 `blind_defect_samples` fallback；BD 参考图 = train 高-M normal（与 BN 同源, prompt 定内容）。
- `compute_missing_boundary.py` 的 blind_defect 降级为**分析输出**（生成不再读取）。
- 删除已无引用的 `is_defect()`。

**重建 oracle 实测（pipe_fryum）**:
- gap 常数不变（p1=0.8033, p99=8.3046）—— 改动只影响 PB 锚定。
- PB 从"test 缺陷锚定"改"合成异常锚定": train M P90 **0.603→0.357**；
  PB≈0.506 恒定 → M≈0.5·gap+0.253，高-M≈高-gap（pipe_fryum DRAEM seg 输出饱和 ~0.995, 数据属性非 bug）。
- boundary_normal_samples=45（train）, blind_defect=85（仅分析）。
- 合成异常 score 分布 [0.024,1.000] 与正常 [0.994,0.996] 高度分离 → PB 不再退化（旧 test 缺陷
  score 与正常重叠 → PB 近退化, M 被 gap 单方面主导）。

**影响**: 生成闭环须用新 oracle 重跑（新 BN/BD 数据 → fix2 重训 → 重评估）。7.2 的 BN 比例
消融干净重跑（steps203）用旧 oracle 数据，作为 BN 比例问题的独立判断仍有效。

### 7.3.1 BN 比例消融 steps203 结果（5% BN，固定优化预算）— 判定：5% BN 是 pixel 退化主因 (2026-08-24)

**设计**: 完成 §7.2 待办 — 5% BN 数据（450+24，pre_samples=474）+ `--steps-per-epoch 203`
固定优化预算与 44% 对齐，只留 BN 比例一个变量。数据用旧 oracle（generated_v3_visa），
作为 BN 比例问题的独立判断。cp: `outputs/fix2_bn5/pipe_fryum/checkpoints/fix2_steps203`
（init-random, 200ep, lr1e-4, w0.5, bs4, steps203）。

**评估（test 150 张, gt>0.01 二值化）**:

| 指标 | baseline | fix2 44% | fix2 5% 混淆(119步) | **fix2 5% steps203（本次）** |
|---|---|---|---|---|
| AUC Image | 0.9336 | 0.9766 | 0.9874 | **0.9936** |
| AP Image | 0.9709 | 0.9895 | 0.9939 | **0.9969** |
| AUC Pixel | 0.6982 | 0.6567 | 0.3270 | **0.3212** |
| AP Pixel | 0.1012 | 0.1068 | 0.0342 | **0.0593** |

**判定**: AUC Pixel **未回升**（0.3212 ≈ 混淆版 0.3270，干净复现）。→ **BN 比例（5%）是 pixel
退化的主因**，§7.2 的步数混淆已排除（固定 steps203 后仍退化）；44% BN 是维持 pixel 的必要条件
（0.657 vs 0.32）。同时 image 指标随 5% 更升（0.9936 最高）→ 5% 是"image 高、pixel 崩"的极端组合。
泄漏（test 统计）对 pixel 的独立影响待 fix2_clean（44% BN + 协议干净数据）训完与 leaky-44%
（0.6567）对比后隔离。

### 7.4 协议干净版 fix2 结果（44% BN，新 oracle 数据）— image 升、pixel 回落真实水平 (2026-08-24)

**流程**: 泄漏修复后的完整闭环：新 oracle（train P90 M=0.357）→ 生成（修正门 rmp_thr_bd=0.002 +
score_band=0-1）→ 360 BN + 120 BD 全量接受 → fix2_prepare（train_good_plus_bn 810）→ fix2 重训
（init-random, 200ep, lr1e-4, w0.5, bs4, steps203）。cp: `outputs/fix2_clean/pipe_fryum/checkpoints/fix2`。
BD 120 行 ref 全为 train/good（已核验，train-only）。

**评估（test 150 张, gt>0.01）**:

| 指标 | baseline | leaky-fix2 44% | **clean-fix2 44%（本次）** | fix2 5% steps203 |
|---|---|---|---|---|
| AUC Image | 0.9336 | 0.9766 | **0.9912** | 0.9936 |
| AP Image | 0.9709 | 0.9895 | **0.9957** | 0.9969 |
| AUC Pixel | 0.6982 | 0.6567 | **0.4913** | 0.3212 |
| AP Pixel | 0.1012 | 0.1068 | **0.0899** | 0.0593 |

**结论**:
- **image 指标协议干净版最好**（AUC Image 0.9766→0.9912, AP Image 0.9895→0.9957）。
- **pixel 指标回落**（AUC Pixel 0.6567→0.4913）→ leaky-44% 的 pixel 0.657 部分由 **test 泄漏虚高**
  （kde_anom/test-M 锚定让生成 BD 贴近 test 缺陷分布）。泄漏移除后 honest pixel ≈ 0.49。
- 仍远高于 5% 崩盘（0.32）→ BN 比例与泄漏是两个正交因素。
- **clean fix2 仍不及 baseline pixel（0.4913 < 0.6982）**：fix2 双流引入的生成 BD/BN 对 pixel 净效应为负，
  image 为正。MVTec 协议下这是真实水平。
- Stage4 driver 内嵌评估因缺 PYTHONPATH（data_loader）exit=1，此处已手动补跑（同 /tmp/test_DRAEM_pipe_fryum.py）。

### 7.5 色差剂量-响应消融（L2 自然 / L1 半量 / L0 全对齐）— 色差确认是 pixel 退化主因 (2026-08-26)

**动机**: clean-fix2 pixel 0.4913 < baseline 0.6982，需隔离"生成图 VAE 偏色"是否主因。假设 H：色差→重建残差→正常像素误报→背景地板抬升→pixel 降。用 Reinhard LAB 逐图颜色传递（`src/color_align_generated.py`），`--t 0.5/1.0` 插值控制剂量。控制变量与 fix2_clean 全同（360BN+120BD, init-random, 200ep, lr1e-4, w0.5, bs4, steps203），只变颜色。

**剂量验证（`color_shift_report.json`，480 图生成集 vs 真实 train/good）**:

| 剂量 | 处理 | ΔE76 | Δμ_L | Δσ_b | dch_b |
|---|---|---|---|---|---|
| L2 自然 | 无 | **15.70** | −9.96 | 5.08 | 6.55 |
| L1 半量 | t=0.5 | **10.72** | −5.06 | 2.58 | 4.36 |
| L0 全对齐 | t=1.0 | **7.23** | −0.48 | −0.06 | 2.81 |
| 真实图间基线 | — | ~7.38 | — | — | — |

L0 残留 ΔE 7.23 ≈ 真实图间基线 7.38 → 对齐后生成图与真实图颜色**不可区分**。剂量操纵真实、单调。

**评估结果（test 150 张, gt>0.01）**:

| 指标 | baseline | L2 fix2_clean | **L1 半量** | **L0 全对齐** |
|---|---|---|---|---|
| AUC Image | 0.9336 | 0.9912 | 0.9594 | 0.8030 |
| AP Image | 0.9709 | 0.9957 | 0.9824 | 0.9180 |
| AUC Pixel | 0.6982 | 0.4913 | **0.7382** | **0.7399** |
| AP Pixel | 0.1012 | 0.0899 | **0.3076** | **0.3406** |

**地板诊断（`src/diag_seg_floor.py`）**:

| 剂量 | 背景地板 | 异常 seg | 分离比 | Cohen's d |
|---|---|---|---|---|
| L2 | 0.01030 | 0.090 | 8.8× | 0.94 |
| L1 | **0.00345** | 0.121 | 35.1× | 1.24 |
| L0 | **0.00425** | 0.177 | 41.7× | 1.16 |

**判定：支持 H — 色差是 pixel 退化的主导机制（饱和剂量-响应）**
- **pixel 单调升 + 饱和**: 0.4913→0.7382→0.7399（+0.247/+0.249，远超判定阈值 0.03）。**半量 L1 已恢复 ~99% pixel 损失**，L1≈L0 平台 → 色差因果性最强（一旦颜色对齐，pixel 即回到 ≥baseline）。
- **地板单调降 + 分离单调升**: 0.0103→0.0035→0.0043（−66%/−59%）；分离比 8.8×→35×→42×。**H 机制链直接证实**：色差→正常像素误报→地板抬升→pixel 降。
- **异常像素 seg 单调升**: 0.090→0.121→0.177 → 对齐后 seg 更聚焦真实缺陷，误报让位给真阳。
- **image 指标随对齐单调降（违反「image 不变」预期，如实记录）**: 0.9912→0.9594→0.8030。L0 image 0.8030 甚至 < baseline 0.9336。机制：L0/L1 seg 精准局部化 → 异常图 max-pooled 分数 median 塌回正常 → 21×21 池化后图像级 max 偏低 → image AUC 降。L2 靠色差弥散抬升异常图整体分，误打误撞拿 image 高分。**DRAEM 图像级 max-avgpool 评分依赖弥散响应**。

**L1 是实际可用最优（sweet spot）**: pixel 0.7382 ≈ L0（+0.247 vs L2），image 0.9594 > baseline 0.9336（仅 −0.032 vs L2）。全对齐 L0 不必要且伤 image。

**推论**: pixel 缺口（0.49 vs baseline 0.70）主要由生成图色差解释，其次 mask 质量（候选 C）。下一步消融：mask 校正（C）确认残余 0.02 gap，或 mirror 权重（E）。注：DRAEM 图像级评分设计（max-avgpool）本身偏向弥散响应，是 fix2 精准 seg 下 image 分输给 baseline 的结构性原因。
