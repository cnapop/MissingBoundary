# 实验方案：生成图像色差对 fix2 像素级性能的影响

> 日期: 2026-08-25 · 类别: pipe_fryum · 关联: SUMMARY §7.3.1/§7.4

## 0. 背景与假设

已知事实:
- 生成 BN/BD 存在 VAE 偏色（SUMMARY 记 gap~9x），与真实 `train/good` 有色差。
- BN 比例消融（§7.3.1）确认 BN 数量影响背景地板（5%: 0.0987 vs 44%: 0.0068），但**色差本身**是否是"地板抬升/像素误报"的独立机制，尚未隔离。
- fix2_clean（44% BN, 自然色差）: AUC Pixel 0.4913。

**中心假设 H**: 生成图系统色差 → 重建残差出现在 `reconstruction_error`/`SSIM` 通道 → seg 在真实正常像素上误报 → 背景地板抬升 → pixel AUC 下降。对生成图做颜色校准后，pixel AUC 应上升、地板应下降，image 指标基本不变。

**零假设 H0**: 色差不是 pixel 下降的机制（移除色差后 pixel 不变）。

## 1. 变量设计

**自变量（treatment）: 生成图像的色差大小**，3 个水平（剂量-响应）:

| 水平 | 处理 | 预期残留色差 | 训练跑数 |
|---|---|---|---|
| L0 (aligned) | Reinhard 全量颜色传递到参考图 | ≈0 | 新跑 1 |
| L1 (half) | 校准 50%（线性插值） | 50% 自然值 | 新跑 1 |
| L2 (natural) | 现状自然色差 | 100% 自然值 | **已有 fix2_clean 结果，不重训** |

**控制变量**: 同一套生成数据全集（360 BN + 120 BD）、超参全同
（init-random, 200ep, lr1e-4, w0.5, bs4, steps203）、同一评估脚本、同一诊断脚本。**只变颜色变换**。

**因变量**:
- 主: AUC Image / AP Image / AUC Pixel / AP Pixel
- 诊断: GT 正常像素 seg 均值（背景地板）、异常/正常分数分离比
- 剂量: 生成集 vs 真实集 LAB 每通道 Δμ/Δσ、平均 ΔE76

## 2. 预处理实现 — 新增 `src/color_align_generated.py`

输入:
- `outputs/generated_v3_visa_clean/pipe_fryum/{boundary_normal,blind_defect}/sample_<ref>/acc_XX.png`
- 参考图: `outputs/visa_datasets/pipe_fryum/train/good/<ref>.png`（目录名推导；BD 另由 manifest 的 reference 列复核）
- `--level {0,1}`, `--out_root outputs/generated_v3_visa_clean_align_L<level>`

变换（Reinhard LAB 颜色传递, 逐张对齐到其参考图）:
1. 生成图与参考图各转 LAB
2. 逐通道 `(x - μ_img)/σ_img * σ_ref + μ_ref`
3. 转回 RGB
4. L1: 与原始图插值 `x' = (1-t)*aligned + t*orig`, t=0.5
5. mask 文件**原样复制**（颜色变换不改变 mask 坐标）

输出:
- 对齐后 BN/BD 图（目录结构原样复制）+ mask
- `color_shift_report.json`: 变换前后生成集 vs 真实集色差度量（每通道 Δμ/Δσ, mean ΔE76）——**证明剂量真实被操纵**

下游: 对 align 的 gen_root 跑 `src/fix2_prepare.py` → `outputs/fix2_align_L<level>/pipe_fryum/`（train_good_plus_bn + bd_manifest.csv）

## 3. 训练（L0/L1 并行, 各 ~8h）

```
python src/train_draem_fix2.py \
  --manifest outputs/fix2_align_L<level>/pipe_fryum/bd_manifest.csv \
  --normal-data-dir outputs/fix2_align_L<level>/pipe_fryum/train_good_plus_bn \
  --anomaly-source-path /data/chenjiawen/DRAEM/datasets/dtd/images \
  --init-random --base-name DRAEM_test_0.0001_200_bs8 --category pipe_fryum \
  --output-dir outputs/fix2_align_L<level>/pipe_fryum/checkpoints/fix2 \
  --epochs 200 --lr 0.0001 --mirror-weight 0.5 \
  --batch-size 4 --pre-batch-size 4 --draem-repo /data/chenjiawen/DRAEM \
  --device cuda:<GPU1>/<GPU2>   # 两卡并行
```
用 `setsid nohup` 脱离会话，log 到 `outputs/logs/align_L<level>.log`。

## 4. 评估

主指标（每水平训完即跑）:
```
CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=/data/chenjiawen/DRAEM python /tmp/test_DRAEM_pipe_fryum.py \
  --gpu_id 0 --base_model_name DRAEM_test_0.0001_200_bs8 \
  --data_path outputs/visa_datasets \
  --checkpoint_path outputs/fix2_align_L<level>/pipe_fryum/checkpoints/fix2
```

诊断（背景地板/分离比）: 写小脚本 `src/diag_seg_floor.py`——加载 seg ckpt，对 test 图出 seg map，按 GT mask 分正常/异常像素，输出: 正常像素 seg 均值（地板）、异常像素 seg 均值、分离比、以及各自直方图分布。

## 5. 判定准则

- **支持 H（色差是机制）**: 随色差减小（L2→L1→L0）AUC Pixel **单调上升**（或 L0 显著 > L2, Δ>0.03），背景地板单调下降；image 指标基本不变（|Δ|<0.01）。剂量-响应单调性 = 因果性最强证据。
- **支持 H0（色差不是机制）**: L0 ≈ L2（pixel 差 <0.02）→ pixel 下降另有主因（mask 质量 / 正常锚定量 / mirror 权重）→ 转实验 C（mask 校正）或 E（w 消融）。

## 6. 风险与边界

- Reinhard 传递可能过度拉伸极端颜色产生伪影 → 若 L0 反而更差，检查直方图，改用全局统计匹配（生成集整体对齐真实集整体）。
- 色差剂量必须同时报告"变换前自然色差"与"变换后残留色差"，证明操纵真实。
- 每水平单跑有方差 → 可接受：预期 Δ（参考 0.66→0.49 的 0.17 幅度）远大于噪声。
- 本实验固定 44% BN，与 BN 数量正交；不涉及泄漏（用协议干净数据）。
- 若发现色差主要在 L（亮度）而非 a*/b*（色度），结论按"亮度偏置"重新表述，机制类似。

## 7. 交付物

- 结果表: AUC/AP Image+Pixel × {L0, L1, L2, baseline, fix2_clean}
- 剂量表: Δμ/Δσ/ΔE × {变换前, L0, L1}
- 诊断表: 背景地板 / 分离比 × {L0, L1, L2}
- 结论记录 SUMMARY.md 新 §7.5；更新 memory/NEXT_STEPS
