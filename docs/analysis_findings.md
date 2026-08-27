# Missing Boundary 闭环 — 中间分析发现

## 生成质量验证（DRAEM 打分）

| 数据源 | 平均分 | min | max | 说明 |
|--------|--------|-----|-----|------|
| Boundary Normal (正常prompt) | 0.944 | 0.925 | 0.956 | 介于正常(0.97)和异常(0.83)之间 ✓ |
| Blind Defect (异常prompt) | 0.979 | 0.971 | 0.987 | 分数太高，DRAEM 容易检测 |
| Random Anomalies (异常prompt) | 0.979 | 0.962 | 0.986 | 同样容易检测 |
| 真实 contamination | 0.832 | **0.312** | 0.977 | 有真正难以检测的异常 |
| 真实 broken_small | 0.905 | 0.719 | 0.971 | |
| 真实 broken_large | 0.910 | 0.755 | 0.951 | |

## 关键发现

### 发现 1: Boundary Normal 定义有效
正常 prompt ("a ob1") 生成的图像 DRAEM 分数 0.944，确实落在正常(0.97)与异常(0.83)之间。
这符合"靠近决策边界的正常样本"的定义。 ✓

### 发现 2: Blind Defect 不够"难"
异常 prompt 生成的 Blind Defect 图像 DRAEM 分数 ~0.98，模型能轻松检测。
真正的困难样本是真实异常中的低分项（contamination min=0.31）。

**下一步优化方向**:
- 用 `add_noise_step` 控制异常强度（更小的 noise → 更 subtle 的异常）
- 用更低的 guidance_scale 减弱异常外观
- 从 SeaS 生成池中筛选"低 DRAEM 分数"的样本作为 Blind Defect（即生成后过滤）

### 发现 3: Random 与 Blind Defect 几乎相同
两者分数分布几乎一样（都是 0.979）。这说明当前的 Blind Defect 选择机制
（高 M(x) 的 test 异常）与随机选择相比没有区分度。
改进方向：让 Blind Defect 生成直接以低分真实异常为目标。

## 结论

闭环已跑通（特征提取 → MB 计算 → 生成 → 重训 → 评估）。
但 Blind Defect 的质量需要提升，否则"MB > Random"的效果不明显。
