# Missing Boundary Guided Generation — 实验报告

## 实验配置

- **类别**: bottle
- **检测器**: DRAEM
- **生成器**: SeaS
- **特征层**: ['b3', 'b4', 'b6']
- **M(x)**: α=0.5*Gap + β=0.5*PB
- **Gap k-neighbors**: 10
- **高缺失阈值**: P90
- **BN prompt**: `a ob1`
- **BD prompt**: `a ob1 with sks1 sks2 sks3 sks4`
- **Random 种子**: 42

### 数据集

| 数据集 | 组成 |
|--------|------|
| A_baseline | Original train/good |
| B_random | Original + 50 random SeaS anomalies (seed 777 pool, seed 42 selection) |
| C_mb | Original + 50 Boundary Normal + 50 Blind Defect |

## 评估结果

| Method | Image AUROC | Pixel AUROC | Image AP | Pixel AP |
|--------|------------|------------|----------|----------|

## 生成样本统计

- **boundary_normal**: 50 张
- **blind_defect**: 50 张
- **random_anomalies**: 100 张

## Missing Boundary Top-10 (测试集)

| 图像 | Score | PB | Gap | M |
|------|-------|----|----|----|
| test_contamination_016 | 0.9720 | 0.1533 | 0.1297 | 0.1415 |
| test_contamination_006 | 0.9885 | 0.1302 | 0.1341 | 0.1321 |
| test_broken_small_003 | 0.9754 | 0.1496 | 0.1120 | 0.1308 |
| test_contamination_007 | 0.9732 | 0.1433 | 0.1119 | 0.1276 |
| test_contamination_005 | 0.9807 | 0.1386 | 0.1113 | 0.1250 |
| test_contamination_000 | 0.9719 | 0.1228 | 0.1250 | 0.1239 |
| test_broken_large_003 | 0.9745 | 0.1192 | 0.1233 | 0.1212 |
| test_broken_large_013 | 0.9746 | 0.1396 | 0.1009 | 0.1203 |
| test_contamination_020 | 0.9818 | 0.1386 | 0.1002 | 0.1194 |
| test_broken_large_016 | 0.9795 | 0.1339 | 0.1031 | 0.1185 |


## 中间发现

详见 `analysis_findings.md`
