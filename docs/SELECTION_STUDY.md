结合现在的 **MissingBoundary → SeaS → BN/BD → MirrorEM/DRAEM → 原始 test** 闭环，我建议不要直接把“5% 面积阈值”替换成某一个新规则，而是专门做一个 **SeaS Generated Image Selection Study**。

核心问题定义为：

> **SeaS 会生成大量候选图像，但并不是每张图都值得进入训练集。如何从候选图像中筛选出真正有价值的 Boundary Normal（BN）和 Boundary Defect（BD）？**

这个实验可以同时回答两个问题：

1. **什么指标最适合判断一张生成图是否值得保留？**
2. **更好的筛选策略最终是否真的能提升异常检测器？**

---

# 一、先明确整个实验对象

你的 SeaS 生成：

[
\mathcal G=
{x_1,x_2,\cdots,x_N}
]

每张生成图最好保存：

[
\boxed{
(x_i,M_i,S_i,M(x_i),R_i)
}
]

其中：

* (x_i)：SeaS 生成图
* (M_i)：SeaS 生成 mask
* (S_i)：原始 DRAEM anomaly score
* (M(x_i))：Missing Boundary score
* (R_i)：reference image

另外从 anomaly map 中提取：

[
A_i(p)
]

也就是每个像素的 anomaly response。

于是每张 SeaS 图都有一组可以用于筛选的特征：

[
\boxed{
F_i=
[M(x_i),S_i,Area_i,TopK_i,Max_i,Region_i,Shape_i,\ldots]
}
]

---

# 二、实验总体结构

建议你不要一开始就重新训练几十个模型。

先分成两个阶段：

```text
                SeaS Candidates
                       │
                       ▼
              ┌─────────────────┐
              │ Selection Study │
              └────────┬────────┘
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
   Area-only       Score-only       Multi-evidence
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                  BN / BD sets
                       │
                       ▼
                 DRAEM retrain
                       │
                       ▼
                Original Test
                       │
                       ▼
             AUROC / AP / PRO
```

最终真正决定方案优劣的是：

[
\boxed{
\text{原始 Test 上的性能}
}
]

而不是某个筛选指标本身。

---

# 三、先固定所有实验条件

这是最重要的，否则最后无法证明是“筛选策略”带来了提升。

所有方法必须固定：

### 数据集

第一阶段建议：

[
\boxed{\text{MVTec AD}}
]

先选择：

* bottle
* cable
* capsule
* hazelnut
* metal_nut
* screw

不要一开始 15 类全部跑。

第二阶段再扩展：

[
\boxed{MVTec\ AD+VisA}
]

---

### SeaS

完全固定：

* checkpoint
* reference images
* prompt
* noise schedule
* inference steps
* generation number
* random seed

例如：

[
N_{candidate}=200
]

每个 reference 生成相同数量。

---

### Missing Boundary

固定：

[
M(x)=\alpha Gap(x)+\beta PB(x)
]

以及：

* feature extractor
* KDE bandwidth
* P1/P99
* (\alpha,\beta)
* threshold

**不要让不同 selection 方法重新优化 M(x)。**

---

### DRAEM

固定：

* backbone
* epochs
* batch size
* learning rate
* optimizer
* initialization
* random seed

否则：

> selection 方法 A 的提升可能其实来自训练超参数。

---

# 四、候选方案 0：Random

这个一定要有。

定义：

[
\boxed{
Selection_{random}
}
]

从 SeaS 候选图中随机选：

[
K
]

张。

例如：

[
K=50,100,200
]

作为最基本 baseline。

---

# 五、方案 1：只使用 M(x)

这是你当前 MissingBoundary 最纯粹的方案。

计算：

[
M(x)=\alpha Gap(x)+\beta PB(x)
]

然后选择：

[
TopK(M)
]

即：

[
x_i\in TopK(M)
]

---

### BN / BD 怎么区分？

第一版可以仍然使用：

[
Area
]

例如：

[
BN:
M(x)\ge T_M
\land Area<5%
]

[
BD:
M(x)\ge T_M
\land Area\ge5%
\land Score\in BlindBand
]

这个方案就是：

[
\boxed{M+Area}
]

它应该成为你的主要 baseline。

---

# 六、方案 2：只看 Mask Area

这是一个非常重要的对照实验。

定义：

[
A(x)=\frac{|Mask|}{HW}
]

然后：

[
BD=Area>T_A
]

[
BN=Area<T_A
]

注意：

> 这个方案**不使用 M(x)**。

它的作用不是为了成为好方法，而是回答：

> “如果单纯按照生成缺陷大小筛选，效果怎么样？”

如果：

[
M+Area

>

Area-only
]

就能证明 Missing Boundary 本身有价值。

---

# 七、方案 3：Area + Anomaly Score

加入 DRAEM 的 image-level score：

[
S(x)
]

例如：

[
S(x)=\max_p A(x)_p
]

或者你代码里的 image score。

构造：

[
\boxed{
E_1(x)=
\lambda_A A(x)+
\lambda_S S(x)
}
]

然后：

[
E_1>T_E
\Rightarrow BD
]

否则：

[
BN
]

这个方案回答：

> **面积 + 检测器响应是否比面积单独判断更好？**

---

# 八、方案 4：Area + Top-K Local Response

这是我非常建议重点实验的一个方案。

从 anomaly map：

[
A(x)
]

取最高的 (k%) 像素：

[
TopK(A)
]

定义：

[
\boxed{
S_{topk}
========

\frac{1}{|TopK|}
\sum_{p\in TopK}A(p)
}
]

同时计算：

[
S_{max}
=======

\max_p A(p)
]

于是：

[
\boxed{
E_2(x)
======

\lambda_A A(x)
+
\lambda_K S_{topk}
+
\lambda_M S_{max}
}
]

---

## 为什么这个方案非常重要？

因为它专门解决：

[
\boxed{\text{Small Defect}}
]

例如：

[
Area=0.5%
]

但：

[
S_{max}=0.95
]

那么它依然可以被判断为：

[
BD
]

而不是因为：

[
Area<5%
]

就直接变成 BN。

---

# 九、方案 5：Region-based Evidence

如果 SeaS mask 是：

[
M
]

那么只在 mask 区域计算 anomaly response：

[
\boxed{
S_{region}
==========

\frac{
\sum_{p\in M}A(p)
}{
|M|
}
}
]

同时：

[
S_{region}^{max}
================

\max_{p\in M}A(p)
]

于是：

[
\boxed{
E_3=
\lambda_1S_{region}
+\lambda_2S_{region}^{max}
}
]

这个方案的意义非常明确：

> **不是看“异常区域有多大”，而是看“这个区域内部到底有多强的异常证据”。**

---

# 十、方案 6：多证据融合

这是我认为最终最可能成为主方法的版本。

定义：

[
\boxed{
E_D(x)=
w_1A_{ratio}
+w_2S_{region}
+w_3S_{topk}
+w_4S_{max}
+w_5Shape
}
]

其中：

### 1. Area

[
A_{ratio}=\frac{|M|}{HW}
]

### 2. Region response

[
S_{region}
==========

Mean(A|M)
]

### 3. Top-K

[
S_{topk}=Mean(TopK(A))
]

### 4. Max response

[
S_{max}=\max(A)
]

### 5. Shape

可以包含：

* connected components 数量
* 最大连通区域
* compactness
* aspect ratio
* mask density

---

然后：

[
\boxed{
BN=
M(x)\ge T_M
\land
E_D<T_D
}
]

[
\boxed{
BD=
M(x)\ge T_M
\land
E_D\ge T_D
\land
S(x)\in BlindBand
}
]

这时候：

[
M(x)
]

和：

[
E_D
]

承担不同任务：

[
\boxed{
M(x)=Boundary\ Risk
}
]

[
\boxed{
E_D=Defect\ Evidence
}
]

这是我最推荐你的概念拆分。

---

# 十一、方案 7：Soft Ranking，而不是硬阈值

还有一个非常值得实验的方法：

不要：

[
BD/BN
]

一刀切。

而是直接定义：

[
P_D(x)
]

表示：

> 这张图具有 defect evidence 的程度。

例如：

[
P_D=
\sigma(E_D)
]

其中：

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

然后：

[
P_D>0.8
\Rightarrow BD
]

[
P_D<0.2
\Rightarrow BN
]

中间：

[
0.2\le P_D\le0.8
]

直接：

[
Reject
]

这样可以避免强行分类。

---

# 十二、方案 8：Top-K Ranking

甚至可以完全取消：

[
5%
]

这种绝对阈值。

对于每个 category：

### 第一步

计算：

[
M(x)
]

### 第二步

选：

[
Top\ N
]

作为 boundary candidates。

### 第三步

计算：

[
E_D
]

### 第四步

分别取：

[
TopK(E_D)
]

作为 BD：

[
BD=TopK(E_D)
]

取：

[
BottomK(E_D)
]

作为 BN：

[
BN=BottomK(E_D)
]

这样每个类别都得到固定数量：

[
|BN|=|BD|=K
]

---

# 十三、方案 9：Pareto Selection

这个方案比较有研究味道。

你的目标其实有两个：

### 目标 1

Boundary risk 高：

[
M(x)\uparrow
]

### 目标 2

Defect evidence 合适：

对于 BN：

[
E_D\downarrow
]

对于 BD：

[
E_D\uparrow
]

所以不是简单的单指标排序。

---

## BN

寻找：

[
\boxed{
M(x)\uparrow,\quad E_D(x)\downarrow
}
]

---

## BD

寻找：

[
\boxed{
M(x)\uparrow,\quad E_D(x)\uparrow
}
]

然后选择 Pareto frontier 上的样本。

这比：

[
M>threshold
]

更加严格。

---

# 十四、建议不要一下子做全部方法

实际实验可以分三层。

---

## 第一阶段：快速筛选

只做：

| ID | 方法         |
| -- | ---------- |
| S0 | Random     |
| S1 | Area       |
| S2 | M          |
| S3 | M + Area   |
| S4 | M + Score  |
| S5 | M + TopK   |
| S6 | M + Region |

先跑完。

---

# 十五、第二阶段：重点方案

从第一阶段选出最好的三个。

例如假设：

[
S3,S5,S6
]

最好。

继续：

| 方法 | 说明                 |
| -- | ------------------ |
| S3 | M + Area           |
| S5 | M + TopK           |
| S6 | M + Region         |
| S7 | M + Multi-evidence |
| S8 | M + Soft           |
| S9 | M + Pareto         |

---

# 十六、第三阶段：真正验证“为什么有效”

这一步非常重要。

不能只说：

> Multi-evidence 最好。

要知道：

> **到底是哪一个因素起作用？**

做：

[
Ablation
]

例如：

[
E_D=
Area
]

↓

[
E_D=
Area+TopK
]

↓

[
E_D=
Area+TopK+Region
]

↓

[
E_D=
Area+TopK+Region+Max
]

得到：

| Evidence    | AUROC | Pixel AP | AU-PRO |
| ----------- | ----: | -------: | -----: |
| Area        |       |          |        |
| Area + TopK |       |          |        |
| + Region    |       |          |        |
| + Max       |       |          |        |

---

# 十七、不要只看最终 DRAEM 指标

这是这个实验非常容易犯的错误。

你至少需要三层评价。

---

## Level 1：Selection Quality

问：

> 选出来的图到底怎么样？

统计：

### 接受率

[
AcceptanceRate=
\frac{N_{accepted}}{N_{generated}}
]

---

### BN/BD 数量

[
N_{BN}
]

[
N_{BD}
]

---

### 小缺陷比例

[
R_{small}
=========

\frac{
N(Area<1%)
}{
N_{BD}
}
]

---

### 人工质量评价

随机抽：

[
50\sim100
]

张。

人工标：

* 正常
* 真实异常
* 伪影
* 模糊
* 不确定

这是非常重要的。

---

# 十八、Level 2：Generation Quality

评价 SeaS 生成的图本身。

例如：

### Diversity

避免：

> 200 张图实际上长得一样。

可以计算 feature distance：

[
Diversity=
\frac{1}{N(N-1)}
\sum_{i\ne j}
d(f_i,f_j)
]

---

### Similarity to reference

不能生成得完全不像原物体。

：

[
Sim(x,x_{ref})
]

可以使用 feature cosine similarity：

[
Sim=
\frac{f(x)\cdot f(x_{ref})}
{|f(x)||f(x_{ref})|}
]

---

### Boundary score

[
M(x)
]

应该真的高。

---

# 十九、Level 3：最重要——Downstream Detection

最终：

[
BN/BD
\rightarrow
DRAEM
\rightarrow
Original\ Test
]

评估：

### Image

[
AUROC
]

[
AP
]

### Pixel

[
Pixel\ AUROC
]

[
Pixel\ AP
]

### Region

[
AU\text{-}PRO
]

---

# 二十、一定要固定训练数据量

这是非常重要的。

假设：

```text
方法 A → 200 张
方法 B → 800 张
```

然后：

[
Performance_B>Performance_A
]

你不能说明 B 的筛选策略更好。

可能只是：

[
800>200
]

所以必须控制：

[
\boxed{
|BN|=K_{BN}
}
]

[
\boxed{
|BD|=K_{BD}
}
]

例如：

[
K_{BN}=100
]

[
K_{BD}=100
]

所有方法都只提供：

[
100+100
]

张。

---

# 二十一、还应该做一个“数据量曲线”

然后再研究：

[
K=25,50,100,200
]

画：

[
Performance=f(K)
]

例如：

```text id="p6s7u8"
AU-PRO
 ^
 |                  ●
 |             ●
 |        ●
 |   ●
 |________________________> # selected samples
     25  50  100  200
```

如果你的方法在相同数据量下始终更好：

[
\boxed{
MB\ selection\ is\ more\ sample\ efficient
}
]

这会比单纯“提升 1.2%”更有说服力。

---

# 二十二、特别增加一个“小缺陷专项实验”

因为你刚才提出的正是这个问题。

把候选图按照面积分组：

[
G_1: Area<1%
]

[
G_2:1%\le Area<5%
]

[
G_3:5%\le Area<10%
]

[
G_4:Area\ge10%
]

分别统计：

[
AcceptanceRate
]

和最终训练收益。

---

## 最关键的是比较：

### Area-only

```text
<1% → BN
```

### Multi-evidence

```text
<1%
但 local response 很强
→ BD
```

然后看看：

[
Performance_{multi}

>

Performance_{area}
]

如果成立，你就获得了非常直接的实验依据：

> **固定面积阈值会错误过滤微小但高证据异常，而局部 anomaly evidence 可以恢复这些样本。**

---

# 二十三、最终实验矩阵

我建议你最终整理成下面这样：

| ID | Selection | M(x) | Area | Score | TopK | Region | Shape | Downstream |
| -- | --------- | ---: | ---: | ----: | ---: | -----: | ----: | ---------- |
| S0 | Random    |    ❌ |    ❌ |     ❌ |    ❌ |      ❌ |     ❌ | ✓          |
| S1 | Area      |    ❌ |    ✓ |     ❌ |    ❌ |      ❌ |     ❌ | ✓          |
| S2 | M-only    |    ✓ |    ❌ |     ❌ |    ❌ |      ❌ |     ❌ | ✓          |
| S3 | M+Area    |    ✓ |    ✓ |     ❌ |    ❌ |      ❌ |     ❌ | ✓          |
| S4 | M+Score   |    ✓ |    ✓ |     ✓ |    ❌ |      ❌ |     ❌ | ✓          |
| S5 | M+TopK    |    ✓ |    ✓ |     ✓ |    ✓ |      ❌ |     ❌ | ✓          |
| S6 | M+Region  |    ✓ |    ✓ |     ✓ |    ❌ |      ✓ |     ❌ | ✓          |
| S7 | M+Multi   |    ✓ |    ✓ |     ✓ |    ✓ |      ✓ |     ✓ | ✓          |
| S8 | M+Soft    |    ✓ |    ✓ |     ✓ |    ✓ |      ✓ |     ✓ | ✓          |
| S9 | M+Pareto  |    ✓ |    ✓ |     ✓ |    ✓ |      ✓ |     ✓ | ✓          |

其中：

[
\boxed{S3}
]

是你当前方法的 baseline。

我最看好：

[
\boxed{S7/S8/S9}
]

但**不要预设它们一定最好**，实验结果决定最终方法。

---

# 二十四、你最终应该得到什么结论？

理想情况下，你的实验可以形成这样一个逻辑：

### 实验 1

[
M+Area > Area
]

证明：

> Missing Boundary 信息确实有助于筛选 SeaS 样本。

### 实验 2

[
M+TopK > M+Area
]

证明：

> 仅使用 mask area 会丢失局部高响应的小缺陷。

### 实验 3

[
M+Region > M+TopK
]

证明：

> anomaly evidence 的空间一致性进一步提高筛选质量。

### 实验 4

[
Multi\ Evidence > Single\ Evidence
]

证明：

> BN/BD 应该由多维 defect evidence 判断，而不是固定面积阈值。

### 实验 5

[
MB+Selection

>

Random
]

证明：

> SeaS 不是简单“生成更多数据”就有效，而是**选择 Missing Boundary 样本才有效**。

---

# 二十五、最关键的一组最终对比

如果是为了论文，我会把最终主实验压缩成：

[
\boxed{
Baseline
}
]

[
\boxed{
Random\ SeaS
}
]

[
\boxed{
M+Area
}
]

[
\boxed{
M+Local\ Evidence
}
]

[
\boxed{
M+Multi\ Evidence\ (Ours)
}
]

全部严格控制：

[
N_{BN},N_{BD}
]

完全一致，然后全部：

[
\boxed{
\rightarrow Original\ Test
}
]

最后报告：

[
Image\ AUROC
]

[
Image\ AP
]

[
Pixel\ AUROC
]

[
Pixel\ AP
]

[
AU\text{-}PRO
]

以及：

[
\boxed{
\Delta Performance
}
]

---

## 最终推荐你当前项目的实施顺序

不要马上改代码做所有方案。按下面顺序最省时间：

**Phase A：把所有候选图的中间指标一次性保存下来**

```text
candidate.csv
├── image_path
├── ref_path
├── mask_path
├── M
├── Gap
├── PB
├── image_score
├── mask_area
├── mask_max
├── anomaly_topk
├── anomaly_region_mean
├── anomaly_region_max
├── num_components
└── ...
```

这样以后改变筛选规则**不需要重新跑 SeaS**。

**Phase B：只改变 selector**

[
S0\sim S9
]

都从同一个 `candidate.csv` 选样本。

**Phase C：先用人工检查 + acceptance statistics 筛掉明显不合理的方案。**

**Phase D：剩下 3–5 个方案再真正训练 DRAEM。**

**Phase E：全部使用相同数量的 BN/BD + 相同训练参数 + 相同原始 test。**

这样你最终得到的不是“我觉得这个筛选公式比较合理”，而是一套非常完整的：

[
\boxed{
\text{SeaS Candidate Pool}
\rightarrow
\text{Selection Ablation}
\rightarrow
\text{BN/BD Quality}
\rightarrow
\text{Detector Training}
\rightarrow
\text{Independent Test}
}
]

这套实验尤其适合你现在的项目，因为它能把**“SeaS 生成质量”和“MissingBoundary 筛选质量”彻底分开**：SeaS 固定，只研究 selector。否则后面如果 DRAEM 提升了，你很难判断究竟是 **SeaS 生成得好，还是 M(x) 筛选得好，还是只是增加了训练数据量**。
