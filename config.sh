#!/bin/bash
# ============================================================
# Missing Boundary Guided Generation — 中心配置
# ============================================================

# ------------------- 基础路径 -------------------
export PROJECT_ROOT="/home/chenjiawen/workspace"
export MB_DIR="${PROJECT_ROOT}/MissingBoundary"
export DRAEM_DIR="${PROJECT_ROOT}/DRAEM"
export SEAS_DIR="${PROJECT_ROOT}/SeaS"
export DATASET_DIR="${PROJECT_ROOT}/Datasets/MVTec-AD"
export DTD_DIR="${DRAEM_DIR}/datasets/dtd/images"
export CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

# ------------------- Conda 环境 -------------------
export CONDA_BASE="/home/chenjiawen/anaconda3"
export DRAEM_CONDA="DRAEM"
export SEAS_CONDA="seas"
export DRAEM_PYTHON="${CONDA_BASE}/envs/${DRAEM_CONDA}/bin/python"
export SEAS_PYTHON="${CONDA_BASE}/envs/${SEAS_CONDA}/bin/python"

# ------------------- 实验参数 -------------------
export CATEGORY="bottle"
export GPU_ID=0
export DRAEM_LR=0.0001
export DRAEM_EPOCHS=700
export DRAEM_BS=8
export DRAEM_BASE_NAME="DRAEM_test_${DRAEM_LR}_${DRAEM_EPOCHS}_bs${DRAEM_BS}"

# 特征提取 — 使用哪些 encoder 层
export FEATURE_LAYERS="b3,b4,b6"    # 多尺度: 64x64, 32x32, 8x8

# Density Gap 参数
export GAP_K_NEIGHBORS=10
export GAP_NUM_NORMAL=200           # 构建 normal bank 用的训练图数

# Missing Boundary 融合参数
export MB_ALPHA=0.5    # Gap 权重
export MB_BETA=0.5     # PB 权重

# 高缺失区域筛选
export MB_PERCENTILE=90             # Top 10% 作为高缺失区域
export MB_TOP_K=50                  # 最多选取的样本数

# SeaS 生成
export SEAS_NUM_SAMPLES=100         # 每类生成多少张
export SEAS_NUM_INFERENCE_STEPS=25
export SEAS_GUIDANCE_SCALE=8

# ---- M(x) 闭环生成 (v3) ----
export MB_V3_OUTPUT="outputs/generated_v3"     # 闭环生成输出目录
export MB_SCORE_GPU=7                          # DRAEM 评分专用 GPU (与 SeaS 生成 GPU 分离)
export CL_QUOTA_PER_REF=8                      # 每参考图目标接受数
export CL_MAX_ATTEMPTS=6                       # 每 ref 最大阶梯轮数
export CL_NUM_VARIANTS=10                      # 每噪声级候选数 (>= batch_size 10)
export CL_BN_NOISE_LADDER="300,500,700"        # BN noise 阶梯 (低 = 贴近 ref)
export CL_BD_NOISE_LADDER="1200,1500,1800"     # BD noise 阶梯
export CL_RMP_THR_BN=0.05                      # BN 最大允许 mask 覆盖率 (正常内容)
export CL_RMP_THR_BD=0.05                      # BD 最小要求 mask 覆盖率 (含缺陷)
export CL_BD_SCORE_BAND="0.5,0.995"            # BD blind score 带 (难检测)

# 微调参数
export FT_EPOCHS=20
export FT_LR=1e-5
export FT_BATCH_SIZE=4

# ============================================================
# 工具函数
# ============================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

activate_draem() {
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" != "${DRAEM_CONDA}" ]]; then
        eval "$("${CONDA_BASE}/bin/conda" shell.bash hook)"
        conda activate "${DRAEM_CONDA}"
    fi
}

activate_seas() {
    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" != "${SEAS_CONDA}" ]]; then
        eval "$("${CONDA_BASE}/bin/conda" shell.bash hook)"
        conda activate "${SEAS_CONDA}"
    fi
}

check_exit() {
    local step_name="$1"
    if [[ $? -ne 0 ]]; then
        log "[ERROR] ${step_name} 失败，退出。"
        exit 1
    fi
    log "[OK] ${step_name} 完成。"
}
