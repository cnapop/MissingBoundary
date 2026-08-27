#!/bin/bash
# ============================================================
# Missing Boundary Guided Generation — 完整闭环 Pipeline
# ============================================================
# 用法:
#   ./run_pipeline.sh <子命令> [选项]
#
# 子命令:
#   phase0_extract     DRAEM 特征 + 分数提取
#   phase1_mb          计算 Missing Boundary (PB + Gap + M(x))
#   phase2_generate    SeaS 引导生成 (Boundary Normal + Blind Defect)
#   phase3_retrain     数据增强 + DRAEM 重训
#   phase4_evaluate    评估对比
#   full               完整闭环 (一步到位)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ---- 解析参数 ----
CATEGORY="${CATEGORY}"
GPU_ID="${GPU_ID}"
MB_ALPHA="${MB_ALPHA}"
MB_BETA="${MB_BETA}"
MB_PERCENTILE="${MB_PERCENTILE}"
MB_TOP_K="${MB_TOP_K}"
NUM_BN=100   # boundary normal 生成数量
NUM_BD=100   # blind defect 生成数量

while [[ $# -gt 0 ]]; do
    case "$1" in
        --category)  CATEGORY="$2";         shift 2 ;;
        --gpu)       GPU_ID="$2";           shift 2 ;;
        --alpha)     MB_ALPHA="$2";         shift 2 ;;
        --beta)      MB_BETA="$2";          shift 2 ;;
        --percentile) MB_PERCENTILE="$2";   shift 2 ;;
        --top-k)     MB_TOP_K="$2";         shift 2 ;;
        --num-bn)    NUM_BN="$2";           shift 2 ;;
        --num-bd)    NUM_BD="$2";           shift 2 ;;
        --help|-h)   sed -n '2,15p' "$0";   exit 0 ;;
        *)           break ;;
    esac
done

SUBCOMMAND="${1:-}"

# ---- 路径定义 ----
MB_OUTPUT="${MB_DIR}/outputs"
FEATURE_DIR="${MB_OUTPUT}/features"
MB_SCORE_DIR="${MB_OUTPUT}/mb_scores"
GEN_DIR="${MB_OUTPUT}/generated"
DATASET_DIR="${MB_OUTPUT}/datasets"
EVAL_DIR="${MB_OUTPUT}/evaluation"
LOG_DIR="${MB_OUTPUT}/logs"
mkdir -p "${LOG_DIR}"

# SeaS 路径
SEAS_CKPT="${SEAS_DIR}/outputs/checkpoints/${CATEGORY}/generation-checkpoint"
SEAS_RMP="${SEAS_DIR}/outputs/checkpoints/${CATEGORY}/mask-checkpoint/rmp"
SEAS_SD="${SEAS_DIR}/model_hub/stable-diffusion-v1-4"


# ============================================================
# Phase 0: DRAEM 特征 + 分数提取
# ============================================================
phase0_extract() {
    log "========== Phase 0: DRAEM 特征提取 (${CATEGORY}) =========="

    activate_draem

    # DRAEM checkpoint 路径
    local ckpt_dir="/data/chenjiawen/DRAEM/checkpoints"
    local data_path="/data/chenjiawen/Datasets/MVTec-AD"

    log "提取分数 + 多尺度特征..."
    log "  Feature layers: ${FEATURE_LAYERS}"

    ${DRAEM_PYTHON} "${MB_DIR}/src/extract_features.py" \
        --category "${CATEGORY}" \
        --data_path "${data_path}" \
        --checkpoint_path "${ckpt_dir}" \
        --base_model_name "${DRAEM_BASE_NAME}" \
        --output_dir "${FEATURE_DIR}" \
        --gpu_id "${GPU_ID}" \
        --feature_layers "${FEATURE_LAYERS}" \
        --max_train_images "${GAP_NUM_NORMAL}"

    check_exit "DRAEM 特征提取"

    log "输出: ${FEATURE_DIR}/${CATEGORY}/"
    ls -la "${FEATURE_DIR}/${CATEGORY}/" 2>/dev/null
}


# ============================================================
# Phase 1: Missing Boundary 计算
# ============================================================
phase1_mb() {
    log "========== Phase 1: Missing Boundary 计算 (${CATEGORY}) =========="

    local feat_dir="${FEATURE_DIR}"
    if [[ ! -d "${feat_dir}/${CATEGORY}/train_good/features" ]]; then
        log "[ERROR] 特征未找到, 请先运行 phase0_extract"
        log "  期望路径: ${feat_dir}/${CATEGORY}/train_good/features/"
        exit 1
    fi

    log "计算参数: α=${MB_ALPHA}, β=${MB_BETA}, k=${GAP_K_NEIGHBORS}, P${MB_PERCENTILE}"

    # compute_missing_boundary.py 是纯 numpy，可以用任意 python
    ${DRAEM_PYTHON} "${MB_DIR}/src/compute_missing_boundary.py" \
        --category "${CATEGORY}" \
        --output_dir "${feat_dir}" \
        --alpha "${MB_ALPHA}" \
        --beta "${MB_BETA}" \
        --k "${GAP_K_NEIGHBORS}" \
        --percentile "${MB_PERCENTILE}"

    check_exit "Missing Boundary 计算"

    local mb_dir="${feat_dir}/${CATEGORY}/missing_boundary"
    log "输出:"
    log "  - ${mb_dir}/missing_boundary.csv"
    log "  - ${mb_dir}/high_m_regions.json"
    log "  - ${mb_dir}/pb_scores.csv"
    log "  - ${mb_dir}/gap_scores.csv"

    # 显示 top-5 高缺失样本
    log "\nTop-5 Boundary Normal (高M + 低score):"
    python3 -c "
import json
with open('${mb_dir}/high_m_regions.json') as f:
    data = json.load(f)
for s in data.get('boundary_normal_samples', [])[:5]:
    print(f'  {s[\"img_name\"]}: M={s[\"M\"]:.4f}, score={s[\"score\"]:.4f}')
print('')
print(f'Top-5 Blind Defect (高M + 高score):')
for s in data.get('blind_defect_samples', [])[:5]:
    print(f'  {s[\"img_name\"]}: M={s[\"M\"]:.4f}, score={s[\"score\"]:.4f}')
"
}


# ============================================================
# Phase 2: M(x) 闭环 SeaS 生成 (v3)
# ============================================================
phase2_generate() {
    log "========== Phase 2: M(x) 闭环 SeaS 生成 (${CATEGORY}) =========="

    local high_m_json="${FEATURE_DIR}/${CATEGORY}/missing_boundary/high_m_regions.json"
    local oracle_dir="${FEATURE_DIR}/${CATEGORY}/missing_boundary/mx_oracle"
    if [[ ! -f "${high_m_json}" ]]; then
        log "[ERROR] high_m_regions.json 未找到: ${high_m_json}"
        log "请先运行 phase1_mb"
        exit 1
    fi
    if [[ ! -f "${oracle_dir}/oracle.pkl" ]]; then
        log "[ERROR] M(x) oracle 未找到: ${oracle_dir}/oracle.pkl"
        log "请先运行 phase1_mb (v3 会导出 oracle)"
        exit 1
    fi

    # 检查 SeaS checkpoint
    if [[ ! -d "${SEAS_CKPT}" ]]; then
        log "[ERROR] SeaS checkpoint 未找到: ${SEAS_CKPT}"
        exit 1
    fi

    # 编排器需要 sklearn+torch+DRAEM 模型 → DRAEM env; SeaS 子进程用 seas python
    activate_draem

    log "闭环生成 Boundary Normal + Blind Defect (BN=${NUM_BN}, BD=${NUM_BD})..."
    log "  noise 阶梯: BN=${CL_BN_NOISE_LADDER}, BD=${CL_BD_NOISE_LADDER}"
    log "  rmp 阈值: BN<${CL_RMP_THR_BN}, BD>${CL_RMP_THR_BD}, score band=${CL_BD_SCORE_BAND}"

    ${DRAEM_PYTHON} "${MB_DIR}/src/closedloop_v3/generate_mb_closedloop.py" \
        --category "${CATEGORY}" \
        --high_m_json "${high_m_json}" \
        --oracle_dir "${oracle_dir}" \
        --output_dir "${MB_DIR}/${MB_V3_OUTPUT}" \
        --dataset_dir "/data/chenjiawen/Datasets/MVTec-AD" \
        --seas_dir "${SEAS_DIR}" \
        --seas_python "${SEAS_PYTHON}" \
        --draem_ckpt_dir "/data/chenjiawen/DRAEM/checkpoints" \
        --draem_base_name "${DRAEM_BASE_NAME}" \
        --draem_dir "${DRAEM_DIR}" \
        --gpus "${GPU_ID}" \
        --score_gpu "${MB_SCORE_GPU}" \
        --num_bn "${NUM_BN}" \
        --num_bd "${NUM_BD}" \
        --num_variants "${CL_NUM_VARIANTS}" \
        --quota_per_ref "${CL_QUOTA_PER_REF}" \
        --max_attempts "${CL_MAX_ATTEMPTS}" \
        --bn_noise_ladder "${CL_BN_NOISE_LADDER}" \
        --bd_noise_ladder "${CL_BD_NOISE_LADDER}" \
        --rmp_thr_bn "${CL_RMP_THR_BN}" \
        --rmp_thr_bd "${CL_RMP_THR_BD}" \
        --score_band "${CL_BD_SCORE_BAND}"

    check_exit "M(x) 闭环 SeaS 生成"

    log "\n闭环生成结果:"
    log "  Boundary Normal: ${MB_DIR}/${MB_V3_OUTPUT}/${CATEGORY}/boundary_normal/"
    log "  Blind Defect:    ${MB_DIR}/${MB_V3_OUTPUT}/${CATEGORY}/blind_defect/"
    log "  接受漏斗:        ${MB_DIR}/${MB_V3_OUTPUT}/${CATEGORY}/generation_closedloop_summary.json"
}


# ============================================================
# Phase 3: 数据增强 + DRAEM 重训
# ============================================================
phase3_retrain() {
    log "========== Phase 3: 数据增强 + DRAEM 重训 (${CATEGORY}) =========="

    local bn_dir="${GEN_DIR}/${CATEGORY}/boundary_normal"
    local bd_dir="${GEN_DIR}/${CATEGORY}/blind_defect"

    if [[ ! -d "${bn_dir}" && ! -d "${bd_dir}" ]]; then
        log "[WARNING] 生成的样本目录未找到, 跳过增强"
        log "  期望: ${bn_dir} 或 ${bd_dir}"
        log "  请先运行 phase2_generate"
        # 继续执行, 只训练 baseline
    fi

    activate_draem

    local data_path="/data/chenjiawen/Datasets/MVTec-AD"
    local dtd_dir="/data/chenjiawen/DRAEM/datasets/dtd/images"

    # 为 DRAEM 训练准备增强数据并启动训练
    log "准备增强数据集并启动重训..."

    ${DRAEM_PYTHON} "${MB_DIR}/src/prepare_and_retrain.py" \
        --category "${CATEGORY}" \
        --dataset_dir "${data_path}" \
        --output_dir "${MB_OUTPUT}" \
        --seas_output "${GEN_DIR}" \
        --bn_output "${bn_dir}" \
        --bd_output "${bd_dir}" \
        --draem_dir "${DRAEM_DIR}" \
        --draem_python "${DRAEM_PYTHON}" \
        --dtd_dir "${dtd_dir}" \
        --checkpoint_dir "${MB_OUTPUT}/checkpoints" \
        --lr "${DRAEM_LR}" \
        --epochs "${DRAEM_EPOCHS}" \
        --bs "${DRAEM_BS}" \
        --gpu_id "${GPU_ID}"

    check_exit "DRAEM 重训"

    log "\n重训完成! Checkpoints: ${MB_OUTPUT}/checkpoints/"
}


# ============================================================
# Phase 4: 评估对比
# ============================================================
phase4_evaluate() {
    log "========== Phase 4: 评估对比 (${CATEGORY}) =========="

    activate_draem

    local data_path="/data/chenjiawen/Datasets/MVTec-AD"

    ${DRAEM_PYTHON} "${MB_DIR}/src/evaluate.py" \
        --category "${CATEGORY}" \
        --dataset_dir "${data_path}" \
        --draem_dir "${DRAEM_DIR}" \
        --draem_python "${DRAEM_PYTHON}" \
        --checkpoint_dir "${MB_OUTPUT}/checkpoints" \
        --output_dir "${EVAL_DIR}" \
        --baseline_base_name "${DRAEM_BASE_NAME}" \
        --gpu_id "${GPU_ID}"

    check_exit "评估对比"

    log "\n对比结果:"
    if [[ -f "${EVAL_DIR}/comparison_results.json" ]]; then
        cat "${EVAL_DIR}/comparison_results.json"
    fi
    if [[ -f "${EVAL_DIR}/comparison.csv" ]]; then
        log "\nCSV 格式:"
        cat "${EVAL_DIR}/comparison.csv"
    fi
}


# ============================================================
# Full Pipeline
# ============================================================
full() {
    log "========== 开始完整 Missing Boundary 闭环: ${CATEGORY} =========="
    log "参数: α=${MB_ALPHA}, β=${MB_BETA}, P${MB_PERCENTILE}, k=${GAP_K_NEIGHBORS}"

    # 确保输出目录
    mkdir -p "${FEATURE_DIR}" "${MB_SCORE_DIR}" "${GEN_DIR}" "${DATASET_DIR}" "${EVAL_DIR}"

    phase0_extract
    phase1_mb
    phase2_generate
    phase3_retrain
    phase4_evaluate

    log "\n========================================"
    log "完整闭环完成! 输出目录: ${MB_OUTPUT}"
    log "========================================"
    log "关键文件:"
    log "  - Missing Boundary 分数: ${FEATURE_DIR}/${CATEGORY}/missing_boundary/missing_boundary.csv"
    log "  - 高缺失区域: ${FEATURE_DIR}/${CATEGORY}/missing_boundary/high_m_regions.json"
    log "  - 生成样本: ${GEN_DIR}/${CATEGORY}/boundary_normal/ 和 blind_defect/"
    log "  - 对比结果: ${EVAL_DIR}/comparison_results.json"
    log "  - 对比 CSV:  ${EVAL_DIR}/comparison.csv"
}


# ==== 子命令分发 ====
case "${SUBCOMMAND}" in
    phase0_extract)  phase0_extract ;;
    phase1_mb)       phase1_mb ;;
    phase2_generate) phase2_generate ;;
    phase3_retrain)  phase3_retrain ;;
    phase4_evaluate) phase4_evaluate ;;
    full)            full ;;
    *)
        echo "用法: $0 <子命令> [选项]"
        echo ""
        echo "子命令:"
        echo "  phase0_extract   DRAEM 特征 + 分数提取"
        echo "  phase1_mb        Missing Boundary 计算 (PB + Gap + M)"
        echo "  phase2_generate  SeaS 引导生成"
        echo "  phase3_retrain   数据增强 + DRAEM 重训"
        echo "  phase4_evaluate  评估对比三个方法"
        echo "  full             完整闭环"
        echo ""
        echo "选项:"
        echo "  --category <名称>    MVTec 类别 (默认: bottle)"
        echo "  --gpu <ID>          GPU ID (默认: 0)"
        echo "  --alpha <浮点数>    Gap 权重 (默认: 0.5)"
        echo "  --beta <浮点数>     PB 权重 (默认: 0.5)"
        echo "  --percentile <0-100> 高缺失阈值百分位 (默认: 90)"
        echo "  --num-bn <N>        Boundary Normal 生成数 (默认: 100)"
        echo "  --num-bd <N>        Blind Defect 生成数 (默认: 100)"
        echo ""
        echo "快速运行完整闭环:"
        echo "  bash run_pipeline.sh full --category bottle --gpu 0"
        exit 1
        ;;
esac
