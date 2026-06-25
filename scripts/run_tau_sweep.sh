#!/usr/bin/env bash
# 用法：bash scripts/run_tau_sweep.sh <dataset> <ckpt_dir> <ckpt_name> <output_dir>
# 示例：
#   bash scripts/run_tau_sweep.sh meld outputs/ramra_meld ramra_meld outputs/meld/
#   bash scripts/run_tau_sweep.sh iemocap outputs/ramra_iemocap ramra_iemocap outputs/iemocap/

set -e

DATASET=${1:?"请传入 dataset，例如 meld 或 iemocap"}
CKPT_DIR=${2:?"请传入 ckpt_dir"}
CKPT_NAME=${3:?"请传入 ckpt_name"}
OUTPUT_DIR=${4:?"请传入 output_dir"}

TAUS=(0.1 0.5 1.0 2.0 5.0)

for tau in "${TAUS[@]}"; do
    echo "===== τ=${tau} ====="
    python src/main.py \
        --mode infer \
        --dataset "${DATASET^^}" \
        --ckpt_dir "$CKPT_DIR" \
        --ckpt_name "$CKPT_NAME" \
        --selector_enable val \
        --tau "$tau" \
        --output_dir "${OUTPUT_DIR}/tau_${tau}"
done

echo "τ sweep 完成，结果在 ${OUTPUT_DIR}/"
