#!/bin/bash
# τ 敏感性分析（仅 RaMRA）：对同一 RaMRA checkpoint 跑 5 个 τ 值
# 用法：
#   bash src/scripts/run_tau_sweep.sh <dataset> <ckpt_dir> <ckpt_name> <output_base_dir> [extra_args...]
#
# 示例（MELD）：
#   bash src/scripts/run_tau_sweep.sh meld /path/to/ckpts my_ramra_ckpt outputs/tau_sweep \
#       --meld_text_json_test data/meld_diadict_test.json \
#       --meld_aud_pkl_test data/meld_aud_test.pkl \
#       --meld_img_pkl_test data/meld_img_test.pkl
#
# 示例（IEMOCAP）：
#   bash src/scripts/run_tau_sweep.sh iemocap /path/to/ckpts my_ramra_ckpt outputs/tau_sweep \
#       --val_pkls data/iemocap_test.pkl \
#       --iemocap_text_json data/iemocap_text.json

set -e

DATASET=$1
CKPT_DIR=$2
CKPT_NAME=$3
OUTPUT_BASE=$4
shift 4          # 剩余参数透传给 main_with_selector

for tau in 0.1 0.5 1.0 2.0 5.0; do
    echo "==============================="
    echo "  τ = ${tau}"
    echo "==============================="
    python -m src.main_with_selector \
        --mode infer \
        --dataset "$DATASET" \
        --ckpt_dir "$CKPT_DIR" \
        --ckpt_name "$CKPT_NAME" \
        --tau "$tau" \
        --choose_use_test_or_val test \
        --batch_size 8 \
        --output_dir "${OUTPUT_BASE}/tau_${tau}" \
        "$@"
done

echo ""
echo "All τ runs done. Results in: ${OUTPUT_BASE}/tau_*/"
