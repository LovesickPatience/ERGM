#!/bin/bash

ckpt_name="$1"

if [ -z "$ckpt_name" ]; then
    echo "Error: --ckpt_name is empty. Please provide a value."
else
    echo "Checkpoint name is: $ckpt_name"
    
    python src/main.py \
        --seed=0 \
        --mode="infer" \
        --data_dir="data" \
        --output_dir="outputs" \
        --model_type="gpt2" \
        --bos_token="<bos>" \
        --sp1_token="<sp1>" \
        --sp2_token="<sp2>" \
        --gpu="0" \
        --batch_size=1 \
        --max_len=1024 \
        --max_turns=35 \
        --top_p=0.8 \
        --ckpt_dir="saved_models" \
        --valid_prefix="test" \
        --ckpt_name="$ckpt_name"
fi
