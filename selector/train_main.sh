#!/usr/bin/env bash
set -euo pipefail

LOGDIR="logs"
mkdir -p "${LOGDIR}"
STAMP=$(date +%F_%H%M%S)
LOG="${LOGDIR}/train_${STAMP}.log"

# 进入项目 & 环境
cd /root/autodl-tmp/ERGM-main
source ~/.bashrc
conda activate base

CMD="python -m selector.train_selector --dataset IEMOCAP --task EGC --epochs 5 --warmup_epochs 3 --lr 1e-5 --num_layers 2 --dropout 0.2 --train_pkls /root/autodl-tmp/ERGM-main/datasets/IEMOCAP/iemocap_data_0610.pkl --val_pkls /root/autodl-tmp/ERGM-main/datasets/IEMOCAP/iemocap_data_0610.pkl --iemocap_text_json /root/autodl-tmp/ERGM-main/datasets/IEMOCAP/iemocap_diadict_with_emocause.json --egc_model /root/autodl-tmp/ERGM-main/tools/models/bart-large --cls_model /root/autodl-tmp/ERGM-main/tools/models/roberta-large
"

echo "[`date`] RUN: ${CMD}" | tee -a "${LOG}"
nohup ${CMD} >> "${LOG}" 2>&1 &
disown
echo "Launched. Log: ${LOG}"