# α 可视化 + Case Study 实验操作指南（用户版）

> 适用范围：α 是 RaMRA 论文提出的"主导模态"概念（三塔 selector 的 gate 输出），**只在 RaMRA 上有意义**；ERGM 没有显式 α，因此 α 可视化与多数 case 仅跑 RaMRA。
> 唯一例外：case 5 需要同时拿到 ERGM 与 RaMRA 的 emotion 预测，做"ERGM 错而 RaMRA 对"的对比，因此该 case 还需要跑一次 ERGM 推理。

## 需要从学校电脑拿回的文件

```bash
tar -czf ramra_backup.tar.gz \
  ~/ERGM/save_model/gpt2/meld/ \
  ~/ERGM/save_model/gpt2/iemocap/ \
  ~/ERGM/datasets/MELD/ \
  ~/ERGM/datasets/IEMOCAP/
```
> 注：`save_model` 下既包含 RaMRA 的 ckpt，也包含 ERGM 的 ckpt（仅 case 5 用得到 ERGM ckpt）。

## 拿到后操作步骤

### 1. 让 Agent 改代码（已完成，仅作记录）

- `main_with_selector.py` 已新增 `--save_alpha_log`：推理时把每个样本的 RaMRA α (gate) 落盘到 `outputs/{dataset}/alpha_log.jsonl`
- `analysis/alpha_visualization.py` 与 `analysis/case_study_selection.py` 已就绪

### 2. 跑 RaMRA 推理（α 落盘必须）—— 2 条命令

> 关键：必须加 `--save_alpha_log`，否则不会写 `alpha_log.jsonl`。

```bash
# MELD RaMRA
python -m src.main_with_selector \
  --mode=infer \
  --dataset=MELD \
  --model_type=/root/autodl-tmp/ERGM-main/tools/models/gpt2 \
  --ckpt_dir=/root/autodl-tmp/ERGM-main/save_model/gpt2/meld \
  --ckpt_name=<你的 RaMRA MELD ckpt 名> \
  --meld_aud_pkl_test=/root/autodl-tmp/ERGM-main/datasets/MELD/audio_feats_ndarray.pkl \
  --meld_img_pkl_test=/root/autodl-tmp/ERGM-main/datasets/MELD/img_feats_ndarray.pkl \
  --meld_text_json_test=/root/autodl-tmp/ERGM-main/datasets/MELD/meld_diadict_test.json \
  --meld_text_json_val=/root/autodl-tmp/ERGM-main/datasets/MELD/meld_diadict_val.json \
  --choose_use_test_or_val=test \
  --batch_size=2 \
  --save_alpha_log

# IEMOCAP RaMRA
python -m src.main_with_selector \
  --mode=infer \
  --dataset=IEMOCAP \
  --model_type=/root/autodl-tmp/ERGM-main/tools/models/gpt2 \
  --ckpt_dir=/root/autodl-tmp/ERGM-main/save_model/gpt2/iemocap \
  --ckpt_name=<你的 RaMRA IEMOCAP ckpt 名> \
  --val_pkls=/root/autodl-tmp/ERGM-main/datasets/IEMOCAP/test.pkl \
  --iemocap_text_json=/root/autodl-tmp/ERGM-main/datasets/IEMOCAP/iemocap_text.json \
  --choose_use_test_or_val=test \
  --batch_size=2 \
  --save_alpha_log
```

### 3. 跑 α 可视化与 case 1-4（仅 RaMRA）

```bash
# α 分布可视化（饼图 / 小提琴 / 按情感堆叠）
python analysis/alpha_visualization.py \
  --alpha_log outputs/meld/alpha_log.jsonl \
  --dataset MELD \
  --output_dir analysis/figures/

python analysis/alpha_visualization.py \
  --alpha_log outputs/iemocap/alpha_log.jsonl \
  --dataset IEMOCAP \
  --output_dir analysis/figures/

# Case 1-4：分别选出"文本主导 / 音频主导 / 视觉主导 / 三模态接近"的代表样本
python analysis/case_study_selection.py \
  --alpha_log outputs/meld/alpha_log.jsonl \
  --n_cases 5 \
  --output_dir analysis/case_studies/meld/
```

### 4. Case 5（ERGM 错 vs RaMRA 对）—— 需要同时跑 ERGM 推理

Case 5 需要"ERGM 预测错了 emotion 但 RaMRA 对了"的样本，因此除了 step 2 的 RaMRA 推理外，**还要跑一次 ERGM 推理**拿到 ERGM 的预测：

```bash
# ERGM MELD 推理（只为 case 5 服务；不需要也不可能产出 α）
python -m src.main \
  --mode=infer \
  --dataset=MELD \
  --model_type=/root/autodl-tmp/ERGM-main/tools/models/gpt2 \
  --ckpt_dir=/root/autodl-tmp/ERGM-main/save_model/gpt2/meld \
  --ckpt_name=<你的 ERGM MELD ckpt 名> \
  --meld_aud_pkl_test=/root/autodl-tmp/ERGM-main/datasets/MELD/audio_feats_ndarray.pkl \
  --meld_img_pkl_test=/root/autodl-tmp/ERGM-main/datasets/MELD/img_feats_ndarray.pkl \
  --meld_text_json_test=/root/autodl-tmp/ERGM-main/datasets/MELD/meld_diadict_test.json \
  --meld_text_json_val=/root/autodl-tmp/ERGM-main/datasets/MELD/meld_diadict_val.json \
  --choose_use_test_or_val=test \
  --batch_size=2

# IEMOCAP 同理（换 ckpt_dir / ckpt_name 与 IEMOCAP 数据参数）
```

ERGM 推理会在 `outputs/{dataset}/<ckpt_name>_infer_outputs.pkl` 保存 hypotheses / true_labels / 等。Case 5 选样脚本会同时读 RaMRA 的 `alpha_log.jsonl` 与 ERGM 的 `_infer_outputs.pkl`，挑出"ERGM 预测错、RaMRA 预测对"的样本。

> 如果 `case_study_selection.py` 当前还没支持 case 5（双模型对比），需要单独让 Agent 加：输入两个文件路径，按 emotion 预测正确性对比筛选。

### 5. 人工审核 Case + 生成 LaTeX

打开 `analysis/case_studies/meld/case_*.json` 确认文本通顺，然后跑 `analysis/case_studies_latex.py`（如还没创建则需让 Agent 创建）生成 LaTeX 表格。

## 需要 GPU 吗？

- RaMRA 推理 / ERGM 推理：最好有（CPU 也能跑，2-4h）
- 分析 / 画图 / 选 case：不需要
