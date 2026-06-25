# α 可视化 + Case Study 实验操作指南（用户版）

## 需要从学校电脑拿回的文件

```bash
tar -czf ramra_backup.tar.gz \
  ~/ERGM/outputs/ramra_meld/ \
  ~/ERGM/outputs/ramra_iemocap/ \
  ~/ERGM/outputs/ergm_meld/ \
  ~/ERGM/outputs/ergm_iemocap/ \
  ~/ERGM/data/
```

## 拿到后操作步骤

### 1. 让 Agent 改代码

把 `alpha_case_study_agent.md` + 仓库路径发给 CodeBuddy Agent，让它：
- 在推理 loop 中加 α 采集
- 创建 `analysis/alpha_visualization.py` 和 `analysis/case_study_selection.py`

### 2. 跑推理（4条命令）

```bash
# MELD RaMRA
python src/main.py --mode infer --dataset MELD \
  --ckpt_name ramra_meld --ckpt_dir outputs/ramra_meld/ \
  --meld_text_json_test data/meld/test.json \
  --meld_aud_pkl_test data/meld/test_audio.pkl \
  --meld_img_pkl_test data/meld/test_video.pkl \
  --choose_use_test_or_val test --batch_size 8

# IEMOCAP RaMRA
python src/main.py --mode infer --dataset IEMOCAP \
  --ckpt_name ramra_iemocap --ckpt_dir outputs/ramra_iemocap/ \
  --val_pkls data/iemocap/test.pkl \
  --iemocap_text_json data/iemocap/iemocap_text.json \
  --choose_use_test_or_val test --batch_size 8

# ERGM 同理（换 ckpt_name 和 ckpt_dir）
```

### 3. 跑分析

```bash
python analysis/alpha_visualization.py --alpha_log outputs/meld/alpha_log.jsonl --dataset MELD --output_dir analysis/figures/
python analysis/alpha_visualization.py --alpha_log outputs/iemocap/alpha_log.jsonl --dataset IEMOCAP --output_dir analysis/figures/
python analysis/case_study_selection.py --alpha_log outputs/meld/alpha_log.jsonl --n_cases 5 --output_dir analysis/case_studies/
```

### 4. 人工审核 Case + 生成 LaTeX

打开 case_N.json 确认文本通顺，然后跑 `analysis/case_studies_latex.py` 生成表格。

## 需要 GPU 吗？

- 推理：最好有（CPU 也能跑，2-4h）
- 分析/画图/选 case：不需要
