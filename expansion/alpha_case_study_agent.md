# α 可视化 + Case Study 实验构建（Agent 执行指南）

> 本文档供 CodeBuddy Agent 阅读。目标是基于 **已有代码和 checkpoint**，添加 α 值采集、可视化和 Case Study 分析管线。

---

## 一、现有代码参考

### 1.1 RaMRA 相关文件

- 模态融合模型：`src/model/model_with_fusion.py`、`src/model/qwen2_5_omni_with_fusion.py`
- 选择器/门控：`selector/selector_models.py`、`selector/policies.py`
- 训练/推理入口：`src/main.py`（已支持 `--mode infer`）
- 数据预处理：`selector/data_preprocess.py`（MELDDialoguePKLDataset、IEMOCAPDialoguePKLDataset）

### 1.2 数据加载方式（来自 main.py L156–248）

**MELD**（JSON + 独立 audio/video PKL，split=train/val/test）：
```python
from selector.data_preprocess import MELDDialoguePKLDataset
test_set = MELDDialoguePKLDataset(
    json_path=args.meld_text_json_test,
    audio_pkl=args.meld_aud_pkl_test,
    video_pkl=args.meld_img_pkl_test,
    split="test",
)
```

**IEMOCAP**（PKL + JSON，split=train/val/test）：
```python
from selector.data_preprocess import IEMOCAPDialoguePKLDataset
test_set = IEMOCAPDialoguePKLDataset(
    pkl_path=args.val_pkls, json_path=args.iemocap_text_json, split='test'
)
```

**Collate 函数**（返回 7-tuple）：
- IEMOCAP：`PadCollate(args).iemocap_collate(tokenizer)` → `iemocap_collate_without_prefix(tokenizer)`（带 fusion 时）
- MELD：`PadCollate(args).meld_collate(tokenizer)` → `meld_collate_without_prefix(tokenizer)`（带 fusion 时）

**Batch 结构**：`(input_ids[B,L], token_type_ids, labels, imgs, auds, contexts, emotion_labels)`
- `imgs`: `list[B] of list[L] of tensor[D_v]` — 每个 token 绑定一个视觉特征
- `auds`: 同上结构，音频特征
- `emotion_labels`: `list[B] of int`

### 1.3 推理入口（现有）

`main.py` 的 `Manager.__init__` 中：
- `args.mode == 'infer'` 时创建 `test_loader`
- `args.choose_use_test_or_val == 'test'` 使用测试集
- checkpoint 通过 `--ckpt_name` + `--ckpt_dir` 加载

---

## 二、你需要做的事

### 2.1 找到融合模型中 MRD/α 的输出位置

在 `model_with_fusion.py`（和 qwen 版）的 `forward()` 中搜索：
- `alpha`, `role_weight`, `gated`, `tri_tower`, `score` 等变量
- MRD 的 softmax 输出 → 这个就是 α = [α_T, α_A, α_V]

如果 α 尚未作为 forward 返回值暴露，需要添加一行：`outputs.alpha = alpha`

### 2.2 在推理 loop 中采集 α

在 `main.py` 的推理/评估 loop 中（`Manager` 类里找 `infer` 或 `evaluate` 相关方法），每次 forward 后收集：

```python
alpha_records = []
# 在推理 loop 内：
outputs = model(input_ids=input_ids, ...)
alpha = outputs.alpha  # [B, 3]
for i in range(batch_size):
    alpha_records.append({
        'sample_id': global_idx,
        'dialogue_id': _get_dialogue_id(batch, i),
        'emotion_label': emotion_id_to_name[emotion_labels[i]],
        'emotion_label_id': int(emotion_labels[i]),
        'alpha_T': float(alpha[i, 0]),
        'alpha_A': float(alpha[i, 1]),
        'alpha_V': float(alpha[i, 2]),
        'dominant_modality': ['text','audio','visual'][alpha[i].argmax().item()],
        'generated_text': decoded_text,
        'ground_truth_text': reference_text,
    })

# loop 后保存
import json
with open(f'outputs/{dataset}/alpha_log.jsonl', 'w') as f:
    for r in alpha_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
```

### 2.3 创建分析脚本

新建 `analysis/alpha_visualization.py`：

```python
import json, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

def load(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])

def plot_pie(df, name, out):
    counts = df['dominant_modality'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=['#4C72B0','#DD8452','#55A868'])
    plt.title(f'{name}: Dominant Modality')
    plt.savefig(f'{out}/alpha_pie_{name}.pdf', bbox_inches='tight'); plt.close()

def plot_violin(df, name, out):
    data = pd.melt(df[['alpha_T','alpha_A','alpha_V']], var_name='M', value_name='α')
    data['M'] = data['M'].map({'alpha_T':'Text','alpha_A':'Audio','alpha_V':'Visual'})
    sns.violinplot(data=data, x='M', y='α')
    plt.title(f'{name}: Alpha Distribution')
    plt.savefig(f'{out}/alpha_violin_{name}.pdf', bbox_inches='tight'); plt.close()

def plot_per_emotion(df, name, out):
    ct = pd.crosstab(df['emotion_label'], df['dominant_modality'], normalize='index')
    for c in ['text','audio','visual']:
        if c not in ct.columns: ct[c] = 0
    ct[['text','audio','visual']].plot(kind='bar', stacked=True,
        color=['#4C72B0','#DD8452','#55A868'])
    plt.ylabel('Proportion'); plt.title(f'{name}: Dominant Modality by Emotion')
    plt.tight_layout(); plt.savefig(f'{out}/alpha_per_emotion_{name}.pdf'); plt.close()
    ct.to_csv(f'{out}/alpha_per_emotion_{name}.csv', float_format='%.3f')
```

并添加 `__main__` 入口：
```python
if __name__ == '__main__':
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha_log', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load(args.alpha_log)
    plot_pie(df, args.dataset, args.output_dir)
    plot_violin(df, args.dataset, args.output_dir)
    plot_per_emotion(df, args.dataset, args.output_dir)
    # 打印统计报告
    for mod in ['text','audio','visual']:
        print(f"{mod}: {df['dominant_modality'].eq(mod).mean():.1%}")
```

### 2.4 创建 Case Study 选取脚本

新建 `analysis/case_study_selection.py`，从 `alpha_log.jsonl` 中按以下策略各选 1 个样本：

| 策略 | 条件 | 目的 |
|:--|:--|:--|
| Text-dominant | α_T > 0.7 | 展示文本成为可靠 hub 的场景 |
| Audio-dominant | α_A > 0.5 | 文字平淡但语气强烈 |
| Visual-dominant | α_V > 0.4 | 话语短但表情丰富 |
| Near-tie | max(α) - min(α) < 0.15 | 模态歧义场景 |
| ERGM错误/RaMRA正确 | ERGM emotion ≠ GT, RaMRA = GT | 关键对比（需同时 run ERGM 推理） |

---

## 三、输出物

| 文件 | 说明 |
|:--|:--|
| `outputs/meld/alpha_log.jsonl` | MELD 测试集 α 值 |
| `outputs/iemocap/alpha_log.jsonl` | IEMOCAP 测试集 α 值 |
| `analysis/figures/alpha_pie_MELD.pdf` 等 | 4 张可视化 PDF |
| `analysis/case_studies/case_1~5.json` | 选中的 case study 样本 |
