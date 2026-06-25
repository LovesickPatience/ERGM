# τ 敏感性分析实验（Agent 执行指南）

> 代码仓库已知，无需重新探索。目标是：在已有推理代码上加 τ 可变开关，跑 5 个 τ 值，画折线图。

---

## 一、背景

RaMRA 的 MRD 使用温度缩放 softmax 将分数转为角色权重：

```
α_m = exp(s_m / τ) / Σ exp(s_m' / τ)
```

- τ → 0：分布趋向 one-hot（argmax 硬选择）
- τ → ∞：分布趋向均匀（三模态均权）
- 默认 τ = 1.0

实验：在测试集上对 **同一个训练好的 checkpoint**，仅改变推理时的 τ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}，观察 PPL、BERTScore、Emo Acc 的变化曲线，验证方法对 τ 是否鲁棒。

---

## 二、实现

### 2.1 让 MRD 支持可变 τ

在 MRD 模块（`model_with_fusion.py` 或 `selector_models.py` 中搜索 `softmax` 或 `tau` 或 `temperature`）的 `forward()` 中：

```python
def forward(self, x_text, x_audio, x_visual, tau=1.0):
    # ... 计算 scores s_T, s_A, s_V ...
    alpha = torch.softmax(torch.stack([s_T, s_A, s_V], dim=-1) / tau, dim=-1)
    return alpha
```

如果当前 τ 是硬编码的，添加 `tau` 参数（默认 1.0 保持向后兼容）。

### 2.2 在推理时传入 τ

在 `main.py` 的推理 loop 中，`forward()` 调用处传入 τ：

```python
tau = getattr(self.args, 'tau', 1.0)  # 新增 CLI 参数
outputs = model(input_ids=..., tau=tau, ...)
```

添加 CLI 参数（在 `argparse` 部分）：

```python
parser.add_argument('--tau', type=float, default=1.0,
    help='Temperature for modality role distribution')
```

### 2.3 批量跑 τ 的脚本

新建 `scripts/run_tau_sweep.sh`（或 Python 版 `scripts/run_tau_sweep.py`）：

```bash
#!/bin/bash
DATASET=$1   # meld 或 iemocap
CKPT_DIR=$2
CKPT_NAME=$3
OUTPUT_DIR=$4

for tau in 0.1 0.5 1.0 2.0 5.0; do
    echo "=== τ = $tau ==="
    python src/main.py --mode infer --dataset $DATASET \
        --ckpt_name $CKPT_NAME --ckpt_dir $CKPT_DIR \
        --tau $tau \
        --choose_use_test_or_val test --batch_size 8 \
        --output_dir $OUTPUT_DIR/tau_${tau}/
done
```

### 2.4 绘制敏感性曲线

新建 `analysis/tau_sensitivity_plot.py`：

```python
import json, argparse, matplotlib.pyplot as plt, numpy as np

def load_metrics(tau_dir):
    """从 tau 目录读取 PPL/BERT/EmoAcc（具体路径根据实际输出调整）"""
    with open(f'{tau_dir}/metrics.json') as f:
        return json.load(f)

def plot(args):
    taus = [0.1, 0.5, 1.0, 2.0, 5.0]
    metrics = {'PPL': [], 'BERTScore': [], 'Emo_Acc': []}
    
    for tau in taus:
        m = load_metrics(f'{args.output_dir}/tau_{tau}')
        metrics['PPL'].append(m['ppl'])
        metrics['BERTScore'].append(m['bertscore'])
        metrics['Emo_Acc'].append(m['emo_acc'])
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(taus, values, 'o-', color='#4C72B0', markersize=8)
        ax.set_xlabel('τ')
        ax.set_ylabel(name)
        ax.set_xscale('log')
        ax.set_xticks(taus)
        ax.set_xticklabels([str(t) for t in taus])
        ax.grid(True, alpha=0.3)
        # 标注默认 τ=1.0 的值为 reference
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    fig.suptitle(f'{args.dataset} — Sensitivity to Temperature τ', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/tau_sensitivity_{args.dataset}.pdf')
    plt.close()
```

---

## 三、注意事项

1. **不需要重新训练**——只改推理时的 τ，checkpoint 不变
2. τ=0.1 时分布极尖锐，可能导致 test 时某模态几乎被忽略——这是预期行为
3. 如果所有 τ 值下 PPL/EmoAcc 波动 <3%，说明方法对 τ 不敏感 → 这是**好消息**（证明温和超参选择下性能稳定）
4. 曲线应显示 τ=1.0（训练 τ）附近性能最优，极端 τ 值（0.1, 5.0）性能轻微下降——如果相反，需要检查实现
