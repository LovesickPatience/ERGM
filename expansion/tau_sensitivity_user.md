# τ 敏感性分析实验操作指南（用户版）

## 实验内容

在同一个 RaMRA checkpoint 上，只改变推理时的温度参数 τ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}，观察性能变化。

**不需要重新训练。**

## 你需要的东西

- RaMRA MELD checkpoint（`outputs/ramra_meld/`）
- RaMRA IEMOCAP checkpoint（`outputs/ramra_iemocap/`）
- 测试数据（MELD 和 IEMOCAP 的 test 分片）

## 操作步骤

### 1. 让 Agent 改代码

把 `tau_sensitivity_agent.md` 发给 Agent，让它：
- MRD 模块添加 `tau` 参数
- 推理入口添加 `--tau` CLI 参数
- 创建 `scripts/run_tau_sweep.sh` 和 `analysis/tau_sensitivity_plot.py`

### 2. 跑 τ sweep

```bash
# MELD
bash scripts/run_tau_sweep.sh meld outputs/ramra_meld ramra_meld outputs/meld/

# IEMOCAP
bash scripts/run_tau_sweep.sh iemocap outputs/ramra_iemocap ramra_iemocap outputs/iemocap/
```

每个 τ 一次推理，5 个 τ × 2 个数据集 = 10 次推理，**每数据集约 30 分钟**。

### 3. 画图

```bash
python analysis/tau_sensitivity_plot.py --dataset MELD --output_dir outputs/meld/
python analysis/tau_sensitivity_plot.py --dataset IEMOCAP --output_dir outputs/iemocap/
```

输出：`outputs/meld/tau_sensitivity_MELD.pdf` 和 `outputs/iemocap/tau_sensitivity_IEMOCAP.pdf`

## 预期结果

- τ=1.0 附近性能最优
- τ=0.1 和 τ=5.0 有轻微下降（<3%）
- 结论：「RaMRA is robust to the choice of τ over a wide range」

## 需要 GPU 吗

- 推理：最好有（CPU 慢）
- 画图：不需要
