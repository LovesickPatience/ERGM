# τ 敏感性分析实验操作指南（用户版）

## 实验内容

在同一个 RaMRA checkpoint 上，只改变推理时的温度参数 τ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}，观察性能变化。

> 仅 RaMRA：τ 是 RaMRA 三塔 selector 的 softmax 温度。ERGM 没有 selector / 也没有 τ 的概念，因此本实验**不在 ERGM 上做**。

**不需要重新训练。**

## 你需要的东西

- RaMRA MELD checkpoint（位于 `save_model/gpt2/meld/`）
- RaMRA IEMOCAP checkpoint（位于 `save_model/gpt2/iemocap/`）
- 测试数据（MELD 和 IEMOCAP 的 test 分片）

## 操作步骤

### 1. 让 Agent 改代码（已完成，仅作记录）

- `main_with_selector.py` 推理入口已新增 `--tau` 参数（默认 None ⇒ τ=1.0，等价于原始行为）
- `_selector_forward` 的 softmax 已替换为 `softmax(logits / τ)`
- `scripts/run_tau_sweep.sh` 与 `analysis/tau_sensitivity_plot.py` 已就绪

### 2. 跑 τ sweep

```bash
# MELD
bash scripts/run_tau_sweep.sh meld \
  /root/autodl-tmp/ERGM-main/save_model/gpt2/meld \
  <你的 RaMRA MELD ckpt 名> \
  outputs/meld/

# IEMOCAP
bash scripts/run_tau_sweep.sh iemocap \
  /root/autodl-tmp/ERGM-main/save_model/gpt2/iemocap \
  <你的 RaMRA IEMOCAP ckpt 名> \
  outputs/iemocap/
```

脚本内部使用 `python -m src.main_with_selector --mode infer --tau X` 循环 5 个 τ，每个 τ 一次推理。
5 个 τ × 2 个数据集 = 10 次推理，**每数据集约 30 分钟（GPU）**。

> 如果你的实际推理还需要数据路径参数（如 `--meld_aud_pkl_test/--meld_img_pkl_test/--meld_text_json_test/--meld_text_json_val/--choose_use_test_or_val=test/--model_type/--batch_size`），需要在 `scripts/run_tau_sweep.sh` 里把它们补全；当前脚本只透传通用参数，复杂数据集请直接编辑脚本。

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
