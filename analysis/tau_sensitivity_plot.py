"""
tau_sensitivity_plot.py
读取 τ sweep 各目录下的 metrics.json，绘制 PPL / BERTScore / Emo_Acc 随 τ 变化的折线图。

用法：
    python analysis/tau_sensitivity_plot.py \
        --output_dir outputs/tau_sweep \
        --dataset MELD \
        --save_dir analysis/figures
"""
import json
import os
import argparse

import matplotlib.pyplot as plt

TAUS = [0.1, 0.5, 1.0, 2.0, 5.0]
METRICS = [
    ("ppl",        "PPL",        True),   # (json_key, label, lower_is_better)
    ("bertscore",  "BERTScore",  False),
    ("emo_acc",    "Emo Acc",    False),
]


def load_metrics(tau_dir: str) -> dict:
    path = os.path.join(tau_dir, "metrics.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot(args):
    values = {key: [] for key, _, _ in METRICS}
    missing = []

    for tau in TAUS:
        tau_dir = os.path.join(args.output_dir, f"tau_{tau}")
        try:
            m = load_metrics(tau_dir)
            for key, _, _ in METRICS:
                values[key].append(m.get(key, float("nan")))
        except FileNotFoundError:
            missing.append(tau)
            for key, _, _ in METRICS:
                values[key].append(float("nan"))

    if missing:
        print(f"[WARN] metrics.json not found for τ={missing}, those points will be NaN.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (key, label, lower_better) in zip(axes, METRICS):
        ax.plot(TAUS, values[key], "o-", color="#4C72B0", markersize=8, linewidth=2)
        ax.set_xlabel("τ", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_xscale("log")
        ax.set_xticks(TAUS)
        ax.set_xticklabels([str(t) for t in TAUS])
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="default τ=1.0")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{args.dataset} — Sensitivity to Temperature τ", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, f"tau_sensitivity_{args.dataset}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="τ sweep 根目录，下面有 tau_0.1/ tau_0.5/ ... 子目录；pdf 也保存在此目录")
    ap.add_argument("--dataset", default="MELD",
                    help="数据集名称，仅用于图标题和文件名")
    args = ap.parse_args()
    plot(args)
