import json
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load(path: str) -> pd.DataFrame:
    """从 alpha_log.jsonl 加载数据，过滤掉 alpha 为 None 的行。"""
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    df = pd.DataFrame(rows)
    # 过滤掉没有 alpha 的行（selector 未启用时记录的行）
    df = df[df['alpha_T'].notna()].reset_index(drop=True)
    return df


def plot_pie(df: pd.DataFrame, name: str, out: str):
    """各模态成为主导模态的占比饼图。"""
    counts = df['dominant_modality'].value_counts()
    colors = {'text': '#4C72B0', 'audio': '#DD8452', 'visual': '#55A868'}
    c = [colors.get(m, '#999999') for m in counts.index]
    plt.figure()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=c)
    plt.title(f'{name}: Dominant Modality Distribution')
    plt.savefig(os.path.join(out, f'alpha_pie_{name}.pdf'), bbox_inches='tight')
    plt.close()


def plot_violin(df: pd.DataFrame, name: str, out: str):
    """三路 α 值的分布小提琴图。"""
    data = pd.melt(
        df[['alpha_T', 'alpha_A', 'alpha_V']],
        var_name='Modality', value_name='α'
    )
    data['Modality'] = data['Modality'].map(
        {'alpha_T': 'Text', 'alpha_A': 'Audio', 'alpha_V': 'Visual'}
    )
    plt.figure()
    sns.violinplot(data=data, x='Modality', y='α',
                   palette={'Text': '#4C72B0', 'Audio': '#DD8452', 'Visual': '#55A868'})
    plt.title(f'{name}: Alpha Distribution')
    plt.savefig(os.path.join(out, f'alpha_violin_{name}.pdf'), bbox_inches='tight')
    plt.close()


def plot_per_emotion(df: pd.DataFrame, name: str, out: str):
    """各情感类别下主导模态的堆叠条形图，并保存 CSV。"""
    ct = pd.crosstab(df['emotion_label'], df['dominant_modality'], normalize='index')
    for col in ['text', 'audio', 'visual']:
        if col not in ct.columns:
            ct[col] = 0.0
    ct = ct[['text', 'audio', 'visual']]

    plt.figure(figsize=(10, 5))
    ct.plot(kind='bar', stacked=True,
            color=['#4C72B0', '#DD8452', '#55A868'],
            ax=plt.gca())
    plt.ylabel('Proportion')
    plt.title(f'{name}: Dominant Modality by Emotion')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out, f'alpha_per_emotion_{name}.pdf'))
    plt.close()

    ct.to_csv(os.path.join(out, f'alpha_per_emotion_{name}.csv'), float_format='%.3f')


def print_summary(df: pd.DataFrame):
    """打印各模态主导比例及 α 均值统计。"""
    print("\n=== Dominant Modality Ratio ===")
    for mod in ['text', 'audio', 'visual']:
        ratio = df['dominant_modality'].eq(mod).mean()
        print(f"  {mod:<8}: {ratio:.1%}")

    print("\n=== Alpha Mean ± Std ===")
    for col, label in [('alpha_T', 'Text'), ('alpha_A', 'Audio'), ('alpha_V', 'Visual')]:
        print(f"  {label:<8}: {df[col].mean():.4f} ± {df[col].std():.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Visualize alpha (modality weight) logs.')
    ap.add_argument('--alpha_log', required=True, help='Path to alpha_log.jsonl')
    ap.add_argument('--dataset', required=True, help='Dataset name (e.g. MELD / IEMOCAP)')
    ap.add_argument('--output_dir', required=True, help='Directory to save figures')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load(args.alpha_log)
    print(f"Loaded {len(df)} samples from {args.alpha_log}")

    plot_pie(df, args.dataset, args.output_dir)
    plot_violin(df, args.dataset, args.output_dir)
    plot_per_emotion(df, args.dataset, args.output_dir)
    print_summary(df)

    print(f"\nFigures saved to: {args.output_dir}")
