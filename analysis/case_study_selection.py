"""
case_study_selection.py
从 alpha_log.jsonl 中按策略各选 1 个典型样本，输出到 analysis/case_studies/。

用法：
    python analysis/case_study_selection.py \
        --alpha_log outputs/meld/alpha_log.jsonl \
        --output_dir analysis/case_studies
"""
import json
import os
import argparse

import pandas as pd


def load(path: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    df = pd.DataFrame(rows)
    df = df[df['alpha_T'].notna()].reset_index(drop=True)
    return df


# ── 选取策略 ──────────────────────────────────────────────────────────────────

STRATEGIES = [
    {
        "name": "text_dominant",
        "desc": "α_T > 0.7：文本成为可靠 hub 的场景",
        "filter": lambda df: df[df['alpha_T'] > 0.7],
    },
    {
        "name": "audio_dominant",
        "desc": "α_A > 0.5：文字平淡但语气强烈",
        "filter": lambda df: df[df['alpha_A'] > 0.5],
    },
    {
        "name": "visual_dominant",
        "desc": "α_V > 0.4：话语短但表情丰富",
        "filter": lambda df: df[df['alpha_V'] > 0.4],
    },
    {
        "name": "near_tie",
        "desc": "max(α) - min(α) < 0.15：模态歧义场景",
        "filter": lambda df: df[
            df[['alpha_T', 'alpha_A', 'alpha_V']].max(axis=1)
            - df[['alpha_T', 'alpha_A', 'alpha_V']].min(axis=1)
            < 0.15
        ],
    },
]


def select_cases(df: pd.DataFrame, n_cases: int = 1) -> list[dict]:
    """按每种策略各选 n_cases 个样本（优先选 alpha 最极端的那个）。"""
    results = []
    for strat in STRATEGIES:
        subset = strat["filter"](df)
        if subset.empty:
            print(f"[WARN] strategy '{strat['name']}' matched 0 samples, skipped.")
            continue

        # 选"最典型"的那个：按主导模态的 α 值降序取第一个
        dom_col = {
            'text_dominant': 'alpha_T',
            'audio_dominant': 'alpha_A',
            'visual_dominant': 'alpha_V',
            'near_tie': None,
        }.get(strat['name'])

        if dom_col:
            rows = subset.sort_values(dom_col, ascending=False).head(n_cases)
        else:
            # near_tie：选 max-min 最小的前 n_cases 个
            spread = subset[['alpha_T', 'alpha_A', 'alpha_V']].max(axis=1) \
                     - subset[['alpha_T', 'alpha_A', 'alpha_V']].min(axis=1)
            rows = subset.loc[spread.nsmallest(n_cases).index]

        for _, row in rows.iterrows():
            case = row.to_dict()
            case['strategy'] = strat['name']
            case['strategy_desc'] = strat['desc']
            results.append(case)
            print(f"[{strat['name']}] sample_id={case.get('sample_id')} "
                  f"α=({case.get('alpha_T', 'N/A'):.3f}, "
                  f"{case.get('alpha_A', 'N/A'):.3f}, "
                  f"{case.get('alpha_V', 'N/A'):.3f}) "
                  f"emotion={case.get('emotion_label')}")
    return results


def save_cases(cases: list[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, case in enumerate(cases, start=1):
        path = os.path.join(output_dir, f'case_{i}_{case["strategy"]}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(case, f, ensure_ascii=False, indent=2)
        print(f"  saved: {path}")

    # 同时输出一个汇总文件
    summary_path = os.path.join(output_dir, 'all_cases.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"  summary: {summary_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Select case study samples from alpha_log.jsonl.')
    ap.add_argument('--alpha_log', required=True, help='Path to alpha_log.jsonl')
    ap.add_argument('--n_cases', type=int, default=1,
                    help='Number of cases to select per strategy (default: 1)')
    ap.add_argument('--output_dir', default='analysis/case_studies',
                    help='Directory to save selected cases (default: analysis/case_studies)')
    args = ap.parse_args()

    df = load(args.alpha_log)
    print(f"Loaded {len(df)} samples from {args.alpha_log}\n")

    cases = select_cases(df, n_cases=args.n_cases)
    print(f"\nSelected {len(cases)} cases, saving...")
    save_cases(cases, args.output_dir)
