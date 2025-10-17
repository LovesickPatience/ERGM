# -*- coding: utf-8 -*-
"""
Selector training entry point.
- Supports tasks: EGC (generation), ERC (emotion classif.), EC (emotion cause)
- Supports datasets: MELD / IEMOCAP / MEDIC (assumes pre-extracted PKL feature files)
- Supports RL method: DPO or PPO (DPO recommended for discrete action set)

Expected PKL structure (per sample), minimally:
{
  'text_feat': np.ndarray or list[float],
  'audio_feat': np.ndarray,
  'video_feat': np.ndarray,
  'handcrafted': np.ndarray (optional),
  # For EGC
  'input_text': str,
  'target_text': str,
  # For ERC / EC
  'text': str,
  'label': int or str,
}
You can adapt the keys via --feat-keys arguments.
"""
import argparse
import os
import sys
import time
import pickle
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

from tqdm import tqdm

from data_process.feature_extraction import TextEncoder
from selector.selector_models import SelectorConfig, build_selector
from selector.policies import DiscretePolicy
from selector.dpo_losses import dpo_loss
from selector.ppo_losses import ppo_clipped_loss, entropy_from_logits
from selector.data_preprocess import IEMOCAPDialoguePKLDataset, FeatureDataset, make_selector_collate_fn, \
    feature_dataset_collate_fn

# Optional imports for evaluators (wrap in try/except)
try:
    from transformers import (
        BartForConditionalGeneration, BartTokenizerFast,
        RobertaForSequenceClassification, RobertaTokenizerFast,
        RobertaModel, BartModel
    )
except Exception:
    BartForConditionalGeneration = None
    BartTokenizerFast = None
    RobertaForSequenceClassification = None
    RobertaTokenizerFast = None
    RobertaModel = None
    BartModel = None

try:
    from bert_score import score as bertscore, BERTScorer
except Exception:
    bertscore = None


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Evaluators (frozen)
# ----------------------------
class EGCEvaluator:
    def __init__(self, model_name: str = 'facebook/bart-base', device: str = 'cuda'):
        if BartForConditionalGeneration is None or BartTokenizerFast is None:
            raise ImportError("transformers (Bart) not available. Install transformers to use EGC evaluator.")
        self.tok = BartTokenizerFast.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        self.bertscorer = BERTScorer(
            model_type='/root/autodl-tmp/ERGM-main/tools/models/roberta-large',
            rescale_with_baseline=False,
            device=device,
            num_layers=24,
        )
        self.device = device

    @torch.no_grad()
    def generate(self, inputs: List[str], guidance: List[str] = None, max_new_tokens: int = 64) -> List[str]:
        if guidance is None:
            guidance = [""] * len(inputs)
        prompts = [(g + " " + x).strip() for x, g in zip(inputs, guidance)]
        enc = self.tok(prompts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens)
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def reward_bertscore(self, hyps: List[str], refs: List[str]) -> np.ndarray:
        if hasattr(self, 'bertscorer'):
            _, _, F = self.bertscorer.score(hyps, refs)
            return F.cpu().numpy().astype(np.float32)
        return np.zeros(len(hyps), dtype=np.float32)

    def generate_with_weights(self, inputs: List[str], weights: np.ndarray,
                              guidance: List[str] = None, max_new_tokens: int = 64,
                              num_beams: int = 4, min_new_tokens: int = 8,
                              no_repeat_ngram_size: int = 3, do_sample: bool = False,
                              top_p: float = 0.9, temperature: float = 1.0):
        if guidance is None: guidance = [""] * len(inputs)
        # 简单模板：把动作权重以 tag 注入到 prompt
        # 例如：[T=0.7 A=0.2 V=0.1] 也可以把 ERC/EC 的文本 guidance 拼上（若可用）
        prefixes = [f"[T={w[0]:.2f} A={w[1]:.2f} V={w[2]:.2f}] " for w in weights]
        prompts = [(p + g + " " + x).strip() for p, g, x in zip(prefixes, guidance, inputs)]
        enc = self.tok(prompts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            early_stopping=True,
        )
        out = self.model.generate(**enc, **gen_kwargs)
        return self.tok.batch_decode(out, skip_special_tokens=True)


class ClassifierEvaluator:
    """For ERC / EC. Frozen text classifier (RoBERTa). Reward: -CE or logit margin."""

    def __init__(self, model_name: str = 'roberta-base', num_labels: int = 7, device: str = 'cuda'):
        if RobertaForSequenceClassification is None:
            raise ImportError("transformers (RoBERTa) not available. Install transformers to use ERC/EC evaluator.")
        self.tok = RobertaTokenizerFast.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
        self.model.eval()
        self.device = device
        self.num_labels = num_labels

    @torch.no_grad()
    def reward(self, texts: List[str], labels: List[int], reduction: str = 'none') -> np.ndarray:
        enc = self.tok(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        logits = self.model(**enc).logits  # (B, C)
        y = torch.tensor(labels, device=self.device)
        ce = nn.CrossEntropyLoss(reduction='none')(logits, y)  # lower is better
        # convert to reward: negative CE
        r = (-ce).detach().cpu().numpy().astype(np.float32)
        return r


ACTION_SETS = {
    # 3 modalities: T, A, V
    'small6': [
        # [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [8 / 10, 1 / 10, 1 / 10], [7 / 10, 2 / 10, 1 / 10], [6 / 10, 2 / 10, 2 / 10],
        [1 / 10, 8 / 10, 1 / 10], [2 / 10, 7 / 10, 1 / 10], [2 / 10, 6 / 10, 2 / 10],
        [1 / 10, 1 / 10, 8 / 10], [2 / 10, 1 / 10, 7 / 10], [2 / 10, 2 / 10, 6 / 10],
        # [1 / 2, 1 / 2, 0], [6 / 10, 0, 4 / 10]
    ],
    'onehot3': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    'mix3': [
            [0.7, 0.2, 0.1],  # T 主
            [0.1, 0.7, 0.2],  # A 主
            [0.2, 0.1, 0.7],  # V 主
        ],
}


def action_idx_to_weights(action_idx: torch.Tensor, action_table) -> np.ndarray:
    """
    action_idx: (B,) tensor on any device
    return: (B, K) np.float32
    """
    idx = action_idx.detach().cpu().numpy().astype(int)
    return np.stack([np.array(action_table[i], dtype=np.float32) for i in idx], axis=0)


def policy_select_actions(selector, states: torch.Tensor, temperature: float = 1.0, greedy: bool = True):
    """
    selector -> logits; use DiscretePolicy to select per-sample actions
    return: actions(B,), logits(B,A), logp(B,)
    """
    out = selector(states)                 # {'logits': (B,A)}
    logits = out['logits']
    policy = DiscretePolicy(temperature=temperature)
    po = policy.forward(logits)
    actions = policy.greedy(po.probs) if greedy else policy.sample(po.probs)
    logp = DiscretePolicy.log_prob(logits, actions)
    return actions, logits, logp


def build_state_batch(tf: torch.Tensor, af: torch.Tensor, vf: torch.Tensor, hc: torch.Tensor = None) -> torch.Tensor:
    # simple mean-pool over feature dims if needed; assume already pooled vectors here
    xs = [tf, af, vf]
    if hc is not None:
        xs.append(hc)
    return torch.cat(xs, dim=-1)


def apply_action_weights(weights: np.ndarray, tf: torch.Tensor, af: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
    """Linear fusion before feeding to (frozen) decoders; customize for your model.
    Returns a fused representation just for the selector's scoring step.
    For EGC we still pass original modalities to generator; here we only need a proxy fusion if required.
    """
    w = torch.tensor(weights, device=tf.device, dtype=tf.dtype).view(1, -1)
    comps = torch.stack([tf, af, vf], dim=1)  # (B, 3, D)
    fused = (w.unsqueeze(-1) * comps).sum(dim=1)
    return fused  # (B, D)


# ----------------------------
# Training utilities
# ----------------------------

def make_save_name(selector_name: str, task: str, rl: str, evaluator: str, val_score: float) -> str:
    now = time.strftime("%Y%m%d%H%M", time.localtime())
    return f"{selector_name}_{task}_{rl}_{evaluator}: {val_score:.4f}_{now}"


# ----------------------------
# Main training
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['MELD', 'IEMOCAP', 'MEDIC'], required=True)
    parser.add_argument('--task', type=str, choices=['EGC', 'ERC', 'EC'], required=True)
    parser.add_argument('--rl', type=str, choices=['dpo', 'ppo'], default='dpo')
    parser.add_argument('--action_set', type=str, default='small6')
    parser.add_argument('--selector_model', type=str, default='MLP')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='CE warmup epochs for selector before DPO (0 to disable)')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Feature PKLs
    parser.add_argument('--train_pkls', type=str, nargs='+', required=True)
    parser.add_argument('--val_pkls', type=str, nargs='+', required=True)
    parser.add_argument('--iemocap_text_json', type=str, default=None,
                        help='Path to iemocap dialog JSON (text/emo/etc). Required when --dataset IEMOCAP.')

    # Keys mapping (if your PKL field names differ)
    parser.add_argument('--key_text_feat', type=str, default='text_feat')
    parser.add_argument('--key_audio_feat', type=str, default='audio_feat')
    parser.add_argument('--key_video_feat', type=str, default='video_feat')
    parser.add_argument('--key_handcrafted', type=str, default='handcrafted')
    parser.add_argument('--key_text', type=str, default='text')
    parser.add_argument('--key_target_text', type=str, default='target_text')
    parser.add_argument('--key_label', type=str, default='label')

    # Evaluators
    parser.add_argument('--egc_model', type=str, default='facebook/bart-base')
    parser.add_argument('--cls_model', type=str, default='roberta-base')
    parser.add_argument('--num_labels', type=int, default=7)

    args = parser.parse_args()
    set_seed(args.seed)

    ACTIONS = ACTION_SETS[args.action_set]

    # Load data
    key_map = {
        'text_feat': args.key_text_feat,
        'audio_feat': args.key_audio_feat,
        'video_feat': args.key_video_feat,
        'handcrafted': args.key_handcrafted,
        'text': args.key_text,
        'target_text': args.key_target_text,
        'label': args.key_label,
    }

    if args.task == 'EGC':
        tok = BartTokenizerFast.from_pretrained(args.egc_model)
        bos = tok.bos_token
        eos = tok.eos_token
    else:
        tok = RobertaTokenizerFast.from_pretrained(args.cls_model)
        bos = tok.bos_token
        eos = tok.eos_token

    if args.dataset == 'IEMOCAP':
        # Use first PKL for both train and val (IEMOCAP is a single PKL file)
        pkl_path = args.train_pkls[0]
        json_path = args.iemocap_text_json
        train_ds = IEMOCAPDialoguePKLDataset(json_path=json_path, pkl_path=pkl_path, split='train',
                                             session_filter=None, bos=bos, eos=eos,)
        val_ds = IEMOCAPDialoguePKLDataset(json_path=json_path, pkl_path=pkl_path, split='val',
                                           session_filter=None, bos=bos, eos=eos,)
    else:
        train_ds = FeatureDataset(args.train_pkls, key_map, args.task)
        val_ds = FeatureDataset(args.val_pkls, key_map, args.task)

    # Choose collate function and text encoder
    if args.dataset == 'IEMOCAP':
        text_encoder = TextEncoder(task=args.task, egc_model=args.egc_model, cls_model=args.cls_model, device=args.device)
        collate_fn = make_selector_collate_fn(text_encoder, args.task)
    else:
        collate_fn = feature_dataset_collate_fn
        text_encoder = None

    # Infer dims from first batch of DataLoader
    tmp_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    tf, af, vf, hc, texts, targets, labels = next(iter(tmp_loader))
    input_dim = tf.shape[-1] + af.shape[-1] + vf.shape[-1] + (0 if hc is None else hc.shape[-1])

    # Build selector (DISCRETE policy assumed for DPO/PPO on action set)
    actions = ACTION_SETS[args.action_set]
    actions4warmup = ACTION_SETS["mix3"]
    num_actions = len(actions)
    num_warmup_actions = len(actions4warmup)
    cfg = SelectorConfig(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                         dropout=args.dropout, mode='discrete', num_actions=num_actions)
    selector = build_selector(args.selector_model, cfg).to(args.device)

    # Build evaluator per task (frozen)
    if args.task == 'EGC':
        egc = EGCEvaluator(args.egc_model, args.device)
        evaluator_name = 'Bertscore'
    else:
        cls = ClassifierEvaluator(args.cls_model, num_labels=args.num_labels, device=args.device)
        evaluator_name = 'NegCE'

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optim = torch.optim.AdamW(selector.parameters(), lr=args.lr)
    policy = DiscretePolicy()

    # Helper to evaluate 1 batch under a specific action index → reward vector
    def batch_reward(action, tf, af, vf, hc, texts, targets, labels, action_set) -> np.ndarray:
        B = len(texts)
        if isinstance(action, int):
            w = np.array(action_set[action], dtype=np.float32)
            weights = np.tile(w, (B, 1))
        elif isinstance(action, torch.Tensor):
            weights = action_idx_to_weights(action, action_set)  # (B,K)
        else:
            raise TypeError("action must be int or torch.Tensor(B,)")

        # For EGC: generate with guidance = texts (or empty) — here we just use texts as context
        if args.task == 'EGC':
            # Simple prompt composition; replace with your actual pipeline where modalities condition the generator
            hyps = egc.generate_with_weights(texts, np.tile(weights, (len(texts), 1)))
            refs = [t if t is not None else "" for t in targets]
            r = egc.reward_bertscore(hyps, refs)
            return r
        else:
            # For ERC/EC, reward from classifier (negative CE)
            # In a full system, you'd condition inputs by modality weights; here we proxy with text only
            lab = [int(l) if l is not None else 0 for l in labels]
            return cls.reward(texts, lab)

    def warmup_selector_by_ce(warmup_epochs=1, log_interval=10):
        selector.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        for ep in range(warmup_epochs):
            epoch_loss_sum = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"[Warmup {ep + 1}/{warmup_epochs}]", dynamic_ncols=True, ascii=True, disable=not sys.stdout.isatty())

            for bidx, (tf, af, vf, hc, texts, targets, labels) in enumerate(pbar, start=1):
                tf, af, vf = tf.to(args.device), af.to(args.device), vf.to(args.device)
                if hc is not None: hc = hc.to(args.device)
                states = build_state_batch(tf, af, vf, hc)
                states = states.to(args.device).to(torch.float32)
                # pol_actions, pol_logits, pol_logp = policy_select_actions(selector, states, temperature=args.temperature,
                #                                                           greedy=True)
                # _ = batch_reward(pol_actions, tf, af, vf, hc, texts, targets, labels, actions4warmup)
                # 离线打分：用 batch_reward 评估每个 action
                with torch.no_grad():
                    rewards = []
                    for a in range(num_warmup_actions):
                        r = batch_reward(a, tf, af, vf, hc, texts, targets, labels, actions4warmup)  # (B,)
                        rewards.append(r)
                    rewards = np.stack(rewards, axis=1)  # (B, A)
                    y = rewards.argmax(axis=1)

                logits = selector(states)['logits']
                loss = ce_loss(logits, torch.from_numpy(y).to(args.device))

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
                optim.step()

                # —— 累计并按需打印 —— #
                loss_val = float(loss.item())
                epoch_loss_sum += loss_val
                n_batches += 1

                if (bidx % log_interval) == 0:
                    pbar.set_postfix({"CE": f"{loss_val:.4f}"})  # 进度条右侧显示本 batch loss

                # —— 每个 epoch 结束打印平均 loss —— #
            avg = epoch_loss_sum / max(1, n_batches)
            print(f"[Warmup][Epoch {ep + 1}] avg CE loss = {avg:.4f}")

    # -----------------
    # Training Loops
    # -----------------
    def train_one_epoch_dpo(epoch: int, log_interval=10):
        selector.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[DPO {epoch}]")
        for bidx, (tf, af, vf, hc, texts, targets, labels) in enumerate(pbar, start=1):
            tf, af, vf = tf.to(args.device), af.to(args.device), vf.to(args.device)
            if hc is not None:
                hc = hc.to(args.device)
            states = build_state_batch(tf, af, vf, hc)

            # (1) 当前策略先选动作并驱动一次生成（仅用于监控，不参与loss）
            pol_actions, pol_logits, pol_logp = policy_select_actions(selector, states, temperature=args.temperature,
                                                                      greedy=True)
            _ = batch_reward(pol_actions, tf, af, vf, hc, texts, targets, labels, actions)

            # (2) 离线枚举动作 → 构造 (a+, a-) 偏好对
            with torch.no_grad():
                reward_cols = []
                for a in range(num_actions):
                    r = batch_reward(a, tf, af, vf, hc, texts, targets, labels, actions)  # (B,)
                    reward_cols.append(r)
                R = np.stack(reward_cols, axis=1)  # (B, A)
                a_pos = R.argmax(axis=1)
                a_neg = R.argmin(axis=1)
                valid = a_pos != a_neg
            if not valid.any():
                continue

            a_pos = torch.from_numpy(a_pos[valid]).to(args.device)
            a_neg = torch.from_numpy(a_neg[valid]).to(args.device)
            states_v = states[valid]

            # (3) 严格 DPO：new vs ref
            logits_new = selector(states_v)['logits']
            # logits_ref = reference_selector(states_v)['logits']  # 冻结的 reference

            lp_pos_new, lp_neg_new = DiscretePolicy.log_probs_pair(logits_new, a_pos, a_neg)
            # lp_pos_ref, lp_neg_ref = DiscretePolicy.log_probs_pair(logits_ref, a_pos, a_neg)

            loss = dpo_loss(lp_pos_new, lp_neg_new, beta=args.beta)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optim.step()

            # 记录与显示
            loss_val = float(loss.item())
            total_loss += loss_val
            n_batches += 1

            if (bidx % log_interval) == 0:
                pbar.set_postfix({"DPO": f"{loss_val:.4f}"})

        return total_loss / max(1, n_batches)

    def validate_mean_reward(log_interval=10):
        selector.eval()
        all_rewards = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="[Validate]")
            for bidx, (tf, af, vf, hc, texts, targets, labels) in enumerate(pbar, start=1):
                tf, af, vf = tf.to(args.device), af.to(args.device), vf.to(args.device)
                if hc is not None:
                    hc = hc.to(args.device)
                states = build_state_batch(tf, af, vf, hc)
                logits = selector(states)['logits']  # (B, A)
                acts = torch.softmax(logits, dim=-1).argmax(dim=-1)  # (B,)

                # 向量化计算该 batch 的 reward（一次性），并记录
                r = batch_reward(acts, tf, af, vf, hc, texts, targets, labels, actions)  # np.ndarray (B,)
                all_rewards.extend([float(x) for x in r])

                # 简洁可视化：每 log_interval 个 batch 显示一次该 batch 的均值
                if (bidx % log_interval) == 0:
                    pbar.set_postfix({"R": f"{float(np.mean(r)):.4f}"})

        return float(np.mean(all_rewards)) if all_rewards else 0.0

    if args.warmup_epochs > 0:
        print(f"[Warmup] CE warmup for {args.warmup_epochs} epoch(s)...")
        warmup_selector_by_ce(warmup_epochs=args.warmup_epochs)

    global reference_selector
    reference_selector = deepcopy(selector).to(args.device)
    reference_selector.requires_grad_(False)
    reference_selector.eval()

    if args.rl.lower() == 'dpo':
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch_dpo(epoch)
            val_score = validate_mean_reward()
            print(f"[DPO] Epoch {epoch}: train_loss={train_loss:.4f}  val_reward={val_score:.4f}")
    # else:
    #     # Minimal PPO: do behavior cloning warmup by supervised best-action targets, then PPO updates (left as extension)
    #     raise NotImplementedError("PPO path: add rollout buffer + advantages. Start from DPO for stability.")

    # Final validation
    final_score = validate_mean_reward()
    save_name = make_save_name(args.selector_model, args.task, args.rl.upper(), evaluator_name, final_score)

    os.makedirs('checkpoints/selector', exist_ok=True)
    path = os.path.join('checkpoints/selector', save_name + '.pt')
    torch.save({
        'args': vars(args),
        'state_dict': selector.state_dict(),
        'config': cfg.__dict__,
        'action_set': actions,
        'val_score': final_score,
    }, path)
    print(f"Saved selector to: {path}")


if __name__ == '__main__':
    main()
