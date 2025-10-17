import os
import sys
import argparse
import copy
import math
import random
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from tqdm import tqdm
from transformers import (
    GPT2Tokenizer,
    get_polynomial_decay_schedule_with_warmup,
)

from model import * 
from custom_dataset import *
from eval.evaluate import Evaluator
from selector.selector_models import SelectorConfig, build_selector
from selector.data_preprocess import IEMOCAPDialoguePKLDataset, make_selector_collate_fn

def print_custom(context, ref, sentence):
    """Formats the context, reference, and generated sentence for printing."""
    res = ""
    res += f"Context: {context}\n"
    res += f"GPT-2: {sentence}\n"
    res += f"Ref: {ref}\n"
    res += "---------------------------------------------------------------\n"
    return res


# ===== Selector Loading & Utilities =====

def _load_selector(ckpt_path: str, device: str):
    """
    从 .pt checkpoint 加载 selector 模型（冻结 eval）
    预期 ckpt 中包含:
      - 'state_dict': 选择器参数
      - 'config': SelectorConfig 对应的 dict（input_dim/hidden_dim/num_layers/dropout/mode/num_actions）
      - 'action_set': List[List[float]] 动作权重表（如 small6 / mix3）
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Selector ckpt not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt["config"].copy()
    # 兜底：老 ckpt 没有 num_actions 时从动作表推断
    num_actions = len(ckpt.get("action_set", [])) or cfg_dict.get("num_actions", 0)
    cfg_dict["num_actions"] = num_actions
    cfg = SelectorConfig(**cfg_dict)

    selector = build_selector("MLP", cfg)  # 如果你用的不是 MLP，这里改成对应名字
    selector.load_state_dict(ckpt["state_dict"], strict=True)
    selector.to(device).eval().requires_grad_(False)

    action_table = ckpt["action_set"]  # List[List[float]]，例如 [[0.7,0.2,0.1], ...]
    return selector, action_table


@torch.no_grad()
def _selector_pick_weights(selector, action_table, states: torch.Tensor, temperature: float = 1.0):
    """
    给定 selector 和 batch 状态向量，输出 (B, K) 的权重矩阵（K=模态数，通常=3）。
    - 离散策略：softmax(logits/temperature)，取 argmax 得动作索引，再映射到具体权重。
    """
    out = selector(states)  # {'logits': (B, A)} for discrete
    logits = out["logits"] / float(max(1e-6, temperature))
    probs = torch.softmax(logits, dim=-1)
    a_idx = probs.argmax(dim=-1)  # (B,)

    # map to weights
    at = torch.tensor(action_table, device=states.device, dtype=states.dtype)  # (A, K)
    weights = at.index_select(dim=0, index=a_idx)  # (B, K)
    return weights


def _build_states(text_feat: torch.Tensor, audio_feat: torch.Tensor, video_feat: torch.Tensor, hc_feat: torch.Tensor = None):
    """
    把多模态的句子级特征拼成 selector 的输入（和你训练 selector 时保持一致）。
    假设 text/audio/video 都是 (B, D_t/a/v) 的 pooled embedding。
    """
    xs = [text_feat, audio_feat, video_feat]
    if hc_feat is not None:
        xs.append(hc_feat)
    return torch.cat(xs, dim=-1)  # (B, D_total)


def _apply_weights(weights: torch.Tensor, text_feat: torch.Tensor, audio_feat: torch.Tensor, video_feat: torch.Tensor):
    """
    把 (B,K) 的权重作用到 3 个模态特征上，返回融合后的向量：
      fused = w_T*T + w_A*A + w_V*V
    若你的 ERGM 原始融合是 concat，这里可以返回[加权特征，再 concat 原特征] 或直接替换。
    """
    # 简单的线性融合
    comps = torch.stack([text_feat, audio_feat, video_feat], dim=1)  # (B, 3, D)
    fused = (weights.unsqueeze(-1) * comps).sum(dim=1)               # (B, D)
    return fused

class Manager:
    def __init__(self, args):
        self.args = args

        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")

        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_type)
        special_tokens = {
            "bos_token": self.args.bos_token,
            "additional_special_tokens": [self.args.sp1_token, self.args.sp2_token],
        }
        self.args.eos_token = self.tokenizer.eos_token
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]

        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.resize_token_embeddings(self.args.vocab_size)
        # === Load selector (optional) ===
        self.selector = None
        self.selector_action_table = None
        if self.args.selector_enable != "off":
            sel_device = self.args.selector_device or str(self.args.device)
            try:
                self.selector, self.selector_action_table = _load_selector(self.args.selector_ckpt, sel_device)
                print(f"[Selector] loaded: {self.args.selector_ckpt} | actions={len(self.selector_action_table)}")
            except Exception as e:
                print(f"[Selector] WARNING: failed to load selector: {e}. Continue without selector.")
                self.selector = None
                self.selector_action_table = None
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)

        if self.args.mode in ["train", "infer"]:
            print("Loading the optimizer...")
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
            self.best_ppl = sys.float_info.max
            self.last_epoch = 0

            print("Loading train & valid data...")
            ppd = PadCollate(eos_id=self.args.eos_id, args=self.args)
            if getattr(self.args, 'dataset', 'ERGM') == 'IEMOCAP':
                # IEMOCAP: audio/video from PKL, text from JSON; encode in collate
                assert self.args.train_pkls and self.args.val_pkls and self.args.iemocap_text_json, \
                    "For IEMOCAP, please set --train_pkls, --val_pkls and --iemocap_text_json"

                # Support comma-separated multiple PKLs
                train_pkl_list = [p.strip() for p in self.args.train_pkls.split(',')] \
                    if isinstance(self.args.train_pkls, str) else self.args.train_pkls
                val_pkl_list   = [p.strip() for p in self.args.val_pkls.split(',')] \
                    if isinstance(self.args.val_pkls, str) else self.args.val_pkls

                train_set = IEMOCAPDialoguePKLDataset(pkl_path=train_pkl_list[0], json_path=self.args.iemocap_text_json, split='train')
                valid_set = IEMOCAPDialoguePKLDataset(pkl_path=val_pkl_list[0], json_path=self.args.iemocap_text_json, split='val')

                collate = ppd.iemocap_collate(self.tokenizer)
                self.train_loader = DataLoader(
                    train_set, collate_fn=collate, shuffle=True,
                    batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                )
                self.valid_loader = DataLoader(
                    valid_set, collate_fn=collate, shuffle=False,
                    batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                )
            else:
                # Default ERGM baseline path: use original CustomDataset & PadCollate
                train_set = CustomDataset(self.args.train_prefix, self.args)
                valid_set = CustomDataset(self.args.valid_prefix, self.args)
                self.train_loader = DataLoader(
                    train_set, collate_fn=ppd.pad_collate, shuffle=True,
                    batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                )
                self.valid_loader = DataLoader(
                    valid_set, collate_fn=ppd.pad_collate, shuffle=False,
                    batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                )

            os.makedirs(self.args.ckpt_dir, exist_ok=True)

            num_batches = len(self.train_loader)
            args.total_train_steps = args.num_epochs * num_batches
            args.warmup_steps = int(args.warmup_ratio * args.total_train_steps)

            self.sched = get_polynomial_decay_schedule_with_warmup(
                self.optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_train_steps, power=2,
            )
            self.writer = SummaryWriter()

        if self.args.ckpt_name is not None:
            ckpt_path = os.path.join(self.args.ckpt_dir, f"{self.args.ckpt_name}.ckpt")
            if os.path.exists(ckpt_path):
                print("Loading the trained checkpoint...")
                ckpt = torch.load(ckpt_path, map_location=self.args.device)
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False) # Use strict=False to handle the new emotion_head

                if self.args.mode == "train":
                    print(f"Training will resume from checkpoint: {self.args.ckpt_name}.ckpt")
                    self.optim.load_state_dict(ckpt["optim_state_dict"])
                    self.sched.load_state_dict(ckpt["sched_state_dict"])
                    self.best_ppl = ckpt.get('ppl', sys.float_info.max) # Load best ppl if available
                    self.last_epoch = ckpt["epoch"]
                else:
                    print("Inference will start with the specified checkpoint.")
            else:
                print(f"Cannot find the specified checkpoint: {ckpt_path}")
                if self.args.mode == "train":
                    print("Training will start with an initialized model.")
                else:
                    print("Cannot run inference without a valid checkpoint.")
                    exit()

        print("Setting finished.")

    def train(self):
        self.fix_seed(self.args.seed)
        print("Training starts.")
        start_epoch = self.last_epoch + 1
        
        for epoch in range(start_epoch, start_epoch + self.args.num_epochs):
            self.model.train()
            print(f"-" * 35 + f"Epoch: {epoch}" + "-" * 35)
            
            train_total_losses = []
            train_lm_losses = [] # For PPL calculation
            train_correct_emotions = 0
            train_total_emotions = 0

            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch
                
                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device), # Ensure emotion labels are tensors
                )

                # === selector-driven guidance injection (only when enabled) ===
                if self.args.selector_enable != "off" and (self.selector is not None):
                    try:
                        input_ids, token_type_ids = self._maybe_apply_selector(input_ids, token_type_ids, imgs, auds)
                    except Exception as e:
                        print(f"[Selector][train] skip due to: {e}")

                outputs = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids,
                    labels=lm_labels, emotion_labels=emotion_labels,
                    imgs=imgs, auds=auds,
                )

                loss = outputs.loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.sched.step()

                train_total_losses.append(loss.item())

                with torch.no_grad():
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = lm_labels[..., 1:].contiguous()
                    loss_fct_lm = nn.CrossEntropyLoss()
                    lm_loss = loss_fct_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    train_lm_losses.append(lm_loss.item())

                    preds = torch.argmax(outputs.emotion_logits, dim=-1)
                    train_correct_emotions += (preds == emotion_labels).sum().item()
                    train_total_emotions += emotion_labels.size(0)

            avg_train_loss = np.mean(train_total_losses)
            avg_lm_loss = np.mean(train_lm_losses)
            train_ppl = math.exp(avg_lm_loss)
            train_acc = (train_correct_emotions / train_total_emotions) * 100

            print(f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.4f} | Train Emotion Acc: {train_acc:.2f}%")
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.writer.add_scalar("PPL/train", train_ppl, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)

            self.last_epoch += 1
            valid_loss, valid_ppl, valid_acc = self.validation()

            if valid_ppl < self.best_ppl:
                self.best_ppl = valid_ppl
                state_dict = {
                    "model_state_dict": self.model.state_dict(),
                    "optim_state_dict": self.optim.state_dict(),
                    "sched_state_dict": self.sched.state_dict(),
                    "ppl": self.best_ppl,
                    "epoch": self.last_epoch,
                }
                save_path = os.path.join(self.args.ckpt_dir, f"best_ckpt_epoch={epoch}_valid_ppl={self.best_ppl:.4f}.ckpt")
                torch.save(state_dict, save_path)
                print("*" * 10 + " Current best checkpoint is saved. " + "*" * 10)
                print(save_path)

            print(f"Best valid PPL: {self.best_ppl:.4f}")
            print(f"Current valid loss: {valid_loss:.4f} | Current valid PPL: {valid_ppl:.4f} | Current valid Emotion Acc: {valid_acc:.2f}%")
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            self.writer.add_scalar("PPL/valid", valid_ppl, epoch)
            self.writer.add_scalar("Accuracy/valid", valid_acc, epoch)

        print("Training finished!")

    def validation(self):
        print("Validation processing...")
        self.model.eval()

        valid_total_losses = []
        valid_lm_losses = []
        valid_correct_emotions = 0
        valid_total_emotions = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch

                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device),
                )

                # === selector-driven guidance injection (only when enabled) ===
                if self.args.selector_enable != "off" and (self.selector is not None):
                    try:
                        input_ids, token_type_ids = self._maybe_apply_selector(input_ids, token_type_ids, imgs, auds)
                    except Exception as e:
                        print(f"[Selector][val] skip due to: {e}")

                outputs = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids,
                    labels=lm_labels, emotion_labels=emotion_labels
                )

                valid_total_losses.append(outputs.loss.item())

                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = lm_labels[..., 1:].contiguous()
                loss_fct_lm = nn.CrossEntropyLoss()
                lm_loss = loss_fct_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                valid_lm_losses.append(lm_loss.item())

                preds = torch.argmax(outputs.emotion_logits, dim=-1)
                valid_correct_emotions += (preds == emotion_labels).sum().item()
                valid_total_emotions += emotion_labels.size(0)

        avg_valid_loss = np.mean(valid_total_losses)
        avg_lm_loss = np.mean(valid_lm_losses)
        valid_ppl = math.exp(avg_lm_loss)
        valid_acc = (valid_correct_emotions / valid_total_emotions) * 100

        if math.isnan(valid_ppl):
            valid_ppl = 1e8

        return avg_valid_loss, valid_ppl, valid_acc

    def nucleus_sampling(self, input_ids, token_type_ids, input_len):
        output_ids = []
        for pos in range(input_len, self.args.max_len):
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
            next_token_logits = outputs.logits[:, pos - 1, :]

            probs = F.softmax(next_token_logits, dim=-1)

            sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            idx_remove = cumsum_probs > self.args.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)

            probs = torch.zeros(probs.shape, device=self.args.device).scatter_(-1, sorted_idxs, sorted_probs)
            idx = torch.multinomial(probs, 1)
            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)

            if idx_item == self.args.eos_id:
                break

            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor([[self.args.sp2_id]]).to(self.args.device)
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape

        return output_ids

    def fix_seed(self, seed):
        """Sets a random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    def _pool_text_feat(self, input_ids: torch.Tensor) -> torch.Tensor:
        """用 GPT-2 的 token embedding 做均值池化，得到 (B, E) 的轻量文本特征供 selector 使用。"""
        with torch.no_grad():
            emb = self.model.transformer.wte(input_ids)  # (B, L, E)
            feat = emb.mean(dim=1)  # (B, E)
        return feat.float()

    @staticmethod
    def _pool_modal_feat(x):
        """接受 (B,D) 或 (B,T,D)，返回 (B,D)，其他情况返回 None。"""
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            try:
                x = torch.tensor(x)
            except Exception:
                return None
        if not torch.is_tensor(x):
            return None
        x = x.float()
        if x.dim() == 3:
            return x.mean(dim=1)
        return x

    def _maybe_apply_selector(self, input_ids, token_type_ids, imgs, auds):
        """
        若启用 selector：计算 (B,3) 权重 -> 生成文本 tag -> 逐样本前缀注入 -> 重新 pad。
        返回可能修改后的 (input_ids, token_type_ids)。
        """
        use_selector = (self.args.selector_enable == "train_eval") or \
                       (self.args.selector_enable == "val" and not self.model.training)
        if (not use_selector) or (self.selector is None):
            return input_ids, token_type_ids

        # 1) 构造 selector 的状态向量 states=(B,3D)
        text_feat = self._pool_text_feat(input_ids).to(self.args.device)
        vid_feat  = self._pool_modal_feat(imgs)
        aud_feat  = self._pool_modal_feat(auds)
        if vid_feat is None:
            vid_feat = torch.zeros_like(text_feat)
        else:
            vid_feat = vid_feat.to(self.args.device)
            if vid_feat.size(-1) != text_feat.size(-1):
                proj = torch.nn.Linear(vid_feat.size(-1), text_feat.size(-1), bias=False).to(self.args.device)
                with torch.no_grad():
                    vid_feat = proj(vid_feat)
        if aud_feat is None:
            aud_feat = torch.zeros_like(text_feat)
        else:
            aud_feat = aud_feat.to(self.args.device)
            if aud_feat.size(-1) != text_feat.size(-1):
                proj = torch.nn.Linear(aud_feat.size(-1), text_feat.size(-1), bias=False).to(self.args.device)
                with torch.no_grad():
                    aud_feat = proj(aud_feat)

        states = _build_states(text_feat, aud_feat, vid_feat)  # (B, 3D)

        # 2) selector 选动作 -> 得到 (B,3) 权重
        weights = _selector_pick_weights(self.selector, self.selector_action_table, states, self.args.selector_temperature)  # (B,3)

        # 3) 把权重编码为文本 tag 并前缀到每个样本（与 tokenizer 兼容，最小侵入）
        tags = []
        for w in weights.tolist():
            t, a, v = [max(0.0, float(x)) for x in w]
            tags.append(f"<mw t={t:.2f} a={a:.2f} v={v:.2f}>")
        tag_ids = [self.tokenizer.encode(t, add_special_tokens=False) for t in tags]

        B = input_ids.size(0)
        new_input = []
        new_type  = []
        for i in range(B):
            ti = torch.tensor(tag_ids[i], device=self.args.device, dtype=input_ids.dtype)
            ni = torch.cat([ti, input_ids[i]], dim=0)
            # 对应的 token_type_ids：用 sp1_id 填充 tag 段
            tti = torch.full((ti.size(0),), int(self.args.sp1_id), device=self.args.device, dtype=token_type_ids.dtype)
            nti = torch.cat([tti, token_type_ids[i]], dim=0)
            # 截断到 max_len
            L = min(ni.size(0), self.args.max_len)
            new_input.append(ni[:L])
            new_type.append(nti[:L])

        # 4) 重新 pad 成 batch
        maxL = max(x.size(0) for x in new_input)
        pad_id = self.args.eos_id
        pad_type = int(self.args.sp1_id)
        out_input = torch.full((B, maxL), pad_id, device=self.args.device, dtype=input_ids.dtype)
        out_type  = torch.full((B, maxL), pad_type, device=self.args.device, dtype=token_type_ids.dtype)
        for i in range(B):
            L = new_input[i].size(0)
            out_input[i, :L] = new_input[i]
            out_type[i, :L]  = new_type[i]
        return out_input, out_type

    def test(self):
        print("Test processing: Collecting generated texts and references...")
        self.model.eval()
        self.fix_seed(self.args.seed)

        all_hypotheses = []
        all_references = []
        all_true_labels = []
        all_losses = [] # For overall test PPL

        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch

                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device),
                )

                # === selector-driven guidance injection (only when enabled) ===
                if self.args.selector_enable != "off" and (self.selector is not None):
                    try:
                        input_ids, token_type_ids = self._maybe_apply_selector(input_ids, token_type_ids, imgs, auds)
                    except Exception as e:
                        print(f"[Selector][test] skip due to: {e}")

                for i in range(input_ids.size(0)):
                    current_input = input_ids[i].unsqueeze(0)
                    current_token_types = token_type_ids[i].unsqueeze(0)
                    
                    input_len = (current_input != self.args.eos_id).sum().item()

                    output_ids = self.nucleus_sampling(current_input[:, :input_len], current_token_types[:, :input_len], input_len)
                    hypothesis_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    all_hypotheses.append(hypothesis_text)

                    ref_ids = lm_labels[i][lm_labels[i] != -100] # Filter out padding
                    reference_text = self.tokenizer.decode(ref_ids, skip_special_tokens=True)
                    all_references.append(reference_text)
                    
                    all_true_labels.append(emotion_labels[i].item())

                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=lm_labels)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = lm_labels[..., 1:].contiguous()
                loss_fct_lm = nn.CrossEntropyLoss()
                lm_loss = loss_fct_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                all_losses.append(lm_loss.item())

        return all_hypotheses, all_references, all_true_labels, all_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="The random seed.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"], help="The running mode: train or infer.")
    parser.add_argument("--data_dir", type=str, default="data", help="The parent directory where data files are stored.")
    parser.add_argument("--train_prefix", type=str, default="train", help="The prefix of the train data files' name.")
    parser.add_argument("--valid_prefix", type=str, default="valid", help="The prefix of the validation data files' name.")
    parser.add_argument("--model_type", type=str, default="gpt2", help="The model type of GPT-2.")
    parser.add_argument("--bos_token", type=str, default="<bos>", help="The BOS token.")
    parser.add_argument("--sp1_token", type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument("--sp2_token", type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument("--gpu", type=str, default="0", help="The index of GPU to use.")
    parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument("--num_epochs", type=int, default=5, help="The number of total epochs.")
    parser.add_argument("--max_len", type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument("--max_turns", type=int, default=10, help="The maximum number of dialogue histories to include.")
    parser.add_argument("--top_p", type=float, default=0.95, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument("--ckpt_dir", type=str, default="/root/autodl-tmp/ERGM-main/save_model", help="The directory name for saved checkpoints.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="The directory name for outputs.")
    parser.add_argument("--ckpt_name", type=str, default=None, help="The name of the trained checkpoint (without extension).")
    parser.add_argument("--selector_ckpt", type=str, default="checkpoints/selector/latest.pt", help="路径指向你训练好的 selector checkpoint（.pt）")
    parser.add_argument("--selector_device", type=str, default=None, help="selector 推理放到哪块设备；默认跟主模型一致")
    parser.add_argument("--selector_temperature", type=float, default=1.0, help="离散策略温度：>1更平；<1更尖锐；一般 1.0 即可")
    parser.add_argument("--selector_enable", type=str, choices=["off", "val", "train_eval"], default="off",
                        help="off=不用selector；val=只在验证/测试用；train_eval=训练和验证都用")
    parser.add_argument("--dataset", type=str, default="ERGM", choices=["ERGM", "IEMOCAP"], help="Choose data pipeline. IEMOCAP uses PKL+JSON via IEMOCAPDialoguePKLDataset.")
    parser.add_argument("--train_pkls", type=str, default=None, help="(IEMOCAP) path to train PKL or a comma-separated list of PKLs")
    parser.add_argument("--val_pkls", type=str, default=None, help="(IEMOCAP) path to val PKL or a comma-separated list of PKLs")
    parser.add_argument("--iemocap_text_json", type=str, default=None, help="(IEMOCAP) path to JSON holding raw text/dialogue info to be encoded")
    
    args = parser.parse_args()

    model_name = args.model_type.split("/")[-1]
    args.data_dir = os.path.join(args.data_dir, model_name)
    args.ckpt_dir = os.path.join(args.ckpt_dir, model_name)
    
    if args.mode == 'train':
        manager = Manager(args)
        manager.train()
    elif args.mode == 'infer':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint using --ckpt_name."
        manager = Manager(args)
        
        hypotheses, references, true_labels, losses = manager.test()
        
        evaluator = Evaluator(device=manager.args.device)
        
        final_metrics = evaluator.evaluate_all(
            hypotheses=hypotheses,
            references=references
        )

        print("\n--- Final Evaluation Results ---")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{metric.upper():<10}: {value:.4f}")
            else:
                print(f"{metric.upper():<10}: {value}")
        print("--------------------------------")
        
        results_file_path = os.path.join(args.data_dir, f"{args.ckpt_name}_evaluation_results.txt")
        with open(results_file_path, "w", encoding="utf-8") as f:
            for metric, value in final_metrics.items():
                f.write(f"{metric}: {value}\n")
