from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math

from evaluator_model import *

try:
    from transformers import (
        BartForConditionalGeneration, BartTokenizerFast,
        RobertaForSequenceClassification, RobertaTokenizerFast,
        RobertaModel, BartModel,
        GPT2Tokenizer
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



# ----------------------------
# Evaluators (frozen)
# ----------------------------
class EGCEvaluator:
    """
    Text generator used only for reward computation in the EGC task.
    - If `model_name` contains 'bart' -> use BartForConditionalGeneration (encoder-decoder).
    - If `model_name` contains 'gpt2' -> use GPT2LMHeadModel (decoder-only).
    """

    def __init__(self, model_name: str = 'facebook/bart-base', device: str = 'cuda'):
        self.device = device
        name_lower = str(model_name).lower()

        if 'gpt2' in name_lower:
            # ---- GPT-2 branch (decoder-only) ----
            self.kind = 'gpt2'
            self.tok = GPT2Tokenizer.from_pretrained(model_name)
            special_tokens = {
                "additional_special_tokens": ["<sp1>", "<sp2>"],
            }
            num_new_tokens = self.tok.add_special_tokens(special_tokens)
            # GPT-2 没有 pad，常用做法：把 pad 设为 eos
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token

            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.model.config.eos_token_id
                # 若外部曾 add_special_tokens，这里保险起见同步一下词表大小
                try:
                    self.model.resize_token_embeddings(len(self.tok))
                except Exception:
                    pass
            self.model.eval()
            hidden = int(self.model.config.hidden_size)

            def _pick_num_heads(dim: int) -> int:
                for h in (8, 4, 2, 1):
                    if dim % h == 0:
                        return h
                return 1

            self.cross_attn_heads = _pick_num_heads(hidden)
            mha = nn.MultiheadAttention(embed_dim=hidden, num_heads=self.cross_attn_heads,
                                        bias=False, batch_first=True)
            with torch.no_grad():
                eye = torch.eye(hidden)
                mha.in_proj_weight.copy_(torch.cat([eye, eye, eye], dim=0))
                if mha.in_proj_bias is not None:
                    mha.in_proj_bias.zero_()
                mha.out_proj.weight.copy_(eye)
                if mha.out_proj.bias is not None:
                    mha.out_proj.bias.zero_()
            mha.requires_grad_(False)
            self.cross_attn_mha = mha.to(device)

            proj = nn.Linear(hidden * 4, hidden, bias=False)
            with torch.no_grad():
                eye = torch.eye(hidden).repeat(1, 4) / 4.0
                proj.weight.copy_(eye)
            proj.requires_grad_(False)
            self.cross_attn_proj = proj.to(device)

            # BERTScore 与原实现一致
            from bert_score import BERTScorer
            self.bertscorer = BERTScorer(
                model_type='/root/autodl-tmp/ERGM-main/tools/models/roberta-large',
                rescale_with_baseline=False,
                device=device,
                num_layers=24,
            )

        else:
            # ---- BART branch (encoder-decoder) ----
            if BartForConditionalGeneration is None or BartTokenizerFast is None:
                raise ImportError("transformers (Bart) not available. Install transformers to use EGC evaluator.")
            self.kind = 'bart'
            self.tok = BartTokenizerFast.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
            self.model.eval()

            from bert_score import BERTScorer
            self.bertscorer = BERTScorer(
                model_type='/root/autodl-tmp/ERGM-main/tools/models/roberta-large',
                rescale_with_baseline=True,
                device=device,
                num_layers=None,
                lang='en',
                idf=True
            )

    @torch.no_grad()
    def generate(self, inputs: List[str], guidance: List[str] = None, max_new_tokens: int = 64) -> List[str]:
        if guidance is None:
            guidance = [""] * len(inputs)
        prompts = [(g + " " + x).strip() for x, g in zip(inputs, guidance)]

        if self.kind == 'gpt2':
            enc = self.tok(prompts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
            out_ids = self.model.generate(**enc, **gen_kwargs)
            return self.tok.batch_decode(out_ids, skip_special_tokens=True)
        else:
            enc = self.tok(prompts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            out_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens)
            return self.tok.batch_decode(out_ids, skip_special_tokens=True)

    @staticmethod
    def _alpha_ratio(s: str) -> float:
        if not s: return 0.0
        letters = re.sub(r'[^A-Za-z]+', '', s)
        return len(letters) / max(1, len(s))

    def reward_bertscore(self, hyps: List[str], refs: List[str]) -> np.ndarray:
        # 真正打分（确保顺序：候选在前，参考在后）
        P, R, F = self.bertscorer.score(hyps, refs, verbose=False)
        f = F.cpu().numpy()

        for i, h in enumerate(hyps):
            if h is None or len(h.strip()) < 3:
                f[i] = 0.0
                continue
            # 3) 符号比过高、字母占比过低 → 线性惩罚
            ar = EGCEvaluator._alpha_ratio(h)
            if ar < 0.3:  # 阈值可调 0.2~0.4
                f[i] = f[i] * (ar / 0.3)

        return f

    @torch.no_grad()
    def generate_with_weights(
            self,
            inputs: List[str],
            weights: np.ndarray,
            guidance: List[str] = None,
            max_new_tokens: int = 64,
            num_beams: int = 4,
            min_new_tokens: int = 8,
            no_repeat_ngram_size: int = 3,
            do_sample: bool = False,
            top_p: float = 0.9,
            temperature: float = 1.0,
    ) -> List[str]:
        if guidance is None:
            guidance = [""] * len(inputs)
        prefixes = [f"[T={w[0]:.2f} A={w[1]:.2f} V={w[2]:.2f}] " for w in weights]
        prompts = [(p + g + " " + x).strip() for p, g, x in zip(prefixes, guidance, inputs)]

        if self.kind == 'gpt2':
            enc = self.tok(prompts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
            out_ids = self.model.generate(**enc, **gen_kwargs)
            return self.tok.batch_decode(out_ids, skip_special_tokens=True)
        else:
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
            out_ids = self.model.generate(**enc, **gen_kwargs)
            return self.tok.batch_decode(out_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_modalities(
            self,
            contexts: List[str],
            aud_vecs: torch.Tensor,  # (B, D_a)
            vid_vecs: torch.Tensor,  # (B, D_v)
            weights: np.ndarray,  # (B, 3) -> [T, A, V]
            max_new_tokens: int = 64,
            num_beams: int = 1,
            do_sample: bool = False,
            top_p: float = 0.9,
            temperature: float = 1.0,
            alpha_aud: float = 0.05,  # 更小、更稳
            alpha_vid: float = 0.05,
    ) -> List[str]:
        # 该方法存在一定的问题，轻量注入不可靠！！


        # 非 GPT-2 分支走 prefix 路线
        if self.kind != 'gpt2':
            return self.generate_with_weights(contexts, weights, max_new_tokens=max_new_tokens,
                                              num_beams=num_beams, do_sample=do_sample,
                                              top_p=top_p, temperature=temperature)

        device = next(self.model.parameters()).device
        tok = self.tok

        # Helper: fetch <img> and <aud> token ids, add if missing
        img_id = tok.convert_tokens_to_ids("<img>")
        aud_id = tok.convert_tokens_to_ids("<aud>")
        if img_id is None or aud_id is None or img_id == tok.unk_token_id or aud_id == tok.unk_token_id:
            # Safety: if special tokens somehow not added, add now and resize
            tok.add_special_tokens({"additional_special_tokens": ["<img>", "<aud>"]})
            try:
                self.model.resize_token_embeddings(len(tok))
            except Exception:
                pass
            img_id = tok.convert_tokens_to_ids("<img>")
            aud_id = tok.convert_tokens_to_ids("<aud>")

        # Tokenize and ensure <img> and <aud> are present at the start of each context
        enc0 = tok(contexts, padding=True, truncation=True, max_length=1024-128, return_tensors='pt')
        input_ids = enc0['input_ids']
        attn_mask = enc0['attention_mask']

        # Ensure each sequence explicitly contains <img> and <aud> at the very beginning.
        # If missing, we prepend them (after BOS if any).
        new_ids = []
        new_mask = []
        for row_ids, row_mask in zip(input_ids, attn_mask):
            ids = row_ids.tolist()
            has_img = img_id in ids[:8]  # only trust presence near the beginning
            has_aud = aud_id in ids[:8]
            prefix = []
            if not has_img:
                prefix.append(img_id)
            if not has_aud:
                prefix.append(aud_id)
            if prefix:
                ids = prefix + ids
                row_mask = torch.cat([torch.ones(len(prefix), dtype=row_mask.dtype), row_mask], dim=0)
            new_ids.append(torch.tensor(ids, dtype=torch.long))
            new_mask.append(row_mask)

        # Pad back to a rectangular tensor
        max_len_now = max(t.numel() for t in new_ids)
        def _pad_to(t, L, pad=tok.pad_token_id):
            if t.numel() < L:
                return torch.cat([t, torch.full((L - t.numel(),), pad, dtype=torch.long)], dim=0)
            return t[:L]
        input_ids_full = torch.stack([_pad_to(t, max_len_now) for t in new_ids], dim=0)
        attn_mask_full = torch.stack([_pad_to(m, max_len_now, pad=0) for m in new_mask], dim=0)

        # Cap prompt length from the left (keep the most recent tokens)
        max_positions = int(getattr(self.model.config, "n_positions", 1024))
        safety = 2
        max_prompt_len = max(8, max_positions - int(max_new_tokens) - safety)
        if input_ids_full.size(1) > max_prompt_len:
            input_ids = input_ids_full[:, -max_prompt_len:]
            attn_mask = attn_mask_full[:, -max_prompt_len:]
        else:
            input_ids = input_ids_full
            attn_mask = attn_mask_full

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device, dtype=torch.long)

        # -> (B, L, H) 且 dtype/device 与 embedding 权重对齐
        emb_w = self.model.transformer.wte.weight
        dtype = emb_w.dtype
        inputs_embeds = self.model.transformer.wte(input_ids).to(device=device, dtype=dtype)

        B, L, H = inputs_embeds.shape
        # --- Build explicit position_ids to avoid overflow in wpe lookup ---
        # Ensure L does not exceed max_positions (already truncated above)
        position_ids = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, L)

        # Sanity: attn_mask must be 0/1 long dtype
        if attn_mask.dtype != torch.long:
            attn_mask = attn_mask.long()
        attn_mask = attn_mask.clamp_(0, 1)

        # Extra guard: make sure token ids are within vocab (defensive; we already used our tokenizer)
        vocab_size = int(self.model.transformer.wte.weight.size(0))
        if torch.any((input_ids < 0) | (input_ids >= vocab_size)):
            # replace out-of-range ids with pad
            pad_id = int(getattr(self.model.config, "pad_token_id", tok.pad_token_id))
            input_ids = input_ids.clamp(min=0, max=vocab_size - 1)
            input_ids[~((input_ids >= 0) & (input_ids < vocab_size))] = pad_id
            # re-embed after correction
            inputs_embeds = self.model.transformer.wte(input_ids).to(device=device, dtype=dtype)

        # --------- 将模态向量对齐到 H 并做规范化 ----------
        def _to_hidden(x: torch.Tensor, hidden: int) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.to(device=device, dtype=dtype)
            D = x.size(-1)
            if D == hidden:
                y = x
            elif D > hidden:
                y = x[..., :hidden]
            else:
                pad = torch.zeros(x.size(0), hidden - D, dtype=dtype, device=device)
                y = torch.cat([x, pad], dim=-1)
            # 数值安全：去 NaN/Inf + 规一化
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nn.functional.layer_norm(y, (hidden,))
            return y

        aud_vecs = _to_hidden(aud_vecs, H)  # (B, H)
        vid_vecs = _to_hidden(vid_vecs, H)  # (B, H)

        # 权重（来自 selector）：(B,3)
        w = torch.as_tensor(weights, device=device, dtype=dtype)
        # 只用到 A/V 两路
        w_a = w[:, 1:2]  # (B,1)
        w_v = w[:, 2:3]  # (B,1)

        # --------- 轻量注入到 <img>/<aud> 位置（查找实际位置） ----------
        # Locate the actual positions of <img> and <aud> per sample; fallback to 1/2.
        t_img_list = []
        t_aud_list = []
        for b in range(B):
            ids_b = input_ids[b]
            pos_img = (ids_b == img_id).nonzero(as_tuple=False)
            pos_aud = (ids_b == aud_id).nonzero(as_tuple=False)
            t_img_list.append(int(pos_img[0].item()) if pos_img.numel() > 0 else 1)
            t_aud_list.append(int(pos_aud[0].item()) if pos_aud.numel() > 0 else 2)

        # 缩放、数值裁剪，避免爆炸
        add_img = alpha_vid * (w_v * vid_vecs)  # (B,H)
        add_aud = alpha_aud * (w_a * aud_vecs)  # (B,H)
        add_img = torch.nan_to_num(add_img, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-5, 5)
        add_aud = torch.nan_to_num(add_aud, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-5, 5)

        for b in range(B):
            ti = max(0, min(int(t_img_list[b]), L - 1))
            ta = max(0, min(int(t_aud_list[b]), L - 1))
            inputs_embeds[b, ti, :] = (inputs_embeds[b, ti, :] + add_img[b]).clamp_(-10, 10)
            inputs_embeds[b, ta, :] = (inputs_embeds[b, ta, :] + add_aud[b]).clamp_(-10, 10)

        # 最后一次全面检查，发现问题就置零，避免 CUBLAS 崩溃
        if not torch.isfinite(inputs_embeds).all():
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0)

        # 保证 config 里也有 pad/eos
        self.model.config.pad_token_id = getattr(self.model.config, "pad_token_id", tok.pad_token_id)
        self.model.config.eos_token_id = getattr(self.model.config, "eos_token_id", tok.eos_token_id)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=4,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            use_cache=False,  # inputs_embeds 路径更稳
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )

        # --- Single-step forward sanity check (avoids CUBLAS crashes in generate) ---
        try:
            with torch.no_grad():
                _ = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
        except Exception as e:
            print("[SanityCheck] GPT-2 forward failed:", repr(e),
                  "| shapes: emb", tuple(inputs_embeds.shape),
                  "mask", tuple(attn_mask.shape), "pos", tuple(position_ids.shape))
            # Optional fallback: pure-prefix generation without inputs_embeds
            # enc_text = tok(contexts, padding=True, truncation=True, max_length=max_prompt_len, return_tensors='pt').to(device)
            # return tok.batch_decode(self.model.generate(**enc_text, **gen_kwargs), skip_special_tokens=True)
            raise

        out_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            **gen_kwargs
        )
        return tok.batch_decode(out_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_cross_attn(
        self,
        contexts: List[str],
        aud_vecs: torch.Tensor,   # (B, D_a)
        vid_vecs: torch.Tensor,   # (B, D_v)
        weights: np.ndarray,      # (B,3) -> [T, A, V]
        max_new_tokens: int = 64,
        num_beams: int = 1,
        do_sample: bool = False,
        top_p: float = 0.9,
        temperature: float = 1.0,
        alpha: float = 0.25,          # cross-attn injection strength
        boost_main: float = 1.5,      # extra gain for the non-text main modality
    ) -> List[str]:
        """Cross-attn style fusion for GPT-2 branch.
        - If self.kind != 'gpt2', falls back to prefix-weight prompting.
        - Treat text tokens as queries; audio/video vectors are keys/values.
        - We compute token-wise attention from tokens -> {aud,vid}, form a fused vector, then
          inject it back into token embeddings (additive, scaled). This avoids changing hidden size.
        - If the selector says A or V is the main modality, we boost the corresponding channel.
        """
        if self.kind != 'gpt2':
            return self.generate_with_weights(contexts, weights, max_new_tokens=max_new_tokens,
                                              num_beams=num_beams, do_sample=do_sample,
                                              top_p=top_p, temperature=temperature)

        device = next(self.model.parameters()).device
        tok = self.tok

        # 1) Tokenize contexts only (no special placeholders required here)
        prompts = [(c if c is not None else "").strip() for c in contexts]
        prompts = [p if p else "[CTX]" for p in prompts]

        enc = tok(prompts, padding=True, truncation=True, max_length=1024 - 128, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attn_mask = enc['attention_mask'].to(device, dtype=torch.long)
        # seq_lens = attn_mask.sum(dim=1).clamp_min(1)
        # max_valid = int(seq_lens.max().item())
        # if max_valid < attn_mask.size(1):
        #     # GPT-2 tokenizer默认左填充，必须保留右侧真实 token，不能简单截断前半部分
        #     input_ids = input_ids[:, :, max_valid]
        #     attn_mask = attn_mask[:, :, max_valid]

        # position ids computed from mask so paddings stay at position 0
        position_ids = attn_mask.cumsum(-1) - 1
        position_ids = position_ids.clamp_min_(0)

        # 2) Embed tokens -> (B, L, H)
        emb_w = self.model.transformer.wte.weight
        dtype = emb_w.dtype
        tok_emb = self.model.transformer.wte(input_ids)
        pos_emb = self.model.transformer.wpe(position_ids)
        inputs_embeds = tok_emb.to(device=device, dtype=dtype)

        B, L, H = inputs_embeds.shape

        ##################### debug #############################
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=4,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            use_cache=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )
        out_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            **gen_kwargs,
        )
        res = tok.batch_decode(out_ids, skip_special_tokens=True)
        ##########################################################

        # 3) Project modality vectors to hidden size H (linear projection + layer_norm), numeric safe

        def _proj_to_hidden(x: torch.Tensor, name: str) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.to(device=device, dtype=dtype)
            D = x.size(-1)
            attr = f"_proj_{name}"
            proj: nn.Linear = getattr(self, attr, None)
            if proj is None or proj.in_features != D or proj.out_features != H:
                proj = nn.Linear(D, H, bias=False)
                with torch.no_grad():
                    weight = torch.zeros(H, D)
                    if D >= H:
                        for i in range(H):
                            start = int(math.floor(i * D / H))
                            end = int(math.floor((i + 1) * D / H))
                            if end <= start:
                                end = min(D, start + 1)
                            weight[i, start:end] = 1.0 / max(1, end - start)
                    else:
                        repeats = math.ceil(H / D)
                        for i in range(H):
                            weight[i, i % D] = 1.0 / repeats
                proj.weight.copy_(weight)
                proj.requires_grad_(False)
                setattr(self, attr, proj)
            proj = proj.to(device=device, dtype=dtype)
            y = proj(x)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nn.functional.layer_norm(y, (H,))
            return y

        A = _proj_to_hidden(aud_vecs, "aud")   # (B, H)
        V = _proj_to_hidden(vid_vecs, "vid")   # (B, H)

        # 4) Selector weights和主模态
        w = torch.as_tensor(weights, device=device, dtype=dtype)  # (B,3)
        main_idx = torch.argmax(w, dim=1)  # 0=T,1=A,2=V
        txt_pad_mask = (attn_mask == 0)
        ones_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

        mha = self.cross_attn_mha.to(device=device, dtype=dtype)
        fusion_proj: nn.Linear = self.cross_attn_proj.to(device=device, dtype=dtype)

        fused_concat = torch.zeros(B, L, H * 4, dtype=dtype, device=device)
        text_feat = inputs_embeds
        aud_seq = A.unsqueeze(1)
        vid_seq = V.unsqueeze(1)

        def _expand_to_text(seq: torch.Tensor, target_len: int) -> torch.Tensor:
            if seq.size(1) == target_len:
                return seq
            return seq.expand(-1, target_len, -1).clone()

        # --- Case 1: Text is main modality ---
        text_sel = (main_idx == 0)
        if torch.any(text_sel):
            txt = text_feat[text_sel]
            txt_mask_sel = txt_pad_mask[text_sel]
            aud = aud_seq[text_sel]
            vid = vid_seq[text_sel]
            main_self, _ = mha(txt, txt, txt, key_padding_mask=txt_mask_sel)
            cross_a, _ = mha(txt, aud, aud, key_padding_mask=torch.zeros_like(ones_mask[text_sel]))
            cross_v, _ = mha(txt, vid, vid, key_padding_mask=torch.zeros_like(ones_mask[text_sel]))
            fused_concat[text_sel] = torch.cat([txt, main_self, cross_a, cross_v], dim=-1)

        # --- Case 2: Audio is main modality ---
        aud_sel = (main_idx == 1)
        if torch.any(aud_sel):
            txt = text_feat[aud_sel]
            txt_mask_sel = txt_pad_mask[aud_sel]
            aud = aud_seq[aud_sel]
            vid = vid_seq[aud_sel]
            main_self, _ = mha(aud, aud, aud, key_padding_mask=torch.zeros_like(ones_mask[aud_sel]))
            cross_txt, _ = mha(aud, txt, txt, key_padding_mask=txt_mask_sel)
            cross_vid, _ = mha(aud, vid, vid, key_padding_mask=torch.zeros_like(ones_mask[aud_sel]))
            main_self_exp = _expand_to_text(main_self, txt.size(1))
            cross_txt_exp = _expand_to_text(cross_txt, txt.size(1))
            cross_vid_exp = _expand_to_text(cross_vid, txt.size(1))
            fused_concat[aud_sel] = torch.cat([txt, main_self_exp, cross_txt_exp, cross_vid_exp], dim=-1)

        # --- Case 3: Video is main modality ---
        vid_sel = (main_idx == 2)
        if torch.any(vid_sel):
            txt = text_feat[vid_sel]
            txt_mask_sel = txt_pad_mask[vid_sel]
            aud = aud_seq[vid_sel]
            vid = vid_seq[vid_sel]
            main_self, _ = mha(vid, vid, vid, key_padding_mask=torch.zeros_like(ones_mask[vid_sel]))
            cross_txt, _ = mha(vid, txt, txt, key_padding_mask=txt_mask_sel)
            cross_aud, _ = mha(vid, aud, aud, key_padding_mask=torch.zeros_like(ones_mask[vid_sel]))
            main_self_exp = _expand_to_text(main_self, txt.size(1))
            cross_txt_exp = _expand_to_text(cross_txt, txt.size(1))
            cross_aud_exp = _expand_to_text(cross_aud, txt.size(1))
            fused_concat[vid_sel] = torch.cat([txt, main_self_exp, cross_txt_exp, cross_aud_exp], dim=-1)

        fused_concat = torch.nan_to_num(fused_concat, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-5, 5)
        fused_proj = fusion_proj(fused_concat)
        inputs_embeds = ((1 - alpha) * inputs_embeds + alpha * fused_proj).clamp_(-10, 10)
        if not torch.isfinite(inputs_embeds).all():
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0)

        # 7) Prepare gen kwargs and run a one-step sanity forward
        self.model.config.pad_token_id = getattr(self.model.config, 'pad_token_id', tok.pad_token_id)
        self.model.config.eos_token_id = getattr(self.model.config, 'eos_token_id', tok.eos_token_id)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=4,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            use_cache=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )
        try:
            _ = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        except Exception as e:
            print('[CrossAttn SanityCheck] forward failed:', repr(e))
            raise

        out_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            **gen_kwargs,
        )
        return tok.batch_decode(out_ids, skip_special_tokens=True)


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
