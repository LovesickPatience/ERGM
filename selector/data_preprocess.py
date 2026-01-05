import pickle
import json
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", ) as f:
        return json.load(f)

# utilities
def _to_tensor_feat(x):
    # x 可以是 np.ndarray 或 torch.Tensor，形状可能是 (T, D) 或 (D,)
    t = torch.as_tensor(x)
    if t.ndim == 2:           # (T, D) -> 时间平均
        t = t.mean(dim=0)
    elif t.ndim == 1:         # (D,)
        pass
    else:
        raise ValueError(f"Unexpected feature ndim {t.ndim}")
    return t  # (D,)

def _pad_trunc_dim(vec: torch.Tensor, target_D: int) -> torch.Tensor:
    D = vec.shape[-1]
    if D == target_D:
        return vec
    if D > target_D:
        return vec[..., :target_D]
    # pad zeros to the right
    pad = torch.zeros(target_D - D, dtype=vec.dtype, device=vec.device)
    return torch.cat([vec, pad], dim=-1)

def _stack_feats_1d(feat_list: list, target_D: int | None = None) -> torch.Tensor:
    # 将一批 (D_i,) 统一成 (B, target_D)
    cleaned = []
    for f in feat_list:
        if f is None:
            cleaned.append(None)
        else:
            cleaned.append(_to_tensor_feat(f))
    # 自动决定目标维度（也可改为固定常量，比如 512）
    if target_D is None:
        ds = [int(t.shape[-1]) for t in cleaned if t is not None]
        target_D = max(ds) if len(ds) > 0 else 0
    outs = []
    for t in cleaned:
        if t is None:
            outs.append(torch.zeros(target_D))
        else:
            outs.append(_pad_trunc_dim(t, target_D))
    return torch.stack(outs, dim=0)  # (B, target_D)


def _iter_dialogues(diadict: Dict[str, Any], session_filter: Optional[str]):
    for dlg_id, utts in diadict.items():
        if session_filter and not dlg_id.startswith(session_filter):
            continue
        yield dlg_id, utts


def _rolling_pairs(utterances: List[Dict[str, Any]]):
    """Yield (context_utts, label_utt) using rolling concatenation ut0→ut1, (ut0+ut1)→ut2, ..."""
    ctx = []
    for i, u in enumerate(utterances):
        if i == 0:
            ctx.append(u)
            continue
        yield list(ctx), u
        ctx.append(u)


def _concat_text_and_boundaries(
        utts: List[Dict[str, Any]],
        bos: Optional[str] = "<s>",
        eos: str = "</s>",
        speaker_tokens: Optional[List[str]] = None,
        insert_space_after_sp: bool = True,
    ) -> Tuple[str, List[Tuple[str, int, int]], List[Tuple[int, int]]]:
    """Concatenate utterances with a *single* BOS at the start (if provided) and a *single* EOS at the end.
    Each utterance is prefixed with its speaker token (e.g., <sp1>/<sp2>) if provided.

    Returns:
        text: full concatenated string, e.g. "<s> <sp1> u1 ... <sp2> u2 ... </s>"
        utt_bounds: list of (utterance_id, start_char, end_char) covering [speaker_token + space + utterance_text]
        token_offsets: naive whitespace token offsets for the *original utterance text* region in the full string
    """
    # Start with a single BOS if provided
    text = bos if bos is not None else ""
    utt_bounds: List[Tuple[str, int, int]] = []
    token_offsets: List[Tuple[int, int]] = []

    cursor = len(text)  # current length of text

    for i, u in enumerate(utts):
        utext = u.get("text", "")
        sp_tok = speaker_tokens[i] if (speaker_tokens is not None and i < len(speaker_tokens)) else ""
        spacer = (" " if (insert_space_after_sp and sp_tok) else "")

        # Leading space between utterances (except before the first one after BOS)
        lead = "" if i == 0 else " "

        # Core piece for this utterance: <spX> + optional space + raw utterance text
        piece_core = f"{sp_tok}{spacer}{utext}"
        piece = f"{lead}{piece_core}"

        # Bounds for this utterance within the concatenated string (excluding BOS, including speaker token)
        start = cursor + len(lead)
        end = start + len(piece_core)
        utt_bounds.append((u.get("utterance_id", "unk"), start, end))

        # Append to full text and move cursor
        text += piece
        cursor = len(text)

        # Token offsets inside the *raw utterance text* region
        utext_start = start + len(sp_tok) + len(spacer)
        idx = utext_start
        for tok in utext.split():
            s = text.find(tok, idx)
            if s == -1:
                continue
            e = s + len(tok)
            token_offsets.append((s, e))
            idx = e

    # Append a single EOS at the very end
    if eos is not None:
        text += eos
    return text, utt_bounds, token_offsets


    # ---------------
    # Collate Functions
    # ---------------
def make_selector_collate_fn(text_encoder: Any, task: str, a_dim: int = 64, v_dim: int = 64):
    """
    Collate function for IEMOCAPDialoguePKLDataset.
    """
    def collate(batch: List[Dict[str, Any]]):
        # batch is a list of dicts from IEMOCAPDialoguePKLDataset
        # Each dict: 'ctx_text', 'label_text', 'erc_label_text', 'audio_feats', 'video_feats', 'handcrafted' (opt)
        ctx_texts = [ex['ctx_text'] for ex in batch]
        tf = text_encoder.encode(ctx_texts)  # (B, D)
        # audio_feats, video_feats: (B, T, D) → mean-pool over T to (B, D)
        # audio_feats_list / video_feats_list 是 batch 内每个样本的原始特征
        audio_feats_list = [b.get("audio_feat") or b.get("audio_feats") for b in batch]
        video_feats_list = [b.get("video_feat") or b.get("video_feats") for b in batch]

        # 统一成 (B, D_a) / (B, D_v)
        af = _stack_feats_1d(audio_feats_list, target_D=a_dim).to(torch.float32)  # 传入固定 target_D
        vf = _stack_feats_1d(video_feats_list, target_D=v_dim).to(torch.float32)
        hc = None
        if 'handcrafted' in batch[0] and batch[0]['handcrafted'] is not None:
            hc = torch.stack([
                torch.tensor(ex['handcrafted'], dtype=torch.float32)
                for ex in batch
                ])
        texts = ctx_texts
        if task == 'EGC':
            targets = [ex['label_text'] for ex in batch]
            labels = [0 for _ in batch]
        else:
            targets = ["" for _ in batch]
            # 'erc_label_text' is the class name, but we want mapped int label
            labels = [int(ex['erc_label_text']) if ex['erc_label_text'] is not None else 0 for ex in batch]
        return tf, af, vf, hc, texts, targets, labels
    return collate

def feature_dataset_collate_fn(batch: List[Dict[str, Any]]):
    # For FeatureDataset, stack features and pass through text/target/label
    tf = torch.stack([torch.tensor(ex['text_feat'], dtype=torch.float32) for ex in batch])
    af = torch.stack([torch.tensor(ex['audio_feat'], dtype=torch.float32) for ex in batch])
    vf = torch.stack([torch.tensor(ex['video_feat'], dtype=torch.float32) for ex in batch])
    hc = None
    if 'handcrafted' in batch[0] and batch[0]['handcrafted'] is not None:
        hc = torch.stack([torch.tensor(ex['handcrafted'], dtype=torch.float32) for ex in batch])
    texts = [ex.get('text', "") for ex in batch]
    targets = [ex.get('target_text', "") for ex in batch]
    labels = [ex.get('label', 0) for ex in batch]
    return tf, af, vf, hc, texts, targets, labels


class FeatureDataset(Dataset):
    def __init__(self, pkl_paths: List[str], keys: Dict[str, str], task: str):
        self.samples = []
        for p in pkl_paths:
            with open(p, 'rb') as f:
                part = pickle.load(f)
            if isinstance(part, list):
                self.samples.extend(part)
            elif isinstance(part, dict) and 'samples' in part:
                self.samples.extend(part['samples'])
            else:
                raise ValueError(f"Unsupported PKL format: {p}")
        self.keys = keys
        self.task = task

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        def get(name, default=None):
            k = self.keys.get(name, name)
            return s.get(k, default)

        text = get('text') or get('input_text')
        target_text = get('target_text')
        label = get('label')
        tf = np.asarray(get('text_feat'), dtype=np.float32)
        af = np.asarray(get('audio_feat'), dtype=np.float32)
        vf = np.asarray(get('video_feat'), dtype=np.float32)
        hc = get('handcrafted')
        if hc is not None:
            hc = np.asarray(hc, dtype=np.float32)
        return {
            'text': text,
            'target_text': target_text,
            'label': label,
            'text_feat': tf,
            'audio_feat': af,
            'video_feat': vf,
            'handcrafted': hc,
        }

class IEMOCAPDialoguePKLDataset(Dataset):
    """Use pre-extracted A/V features from a PKL, still build rolling contexts from JSON.
    PKL structure:
      data = {
        'audio': [train_dict, test_dict, val_dict],
        'video': [train_dict, test_dict, val_dict]
      }
    Each dict maps utterance_id -> np.ndarray (T, D) or (D,) feature; we will accept either and time-pool to (D,).
    We will gather **only context utterances** A/V features (before label utterance) and concatenate along time axis → (Lc, D).

    We also extract per-utterance speakers from the JSON ('speaker' key) and build speaker id mapping per dialog. The sample now includes: 'ctx_speakers', 'ctx_speaker_ids', 'label_speaker', 'label_speaker_id', and 'speaker_map'.
    """
    SPLIT_INDEX = {"train":0, "test":2, "val":1}

    def __init__(self, json_path: str, pkl_path: str, split: str = "train", session_filter: Optional[str] = None,
                 bos: str = None, eos: str = "</s>", sp1_token: str = "<sp1>", sp2_token: str = "<sp2>"):
        super().__init__()
        self.meta = _load_json(json_path)
        with open(pkl_path, "rb") as f:
            blob = pickle.load(f)
        si = self.SPLIT_INDEX[split]
        self.audio_bank: Dict[str, Any] = blob["audio"][si]
        self.video_bank: Dict[str, Any] = blob["video"][si]
        # Restrict to utterances that belong to the requested split (keys come from the pre-split A/V banks)
        # Otherwise train/val/test will see the same text dialogues.
        allowed_utt_ids = set(self.audio_bank.keys()) | set(self.video_bank.keys())
        self.samples: List[Dict[str, Any]] = []
        self.bos = bos
        self.eos = eos
        self.sp1_token = sp1_token
        self.sp2_token = sp2_token

        for dlg_id, utts in _iter_dialogues(self.meta, session_filter):
            # Build a per-dialog speaker-id map (first seen -> 0, second -> 1, ...)
            sp_map: Dict[str, int] = {}
            def _sp_id_and_token(name: str) -> Tuple[int, str]:
                if name not in sp_map:
                    idx = len(sp_map)
                    if idx == 0:
                        sp_map[name] = (self.sp1_token, 0)
                    elif idx == 1:
                        sp_map[name] = (self.sp2_token, 1)
                    else:
                        sp_map[name] = (f"<sp{idx+1}>", idx)
                tok, sid = sp_map[name]
                return sid, tok

            # ensure utterances are in chronological order (if time key exists)
            utts = [u for u in utts if u.get("utterance_id") in allowed_utt_ids]
            if len(utts) < 2:
                continue
            utts = sorted(utts, key=lambda u: float(u.get("start_time", 0.0)))
            for ctx_utts, label_utt in _rolling_pairs(utts):
                # speakers for context and label (build tokens/ids first)
                ctx_speakers = [u.get("speaker", "UNK") for u in ctx_utts]
                ctx_sp_ids, ctx_sp_toks = zip(*[_sp_id_and_token(s) for s in ctx_speakers]) if ctx_speakers else ([], [])

                # now concatenate text with speaker tokens
                ctx_text, utt_bounds, token_offsets = _concat_text_and_boundaries(
                    ctx_utts, self.bos, self.eos, speaker_tokens=list(ctx_sp_toks)
                )

                label_speaker = label_utt.get("speaker", "UNK")
                label_speaker_id, label_speaker_token = _sp_id_and_token(label_speaker)
                # collect context utterance ids
                ctx_ids = [u.get("utterance_id") for u in ctx_utts]
                # A/V features list
                a_feats, v_feats = [], []
                for uid in ctx_ids:
                    if uid in self.audio_bank:
                        a = self.audio_bank[uid]
                        a = torch.tensor(a)
                        if a.ndim == 1:
                            a = a.unsqueeze(0)
                        a = a.mean(dim=0)  # time-pool to (D,)
                        a_feats.append(a)
                    if uid in self.video_bank:
                        v = self.video_bank[uid]
                        v = torch.tensor(v)
                        if v.ndim == 1:
                            v = v.unsqueeze(0)
                        v = v.mean(dim=0)
                        v_feats.append(v)
                # stack to (Lc, D)
                a_stack = torch.stack(a_feats, dim=0) if a_feats else torch.zeros(1, 64)
                v_stack = torch.stack(v_feats, dim=0) if v_feats else torch.zeros(1, 64)

                self.samples.append({
                    "dialogue_id": dlg_id,
                    "ctx_text": ctx_text,
                    "utt_bounds": utt_bounds,
                    "token_offsets": token_offsets,

                    # 说话人信息（上下文 & label）
                    "ctx_speakers": list(ctx_speakers),
                    "ctx_speaker_ids": list(ctx_sp_ids) if ctx_speakers else [],
                    "ctx_speaker_tokens": list(ctx_sp_toks) if ctx_speakers else [],
                    "label_speaker": label_speaker,
                    "label_speaker_id": label_speaker_id,
                    "label_speaker_token": label_speaker_token,
                    "speaker_map": {k: (v[0], v[1]) for k, v in sp_map.items()},

                    # 多模态特征（上下文拼接为 (Lc, D)）
                    "audio_feats": a_stack,
                    "video_feats": v_stack,

                    # 标签文本/情感
                    "erc_label_text": label_utt.get("emo", "neu"),
                    "label_text": label_utt.get("text", ""),

                    # 情绪原因（如有）
                    "cause_true_utt_ids": label_utt.get("hist_utt_id", []) or label_utt.get("emtion_cause_utterance_id",
                                                                                            []),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
