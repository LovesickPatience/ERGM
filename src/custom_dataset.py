import os
import torch
import pickle
import json
from itertools import chain
from tqdm import tqdm
from torch.utils.data import Dataset

from selector.data_preprocess import _stack_feats_1d

class CustomDataset(Dataset):
    def __init__(self, prefix, args):
        self.args = args
        assert prefix == args.train_prefix or prefix == args.valid_prefix
        
        data_path = os.path.join(args.data_dir, f"multi_{prefix}_data.pkl")
        context_path = os.path.join(args.data_dir, f"context_label_{prefix}_data.pkl")
        
        print(f"Loading main data from {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            # Using slicing for debugging. For a full run, remove the '[:1]'
            texts, videos, audios, targets = data["txt"][:1], data["img"][:1], data["aud"][:1], data["label"][:1]

        print(f"Loading context and emotion labels from {context_path}...")
        with open(context_path, 'rb') as f:
            context_label = pickle.load(f)
            # Using slicing for debugging. For a full run, remove the '[:1]'
            contexts_data = context_label["context"][:1]
            emotion_labels_data = context_label["label"][:1]

        self.input_ids = []
        self.token_type_ids = []
        self.labels = []
        self.imgs = []
        self.auds = []
        self.contexts = []
        self.emotion_labels = []

        for i in tqdm(range(len(texts)), desc=f"Processing {prefix} data"):
            dialogue_texts = texts[i]
            dialogue_targets = targets[i]
            dialogue_contexts = contexts_data[i]
            dialogue_emotion_labels = emotion_labels_data[i]
            
            # Ensure all parts of a dialogue have the same length
            assert len(dialogue_texts) == len(dialogue_targets) == len(dialogue_contexts) == len(dialogue_emotion_labels)

            for j in range(len(dialogue_texts)):
                utterance_tokens = dialogue_texts[j]
                input_ids_list = list(chain.from_iterable(utterance_tokens))

                if len(input_ids_list) >= 1024:
                    continue

                sp1_id, sp2_id = args.sp1_id, args.sp2_id
                current_token_types = [[sp1_id] * len(ctx) if c % 2 == 0 else [sp2_id] * len(ctx) for c, ctx in enumerate(utterance_tokens)]
                current_token_types_list = list(chain.from_iterable(current_token_types))
                assert len(input_ids_list) == len(current_token_types_list)
                
                current_lm_target = dialogue_targets[j]
                current_lm_labels = current_lm_target[2:-2] + [args.eos_id] # Slice to remove special tokens

                len_gap = len(input_ids_list) - len(current_lm_labels)
                if len_gap > 0:
                    current_lm_labels = [-100] * len_gap + current_lm_labels
                elif len_gap < 0:
                    gap_to_add = abs(len_gap)
                    input_ids_list.extend([args.eos_id] * gap_to_add)
                    current_token_types_list.extend([current_token_types_list[-1]] * gap_to_add)

                assert len(input_ids_list) == len(current_lm_labels)

                self.input_ids.append(input_ids_list)
                self.token_type_ids.append(current_token_types_list)
                self.labels.append(current_lm_labels)
                self.emotion_labels.append(dialogue_emotion_labels[j])
                
                v_tmp = [videos[i][0]] * len(input_ids_list)
                a_tmp = [audios[i][0]] * len(input_ids_list)
                self.imgs.append(v_tmp)
                self.auds.append(a_tmp)
                self.contexts.append(dialogue_contexts[j])
        
        assert len(self.input_ids) == len(self.token_type_ids) == len(self.labels) == \
               len(self.imgs) == len(self.auds) == len(self.contexts) == len(self.emotion_labels)
        
        print(f"Finished processing. Total samples: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.token_type_ids[index],
            self.labels[index],
            self.imgs[index],
            self.auds[index],
            self.contexts[index],
            self.emotion_labels[index],
        )

class PadCollate():
    def __init__(self, eos_id, args):
        self.args = args
        self.eos_id = eos_id

    def pad_collate(self, batch):
        input_ids, token_type_ids, labels, imgs, auds, contexts, emotion_labels = [], [], [], [], [], [], []

        for seqs in batch:
            input_ids.append(torch.LongTensor(seqs[0]))
            token_type_ids.append(torch.LongTensor(seqs[1]))
            labels.append(torch.LongTensor(seqs[2]))
            imgs.append(seqs[3]) 
            auds.append(seqs[4])
            contexts.append(seqs[5]) 
            emotion_labels.append(seqs[6])

        # Pad the sequence-like tensors
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.eos_id)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=self.eos_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return (
            input_ids,
            token_type_ids,
            labels,
            imgs, 
            auds,
            contexts, 
            emotion_labels
        )

    def iemocap_collate(self, tokenizer, a_dim: int = 512, v_dim: int = 512):
        # Build ERGM-compatible 7-tuple for IEMOCAP PKL+JSON samples
        # Expected per-sample keys (from IEMOCAPDialoguePKLDataset):
        #  - ctx_text (string with <sp1>/<sp2> already inserted)
        #  - label_text (string)
        #  - audio_feats (Tensor or ndarray) shape (T,D) or (D,)
        #  - video_feats (Tensor or ndarray) shape (T,D) or (D,)
        #  - erc_label_text (str) or emotion_label (int)
        #  - (optional) ctx_speaker_ids / utt_bounds if you want to refine token_type_ids
        def collate(batch):
            input_ids_list: list = []
            token_type_ids_list: list = []
            labels_list: list = []
            imgs: list = []
            auds: list = []
            contexts: list = []
            emotion_labels: list = []

            # speaker tokens & ids (defaults)
            sp1_tok = getattr(self.args, 'sp1_token', '<sp1>')
            sp2_tok = getattr(self.args, 'sp2_token', '<sp2>')
            sp1_id = getattr(self.args, 'sp1_id', 0)
            sp2_id = getattr(self.args, 'sp2_id', 1)
            eos_id = getattr(self.args, 'eos_id', self.eos_id)
            max_len = getattr(self.args, 'max_len', 512)

            # Resolve token ids of speaker tokens if present in tokenizer vocab
            def _tok_id(tok):
                try:
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    # Some tokenizers return unk for custom tokens; guard it
                    if tid is None or tid < 0:
                        return None
                    return tid
                except Exception:
                    return None

            sp1_tid = _tok_id(sp1_tok)
            sp2_tid = _tok_id(sp2_tok)

            def _stack_1d(x):
                # x: (T,D) or (D,) -> (D,) float32; if (T,D), mean-pool on T
                if isinstance(x, torch.Tensor):
                    t = x
                else:
                    t = torch.as_tensor(x)
                if t.ndim == 2:
                    t = t.mean(dim=0)
                return t.to(torch.float32)

            # basic emotion map fallback
            _EMO_MAP = getattr(self.args, 'emo_map', None)
            if _EMO_MAP is None:
                _EMO_MAP = {
                    'neutral': 0, 'neu': 0, 'others': 0,
                    'hap': 1, 'joy': 1,
                    'sad': 2, 'sadness': 2,
                    'ang': 3, 'anger': 3,
                    'sur': 4,
                    'dis': 5,
                    'fea': 6,
                    'fru': 7,
                    'exc': 8,
                }

            for b in batch:
                ctx_text = b.get('ctx_text', '')
                label_text = b.get('label_text', '')
                contexts.append(b.get('contexts', b.get('ctx_text', '')))

                # --- tokenize context (ctx) ---
                enc_ctx = tokenizer(
                    ctx_text,
                    padding=False,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                    return_tensors='pt'
                )
                ids_ctx = enc_ctx.input_ids[0].to(torch.long)

                # build token_type_ids for ctx by scanning speaker tokens
                cur_sp = sp1_id
                tt_ctx = torch.empty_like(ids_ctx)
                for k, tid in enumerate(ids_ctx.tolist()):
                    if sp1_tid is not None and tid == sp1_tid:
                        cur_sp = sp1_id
                    elif sp2_tid is not None and tid == sp2_tid:
                        cur_sp = sp2_id
                    tt_ctx[k] = cur_sp

                # --- tokenize response (resp) ---
                enc_resp = tokenizer(
                    label_text,
                    padding=False,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                    return_tensors='pt'
                )
                ids_resp = enc_resp.input_ids[0].to(torch.long)

                # --- smart truncation to keep full response supervised ---
                # We prefer to trim from the LEFT of ctx so that the entire response stays
                total_len = ids_ctx.size(0) + ids_resp.size(0)
                if total_len > max_len:
                    overflow = total_len - max_len
                    if overflow < ids_ctx.size(0):
                        ids_ctx = ids_ctx[overflow:]
                        tt_ctx = tt_ctx[overflow:]
                    else:
                        # ctx fully removed; keep the last max_len tokens of resp
                        cut = overflow - ids_ctx.size(0)
                        ids_ctx = ids_ctx.new_zeros((0,), dtype=torch.long)
                        tt_ctx = ids_ctx.new_zeros((0,), dtype=torch.long)
                        ids_resp = ids_resp[-max_len:]

                # --- concat inputs; labels only supervise response ---
                input_ids_full = torch.cat([ids_ctx, ids_resp], dim=0)
                tt_full = torch.cat([tt_ctx, (
                    tt_ctx[-1:] if tt_ctx.numel() > 0 else torch.tensor([sp1_id], dtype=torch.long)).repeat(
                    ids_resp.size(0))], dim=0)

                labels_full = torch.cat([
                    torch.full((ids_ctx.size(0),), -100, dtype=torch.long),
                    ids_resp.clone()
                ], dim=0)

                input_ids_list.append(input_ids_full)
                token_type_ids_list.append(tt_full)
                labels_list.append(labels_full)

                # Audio/Video features → (D,) then repeat per-token to align with ERGM expectation
                # ---- audio ----
                af_src = b['audio_feats'] if ('audio_feats' in b and b['audio_feats'] is not None) else b.get(
                    'audio_feat', None)
                if af_src is None:
                    af = torch.zeros(a_dim, dtype=torch.float32)
                else:
                    af = _stack_1d(af_src)

                # ---- video ----
                vf_src = b['video_feats'] if ('video_feats' in b and b['video_feats'] is not None) else b.get(
                    'video_feat', None)
                if vf_src is None:
                    vf = torch.zeros(v_dim, dtype=torch.float32)
                else:
                    vf = _stack_1d(vf_src)
                seq_len = input_ids_full.size(0)
                imgs.append([vf.clone() for _ in range(seq_len)])
                auds.append([af.clone() for _ in range(seq_len)])

                emo_label_val = b.get('emotion_label', None)
                if emo_label_val is None:
                    emo_label_val = _EMO_MAP.get(str(b.get('erc_label_text', 'neu')).lower(), 0)
                emotion_labels.append(int(emo_label_val))

            # Pad sequences across batch
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=eos_id)
            token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids_list, batch_first=True, padding_value=sp1_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

            return (
                input_ids,
                token_type_ids,
                labels,
                imgs,
                auds,
                contexts,
                emotion_labels,
            )
        return collate