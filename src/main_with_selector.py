import sys
import argparse
import time

import random
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    GPT2Tokenizer,
    get_polynomial_decay_schedule_with_warmup,
)

from src.model.model_with_fusion import *
from custom_dataset import *
from eval.evaluate import Evaluator
from selector.selector_models import SelectorConfig, build_selector
from selector.data_preprocess import IEMOCAPDialoguePKLDataset


def print_custom(context, ref, sentence):
    """Formats the context, reference, and generated sentence for printing."""
    res = ""
    res += f"Context: {context}\n"
    res += f"GPT-2: {sentence}\n"
    res += f"Ref: {ref}\n"
    res += "---------------------------------------------------------------\n"
    return res


def _build_states(text_feat: torch.Tensor, audio_feat: torch.Tensor, video_feat: torch.Tensor,
                  hc_feat: torch.Tensor = None):
    """
    把多模态的句子级特征拼成 selector 的输入（和你训练 selector 时保持一致）。
    假设 text/audio/video 都是 (B, D_t/a/v) 的 pooled embedding。
    """
    xs = []
    for feat in (text_feat, audio_feat, video_feat, hc_feat):
        if feat is not None:
            xs.append(feat)
    if len(xs) == 0:
        raise ValueError("No modality feature is available to build selector states.")
    return torch.cat(xs, dim=-1)  # (B, D_total)


def _apply_weights(weights: torch.Tensor, text_feat: torch.Tensor, audio_feat: torch.Tensor, video_feat: torch.Tensor):
    """
    把 (B,K) 的权重作用到 3 个模态特征上，返回融合后的向量：
      fused = w_T*T + w_A*A + w_V*V
    若你的 ERGM 原始融合是 concat，这里可以返回[加权特征，再 concat 原特征] 或直接替换。
    """
    # 简单的线性融合
    comps = torch.stack([text_feat, audio_feat, video_feat], dim=1)  # (B, 3, D)
    fused = (weights.unsqueeze(-1) * comps).sum(dim=1)  # (B, D)
    return fused


class Manager:
    def __init__(self, args):
        self.args = args
        self.selector = None
        self.selector_dims = {}

        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")

        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_type)
        special_tokens = {
            "bos_token": self.args.bos_token,
            "additional_special_tokens": [self.args.sp1_token, self.args.sp2_token, "<img>", "<aud>"],
        }
        self.args.eos_token = self.tokenizer.eos_token
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        self.args.final_vocab_size = len(self.tokenizer)

        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        self.args.img_id = self.tokenizer.convert_tokens_to_ids("<img>")
        self.args.aud_id = self.tokenizer.convert_tokens_to_ids("<aud>")

        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.config.img_id = self.args.img_id
        self.model.config.aud_id = self.args.aud_id
        self.model.resize_token_embeddings(self.args.final_vocab_size)
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)

        if self.args.mode in ["train", "infer"]:
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
                val_pkl_list = [p.strip() for p in self.args.val_pkls.split(',')] \
                    if isinstance(self.args.val_pkls, str) else self.args.val_pkls
                test_pkl_list = [p.strip() for p in self.args.val_pkls.split(',')] \
                    if isinstance(self.args.val_pkls, str) else self.args.val_pkls

                train_set = IEMOCAPDialoguePKLDataset(pkl_path=train_pkl_list[0], json_path=self.args.iemocap_text_json,
                                                      split='train')
                valid_set = IEMOCAPDialoguePKLDataset(pkl_path=val_pkl_list[0], json_path=self.args.iemocap_text_json,
                                                      split='val')
                test_set = IEMOCAPDialoguePKLDataset(pkl_path=train_pkl_list[0], json_path=self.args.iemocap_text_json,
                                                     split='test')

                collate = ppd.iemocap_collate(self.tokenizer)
                self.train_loader = DataLoader(
                    train_set, collate_fn=collate, shuffle=True,
                    batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                )
                self.valid_loader = DataLoader(
                    valid_set, collate_fn=collate, shuffle=False,
                    batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                )
                self.test_loader = DataLoader(
                    test_set, collate_fn=collate, shuffle=False,
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

            # Build selector after seeing one batch to infer feature dims
            self._init_selector()

            print("Loading the optimizer...")
            params = list(self.model.parameters()) + list(self.selector.parameters())
            self.optim = torch.optim.AdamW(params, lr=self.args.lr)

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
                sd = ckpt['model_state_dict']

                if "transformer.wte.weight" in sd:
                    old_wte = sd["transformer.wte.weight"]
                    new_wte = self.model.transformer.wte.weight
                    if old_wte.size(0) != new_wte.size(0):
                        print(f"[vocab expand] ckpt_vocab={old_wte.size(0)}, current_vocab={new_wte.size(0)}")
                        with torch.no_grad():
                            num = min(old_wte.size(0), new_wte.size(0))
                            new_wte[:num] = old_wte[:num]
                        sd.pop("transformer.wte.weight")

                if "lm_head.weight" in sd:
                    old_lm = sd["lm_head.weight"]
                    new_lm = self.model.lm_head.weight
                    if old_lm.size(0) != new_lm.size(0):
                        with torch.no_grad():
                            num = min(old_lm.size(0), new_lm.size(0))
                            new_lm[:num] = old_lm[:num]
                        sd.pop("lm_head.weight")

                self.model.load_state_dict(sd, strict=False)  # Use strict=False to handle the new emotion_head
                if self.selector is not None and "selector_state_dict" in ckpt:
                    self.selector.load_state_dict(ckpt["selector_state_dict"], strict=False)

                if self.args.mode == "train":
                    print(f"Training will resume from checkpoint: {self.args.ckpt_name}.ckpt")
                    # self.optim.load_state_dict(ckpt["optim_state_dict"])
                    # self.sched.load_state_dict(ckpt["sched_state_dict"])
                    self.best_ppl = ckpt.get('ppl', sys.float_info.max)  # Load best ppl if available
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
            if self.selector is not None:
                self.selector.train()
            print(f"-" * 35 + f"Epoch: {epoch}" + "-" * 35)

            train_total_losses = []
            train_lm_losses = []  # For PPL calculation
            train_correct_emotions = 0
            train_total_emotions = 0

            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch

                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device),  # Ensure emotion labels are tensors
                )

                gated_weight, main_idx = self._selector_forward(input_ids, imgs, auds)

                outputs = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids,
                    labels=lm_labels, emotion_labels=emotion_labels,
                    imgs=imgs,
                    auds=auds,
                    gated_weights=gated_weight,
                    main_idx=main_idx,
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

            print(
                f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.4f} | Train Emotion Acc: {train_acc:.2f}%")
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.writer.add_scalar("PPL/train", train_ppl, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)

            self.last_epoch += 1
            valid_loss, valid_ppl, valid_acc = self.validation()

            if valid_ppl < self.best_ppl:
                self.best_ppl = valid_ppl
                state_dict = {
                    "model_state_dict": self.model.state_dict(),
                    "selector_state_dict": self.selector.state_dict(),
                    "optim_state_dict": self.optim.state_dict(),
                    "sched_state_dict": self.sched.state_dict(),
                    "ppl": self.best_ppl,
                    "epoch": self.last_epoch,
                }
                now = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(
                    self.args.ckpt_dir,
                    f"best_ckpt_epoch={epoch}_valid_ppl={self.best_ppl:.4f}_val_emo_acc={valid_acc:.2f}%_{now}_text+imgs+auds.ckpt",
                )
                torch.save(state_dict, save_path)
                print("*" * 10 + " Current best checkpoint is saved. " + "*" * 10)
                print(save_path)

            print(f"Best valid PPL: {self.best_ppl:.4f}")
            print(
                f"Current valid loss: {valid_loss:.4f} | Current valid PPL: {valid_ppl:.4f} | Current valid Emotion Acc: {valid_acc:.2f}%")
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            self.writer.add_scalar("PPL/valid", valid_ppl, epoch)
            self.writer.add_scalar("Accuracy/valid", valid_acc, epoch)

        print("Training finished!")

    def validation(self):
        print("Validation processing...")
        self.model.eval()
        if self.selector is not None:
            self.selector.eval()

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

                gate, main_idx = self._selector_forward(input_ids, imgs, auds)

                outputs = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids,
                    labels=lm_labels, emotion_labels=emotion_labels,
                    imgs=imgs,
                    auds=auds,
                    gated_weights=gate,
                    main_idx=main_idx,
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

    def _pool_modal_feat(self, x):
        """
        Robust pooling utility for modal features.

        Accepts:
          - Tensor: (B, D) or (B, T, D) or (T, D) or (D,)
          - list/tuple length B where each item is (T, D) or (D,) or list of frame tensors

        Returns:
          Tensor of shape (B, D) after temporal mean pooling.
          Assumes a consistent feature dimensionality across samples (e.g., D=64).
        """
        if x is None:
            return None

        # Helper: pool a single sample to 1D vector
        def _pool_one(item):
            # item 可能是 tensor / list[tensor or array] / array / list[list[...]]
            if torch.is_tensor(item):
                t = item.float()
                if t.dim() == 2:  # (T, D)
                    return t.mean(dim=0)
                if t.dim() == 1:  # (D,)
                    return t
                if t.dim() == 3:  # (S, T, D) => 均值
                    return t.mean(dim=(0, 1))
                # 其他形状，尽量压成 1D
                return t.reshape(-1)

            if isinstance(item, (list, tuple)):
                elems = []
                for e in item:
                    if e is None:
                        continue
                    if torch.is_tensor(e):
                        elems.append(e.float().reshape(-1))
                    else:
                        try:
                            te = torch.as_tensor(e, dtype=torch.float32).reshape(-1)
                            elems.append(te)
                        except Exception:
                            continue
                if len(elems) == 0:
                    return None
                # 尝试直接堆叠，如果形状一致就是 (T, D)
                try:
                    M = torch.stack(elems, dim=0)
                    if M.dim() == 2:
                        return M.mean(dim=0)
                    elif M.dim() == 3:
                        return M.mean(dim=(0, 1))
                    else:
                        return M.reshape(-1)
                except RuntimeError:
                    # 帧向量长度不一致：pad/trunc 到最大 D 再均值
                    D = max(e.shape[0] for e in elems)
                    padded = []
                    for e in elems:
                        if e.shape[0] < D:
                            pad = torch.zeros(D - e.shape[0], dtype=e.dtype, device=e.device)
                            padded.append(torch.cat([e, pad], dim=0))
                        else:
                            padded.append(e[:D])
                    M = torch.stack(padded, dim=0)  # (T, D)
                    return M.mean(dim=0)

            # 其他可转 tensor 的类型
            try:
                t = torch.as_tensor(item, dtype=torch.float32)
                if t.dim() == 2:
                    return t.mean(dim=0)
                if t.dim() == 1:
                    return t
                return t.reshape(-1)
            except Exception:
                return None

        # 已是 Tensor 的 batch
        if torch.is_tensor(x):
            x = x.float()
            if x.dim() == 3:  # (B, T, D) -> (B, D)
                return x.mean(dim=1)
            if x.dim() == 2:  # (B, D)
                return x
            if x.dim() == 1:  # (D,) -> (1, D)
                return x.unsqueeze(0)
            if x.dim() >= 2:  # 尝试压平
                B = x.size(0)
                return x.reshape(B, -1).float()
            return None

        # list/tuple 的 batch：逐样本池化 -> 对齐维度 -> stack
        if isinstance(x, (list, tuple)) and len(x) > 0:
            per_sample = []
            for item in x:
                v = _pool_one(item)
                if v is not None:
                    per_sample.append(v)

            if len(per_sample) == 0:
                return None

            # 统一到同一 D（你现在已统一为 64，但这里仍留有保护）
            D = per_sample[0].shape[0]
            aligned = []
            for v in per_sample:
                if v.shape[0] == D:
                    aligned.append(v)
                elif v.shape[0] < D:
                    pad = torch.zeros(D - v.shape[0], dtype=v.dtype)
                    aligned.append(torch.cat([v, pad], dim=0))
                else:
                    aligned.append(v[:D])

            return torch.stack(aligned, dim=0)  # (B, D)

        return None

    def _init_selector(self):
        """Instantiate GatedTriTowerSelector with inferred feature dims."""
        if self.selector is not None:
            return

        loader = getattr(self, "train_loader", None)
        if loader is None or len(loader) == 0:
            loader = getattr(self, "valid_loader", None)
        if loader is None or len(loader) == 0:
            raise RuntimeError("Cannot initialize selector without a non-empty dataloader.")

        sample_batch = next(iter(loader))
        input_ids, _, _, imgs, auds, _, _ = sample_batch

        with torch.no_grad():
            text_feat = self._pool_text_feat(input_ids.to(self.args.device))
        audio_feat = self._pool_modal_feat(auds)
        video_feat = self._pool_modal_feat(imgs)

        audio_dim = audio_feat.shape[-1] if audio_feat is not None else 0
        video_dim = video_feat.shape[-1] if video_feat is not None else 0
        text_dim = text_feat.shape[-1] if text_feat is not None else 0

        self.selector_dims = {"text": text_dim, "audio": audio_dim, "video": video_dim}
        input_dim = text_dim + audio_dim + video_dim
        if input_dim == 0:
            raise ValueError("Selector input_dim is 0; check modality features.")

        cfg = SelectorConfig(
            input_dim=input_dim,
            hidden_dim=self.args.selector_hidden_dim,
            num_layers=self.args.selector_num_layers,
            dropout=self.args.selector_dropout,
            mode="continuous",
            num_actions=3,
            num_modalities=3,
        )
        self.selector = build_selector("GatedTriTower", cfg).to(self.args.device)

    def _apply_gate_to_modality(self, modal_inputs, weights):
        """Apply scalar gate weights (B,) to a list/sequence of modal features."""
        if modal_inputs is None:
            return None
        gated = []
        for weight, sample in zip(weights, modal_inputs):
            w = weight.to(self.args.device)
            if torch.is_tensor(sample):
                t = sample.to(self.args.device, dtype=torch.float32)
                gated.append(t * w)
            elif isinstance(sample, (list, tuple)):
                gated_sample = []
                for elem in sample:
                    t = elem if torch.is_tensor(elem) else torch.as_tensor(elem)
                    t = t.to(self.args.device, dtype=torch.float32)
                    gated_sample.append(t * w)
                gated.append(gated_sample)
            else:
                t = torch.as_tensor(sample, device=self.args.device, dtype=torch.float32)
                gated.append(t * w)
        return gated

    def _selector_forward(self, input_ids, imgs, auds):
        """Run selector获取 gate，顺便返回主模态索引 main_idx（argmax）。"""
        if self.selector is None:
            return None, None

        text_feat = self._pool_text_feat(input_ids)  # (B, E)
        audio_feat = self._pool_modal_feat(auds)
        video_feat = self._pool_modal_feat(imgs)

        states = _build_states(text_feat.to(self.args.device), audio_feat.to(self.args.device) if audio_feat is not None else None,
                               video_feat.to(self.args.device) if video_feat is not None else None)
        sel_out = self.selector(states)
        gate = sel_out.get("gate")
        if gate is None:
            gate = torch.softmax(sel_out["logits"], dim=-1)

        main_idx = torch.argmax(gate, dim=-1)  # (B,)
        return gate, main_idx

    def test(self):
        print("Test processing: Collecting generated texts and references...")
        self.model.eval()
        if self.selector is not None:
            self.selector.eval()
        self.fix_seed(self.args.seed)

        all_hypotheses = []
        all_references = []
        all_true_labels = []
        all_losses = []  # For overall test PPL

        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch

                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device),
                )

                for i in range(input_ids.size(0)):
                    current_input = input_ids[i].unsqueeze(0)
                    current_token_types = token_type_ids[i].unsqueeze(0)

                    input_len = (current_input != self.args.eos_id).sum().item()

                    output_ids = self.nucleus_sampling(current_input[:, :input_len], current_token_types[:, :input_len],
                                                       input_len)
                    hypothesis_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    all_hypotheses.append(hypothesis_text)

                    ref_ids = lm_labels[i][lm_labels[i] != -100]  # Filter out padding
                    reference_text = self.tokenizer.decode(ref_ids, skip_special_tokens=True)
                    all_references.append(reference_text)

                    all_true_labels.append(emotion_labels[i].item())

                gate, main_idx = self._selector_forward(input_ids, imgs, auds)

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    labels=lm_labels,
                    imgs=imgs,
                    auds=auds,
                    gated_weights=gate,
                    main_idx=main_idx,
                )
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = lm_labels[..., 1:].contiguous()
                loss_fct_lm = nn.CrossEntropyLoss()
                lm_loss = loss_fct_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                all_losses.append(lm_loss.item())

        return all_hypotheses, all_references, all_true_labels, all_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="The random seed.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"],
                        help="The running mode: train or infer.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="The parent directory where data files are stored.")
    parser.add_argument("--train_prefix", type=str, default="train", help="The prefix of the train data files' name.")
    parser.add_argument("--valid_prefix", type=str, default="valid",
                        help="The prefix of the validation data files' name.")
    parser.add_argument("--model_type", type=str, default="gpt2", help="The model type of GPT-2.")
    parser.add_argument("--bos_token", type=str, default="<bos>", help="The BOS token.")
    parser.add_argument("--sp1_token", type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument("--sp2_token", type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument("--gpu", type=str, default="0", help="The index of GPU to use.")
    parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="The ratio of warmup steps to the total training steps.")
    parser.add_argument("--selector_hidden_dim", type=int, default=768, help="Hidden size for GatedTriTowerSelector.")
    parser.add_argument("--selector_num_layers", type=int, default=2, help="MLP depth for selector backbone.")
    parser.add_argument("--selector_dropout", type=float, default=0.1, help="Dropout for selector backbone.")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument("--num_epochs", type=int, default=5, help="The number of total epochs.")
    parser.add_argument("--max_len", type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument("--max_turns", type=int, default=10,
                        help="The maximum number of dialogue histories to include.")
    parser.add_argument("--top_p", type=float, default=0.95, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument("--ckpt_dir", type=str, default="/root/autodl-tmp/ERGM-main/save_model",
                        help="The directory name for saved checkpoints.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="The directory name for outputs.")
    parser.add_argument("--ckpt_name", type=str, default=None,
                        help="The name of the trained checkpoint (without extension).")
    parser.add_argument("--dataset", type=str, default="ERGM", choices=["ERGM", "IEMOCAP"],
                        help="Choose data pipeline. IEMOCAP uses PKL+JSON via IEMOCAPDialoguePKLDataset.")
    parser.add_argument("--train_pkls", type=str, default=None,
                        help="(IEMOCAP) path to train PKL or a comma-separated list of PKLs")
    parser.add_argument("--val_pkls", type=str, default=None,
                        help="(IEMOCAP) path to val PKL or a comma-separated list of PKLs")
    parser.add_argument("--iemocap_text_json", type=str, default=None,
                        help="(IEMOCAP) path to JSON holding raw text/dialogue info to be encoded")

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

        evaluator = Evaluator(device=manager.args.device,
                              bert_model_path='/root/autodl-tmp/ERGM-main/tools/models/roberta-large',
                              baseline_path="/root/autodl-tmp/ERGM-main/tools/models/roberta-large/roberta-large.tsv",
                              num_layers=24,
                              rescale_with_baseline=True,)

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
