import sys
import argparse
import os
import pickle
import time
import json

import random
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    GPT2Tokenizer,
    get_polynomial_decay_schedule_with_warmup,
)

from src.model.model_with_fusion import *
from src.custom_dataset import *
from eval.evaluate import Evaluator
from selector.selector_models import SelectorConfig, build_selector
from selector.data_preprocess import IEMOCAPDialoguePKLDataset, MELDDialoguePKLDataset


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
        # 用 <bos> 作为 padding id：IEMOCAP 上下文里从不出现该 token，
        # 因此 attention_mask 可以直接由 (input_ids != pad_id) 得到，且不会误伤真正的 </s>
        self.args.pad_id = self.args.bos_id
        # RaMRA 用融合模块注入模态，不需要 <img>/<aud> 两个占位 token（它们只是随机 embedding 噪声）
        self.args.emit_placeholders = False

        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.config.img_id = self.args.img_id
        self.model.config.aud_id = self.args.aud_id
        # 辅助模态权重的下限（方案 A：辅助权重在辅助集合内归一化 + 地板，避免被 α 清零）
        self.model.config.aux_floor = float(getattr(self.args, 'aux_floor', 0.05))
        self.model.resize_token_embeddings(self.args.final_vocab_size)
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)

        if self.args.mode in ["train", "infer"]:
            self.best_ppl = sys.float_info.max
            self.last_epoch = 0

            print("Loading train & valid data...")
            ppd = PadCollate(eos_id=self.args.eos_id, args=self.args)
            if getattr(self.args, 'dataset', 'ERGM') == 'IEMOCAP':
                # IEMOCAP: audio/video from PKL, text from JSON; encode in collate
                if args.mode == 'train':
                    assert self.args.train_pkls and self.args.val_pkls and self.args.iemocap_text_json, \
                    "For IEMOCAP, please set --train_pkls, --val_pkls and --iemocap_text_json"

                if args.mode == 'train':
                    # Support comma-separated multiple PKLs
                    train_pkl_list = [p.strip() for p in self.args.train_pkls.split(',')] \
                        if isinstance(self.args.train_pkls, str) else self.args.train_pkls
                    val_pkl_list = [p.strip() for p in self.args.val_pkls.split(',')] \
                        if isinstance(self.args.val_pkls, str) else self.args.val_pkls

                    train_set = IEMOCAPDialoguePKLDataset(pkl_path=train_pkl_list[0],
                                                          json_path=self.args.iemocap_text_json, split='train',
                                                          max_turns=self.args.max_turns)
                    valid_set = IEMOCAPDialoguePKLDataset(pkl_path=val_pkl_list[0],
                                                          json_path=self.args.iemocap_text_json, split='val',
                                                          max_turns=self.args.max_turns)
                elif args.mode == 'infer':
                    val_pkl_list = [p.strip() for p in self.args.val_pkls.split(',')] \
                        if isinstance(self.args.val_pkls, str) else self.args.val_pkls
                    test_pkl_list = [p.strip() for p in self.args.val_pkls.split(',')] \
                        if isinstance(self.args.val_pkls, str) else self.args.val_pkls

                    valid_set = IEMOCAPDialoguePKLDataset(pkl_path=val_pkl_list[0],
                                                          json_path=self.args.iemocap_text_json, split='val',
                                                          max_turns=self.args.max_turns)
                    test_set = IEMOCAPDialoguePKLDataset(pkl_path=test_pkl_list[0],
                                                         json_path=self.args.iemocap_text_json, split='test',
                                                         max_turns=self.args.max_turns)

                collate = ppd.iemocap_collate(self.tokenizer)
                if args.mode == 'train':
                    self.train_loader = DataLoader(
                        train_set, collate_fn=collate, shuffle=True,
                        batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                    )
                    self.valid_loader = DataLoader(
                        valid_set, collate_fn=collate, shuffle=False,
                        batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                    )
                elif args.mode == 'infer':
                    self.valid_loader = DataLoader(
                        valid_set, collate_fn=collate, shuffle=False,
                        batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                    )
                    self.test_loader = DataLoader(
                        test_set, collate_fn=collate, shuffle=False,
                        batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                    )
            elif getattr(self.args, 'dataset', 'ERGM') == 'MELD':
                if args.mode == 'train':
                    assert self.args.meld_text_json and self.args.meld_aud_pkl and self.args.meld_img_pkl, \
                        "For MELD, please set --meld_text_json, --meld_aud_pkl, --meld_img_pkl"
                if args.mode == 'train':
                    train_set = MELDDialoguePKLDataset(
                        json_path=self.args.meld_text_json,
                        audio_pkl=self.args.meld_aud_pkl,
                        video_pkl=self.args.meld_img_pkl,
                        split="train",
                    )
                    valid_set = MELDDialoguePKLDataset(
                        json_path=self.args.meld_text_json_val,
                        audio_pkl=self.args.meld_aud_pkl,
                        video_pkl=self.args.meld_img_pkl,
                        split="val",
                    )
                elif args.mode == 'infer':
                    valid_set = MELDDialoguePKLDataset(
                        json_path=self.args.meld_text_json_val,
                        audio_pkl=self.args.meld_aud_pkl,
                        video_pkl=self.args.meld_img_pkl,
                        split="val",
                    )
                    test_set = MELDDialoguePKLDataset(
                        json_path=self.args.meld_text_json_test,
                        audio_pkl=self.args.meld_aud_pkl_test,
                        video_pkl=self.args.meld_img_pkl_test,
                        split="test",
                    )

                collate = ppd.meld_collate(self.tokenizer)
                if args.mode == 'train':
                    self.train_loader = DataLoader(
                        train_set, collate_fn=collate, shuffle=True,
                        batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                    )
                    self.valid_loader = DataLoader(
                        valid_set, collate_fn=collate, shuffle=False,
                        batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                    )
                elif args.mode == 'infer':
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
            # 参数分组：主干用 args.lr；新初始化的模块（融合层 / 情感头 / selector）用更大的 lr，
            # 否则它们和预训练主干共用 1e-5，几乎学不动（α 会塌缩成一个常数偏置）。
            tr = getattr(self.model, "transformer", None)
            new_modules = []
            for name in ("aud_proj", "img_proj", "fus_proj", "ln_fus", "aug_proj", "mha_self", "mha_cross"):
                mod = getattr(tr, name, None) if tr is not None else None
                if mod is not None:
                    new_modules.append(mod)
            if getattr(self.model, "emotion_head", None) is not None:
                new_modules.append(self.model.emotion_head)
            new_ids = {id(p) for m in new_modules for p in m.parameters()}

            base_params = [p for p in self.model.parameters() if id(p) not in new_ids]
            new_params = [p for p in self.model.parameters() if id(p) in new_ids]
            new_params += list(self.selector.parameters())

            self._all_params = base_params + new_params
            self.optim = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": self.args.lr},
                    {"params": new_params, "lr": self.args.new_module_lr},
                ],
                lr=self.args.lr,
            )
            print(f"  param groups: backbone={sum(p.numel() for p in base_params) / 1e6:.1f}M @lr={self.args.lr}"
                  f" | new modules={sum(p.numel() for p in new_params) / 1e6:.1f}M @lr={self.args.new_module_lr}")

            if args.mode == 'train':
                num_batches = len(self.train_loader)
                args.total_train_steps = args.num_epochs * num_batches
                args.warmup_steps = int(args.warmup_ratio * args.total_train_steps)

                self.sched = get_polynomial_decay_schedule_with_warmup(
                    self.optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_train_steps, power=2,
                )
            if args.mode == 'infer' and args.choose_use_test_or_val == 'val':
                num_batches = len(self.valid_loader)
            elif args.mode == 'infer' and args.choose_use_test_or_val == 'test':
                num_batches = len(self.test_loader)

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

    def _init_train_log(self):
        """创建本次训练的日志文件：文件名带启动时间戳，本次训练的所有 epoch 写在同一个文件里。

        日志固定写到 <仓库根>/logs/ 下（用 __file__ 定位，不受启动时工作目录影响）。
        """
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.train_log_path = os.path.join(log_dir, f"train_{str(self.args.dataset).lower()}_{ts}.log")
        self._train_log_fp = open(self.train_log_path, "a", encoding="utf-8")
        self._train_start_time = time.time()
        header = [
            "=" * 110,
            f"RaMRA training run   start = {ts}",
            f"dataset          : {self.args.dataset}",
            f"model_type       : {self.args.model_type}",
            f"lr / batch / ep  : {self.args.lr} / {self.args.batch_size} / {self.args.num_epochs}"
            f"   (new_module_lr={getattr(self.args, 'new_module_lr', None)},"
            f" grad_clip={getattr(self.args, 'grad_clip', None)})",
            f"max_len / turns  : {self.args.max_len} / {getattr(self.args, 'max_turns', None)}",
            f"tau              : {getattr(self.args, 'tau', None)}",
            f"selector         : hidden={self.args.selector_hidden_dim} "
            f"layers={self.args.selector_num_layers} dropout={self.args.selector_dropout}",
            f"richness (A+C)   : lam={self.args.richness_lambda} tau_r={self.args.richness_tau} "
            f"smooth={self.args.richness_smooth} aux_floor={self.args.aux_floor} "
            f"every={self.args.richness_every}",
            f"ckpt_dir         : {self.args.ckpt_dir}",
            "=" * 110,
        ]
        for line in header:
            self._log(line)

    def _log(self, msg):
        """同时打印到控制台并写入本次训练日志。"""
        print(msg)
        fp = getattr(self, "_train_log_fp", None)
        if fp is not None:
            fp.write(msg + "\n")
            fp.flush()

    def _close_train_log(self):
        fp = getattr(self, "_train_log_fp", None)
        if fp is not None:
            fp.close()
            self._train_log_fp = None

    def train(self):
        self.fix_seed(self.args.seed)
        print("Training starts.")
        start_epoch = self.last_epoch + 1
        self._init_train_log()
        global_step = 0

        for epoch in range(start_epoch, start_epoch + self.args.num_epochs):
            self.model.train()
            if self.selector is not None:
                self.selector.train()
            print(f"-" * 35 + f"Epoch: {epoch}" + "-" * 35)

            train_total_losses = []
            train_lm_losses = []  # For PPL calculation
            train_correct_emotions = 0
            train_total_emotions = 0
            # 训练期 α 统计（用于判断 MRD 是否真的在分化）
            tr_alpha_sum = torch.zeros(3)
            tr_main_counts = torch.zeros(3, dtype=torch.long)
            tr_n = 0
            tr_kl_sum = 0.0
            tr_kl_n = 0

            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch

                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device),  # Ensure emotion labels are tensors
                )
                # 融合模块需要 imgs/auds 与模型同设备
                imgs = imgs.to(self.args.device, dtype=torch.float32) if imgs is not None else None
                auds = auds.to(self.args.device, dtype=torch.float32) if auds is not None else None

                gated_weight, main_idx = self._selector_forward(input_ids, imgs, auds)

                if gated_weight is not None:
                    with torch.no_grad():
                        tr_alpha_sum += gated_weight.detach().float().sum(dim=0).cpu()
                        tr_main_counts += torch.bincount(
                            main_idx.detach().view(-1).cpu(), minlength=3)[:3]
                        tr_n += int(gated_weight.size(0))

                outputs = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids,
                    attention_mask=self._attn_mask(input_ids),
                    labels=lm_labels, emotion_labels=emotion_labels,
                    imgs=imgs,
                    auds=auds,
                    gated_weights=gated_weight,
                    main_idx=main_idx,
                )

                loss = outputs.loss
                # ---- 方案 C：richness 监督，让 α 对齐"模态边际贡献" ----
                if (self.args.richness_lambda > 0 and gated_weight is not None
                        and (global_step % max(1, self.args.richness_every) == 0)):
                    warm = max(1, int(self.args.richness_warmup_steps))
                    lam = self.args.richness_lambda * min(1.0, global_step / warm)
                    if lam > 0:
                        rich = self._richness_kl(input_ids, token_type_ids, lm_labels, emotion_labels,
                                                 imgs, auds, gated_weight, main_idx, base_out=outputs)
                        loss = loss + lam * rich
                        tr_kl_sum += float(rich.detach())
                        tr_kl_n += 1
                global_step += 1

                self.optim.zero_grad()
                loss.backward()
                # 梯度裁剪：新模块用了更大的 lr，融合 attention 容易爆梯度
                if getattr(self.args, "grad_clip", 0) and self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._all_params, self.args.grad_clip)
                self.optim.step()
                self.sched.step()

                train_total_losses.append(loss.item())

                with torch.no_grad():
                    shift_logits, shift_labels = self._shift_for_lm(outputs.logits, lm_labels)
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
            # self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            # self.writer.add_scalar("PPL/train", train_ppl, epoch)
            # self.writer.add_scalar("Accuracy/train", train_acc, epoch)

            tr_alpha_txt = ""
            if tr_n > 0:
                tr_alpha = tr_alpha_sum / tr_n
                tr_alpha_txt = (f" | α_train=[{tr_alpha[0]:.2f},{tr_alpha[1]:.2f},{tr_alpha[2]:.2f}]"
                                f" main={int(tr_main_counts[0])}/{int(tr_main_counts[1])}/{int(tr_main_counts[2])}")
            if tr_kl_n > 0:
                tr_alpha_txt += f" rich_kl={tr_kl_sum / tr_kl_n:.4f}"

            self.last_epoch += 1
            valid_loss, valid_ppl, valid_acc = self.validation()

            # ---- 每个 epoch 的训练/验证结果写入日志（同一文件，逐行追加）----
            is_best = valid_ppl < self.best_ppl
            if is_best:
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
                # 先写临时文件再原子改名：中途被打断或磁盘写满也不会留下损坏的 ckpt
                tmp_path = save_path + ".tmp"
                try:
                    torch.save(state_dict, tmp_path)
                    os.replace(tmp_path, save_path)
                    self._log(f"    [ckpt] saved {save_path}  ({os.path.getsize(save_path) / 1e9:.2f} GB)")
                except Exception as e:
                    self._log(f"    [ckpt][ERROR] 保存失败: {e}")
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

            va = getattr(self, "_val_alpha_stats", None)
            va_txt = ""
            if va is not None and va["n"] > 0:
                va_txt = (f" | α_val=[{va['alpha'][0]:.2f},{va['alpha'][1]:.2f},{va['alpha'][2]:.2f}]"
                          f" std={va['std']:.3f} main={va['main'][0]}/{va['main'][1]}/{va['main'][2]}")
            elapsed_min = (time.time() - self._train_start_time) / 60.0
            self._log(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] epoch {epoch}/{start_epoch + self.args.num_epochs - 1}"
                f"  elapsed={elapsed_min:.1f}min"
                f" | train loss={avg_train_loss:.4f} ppl={train_ppl:.4f}"
                f" | valid ppl={valid_ppl:.4f}{'  <== NEW BEST' if is_best else ''}"
                f"{tr_alpha_txt}{va_txt}"
            )

            print(f"Best valid PPL: {self.best_ppl:.4f}")
            print(
                f"Current valid loss: {valid_loss:.4f} | Current valid PPL: {valid_ppl:.4f} | Current valid Emotion Acc: {valid_acc:.2f}%")
            # self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            # self.writer.add_scalar("PPL/valid", valid_ppl, epoch)
            # self.writer.add_scalar("Accuracy/valid", valid_acc, epoch)

        print("Training finished!")
        self._log(f"Training finished. best valid PPL = {self.best_ppl:.4f}")
        self._log(f"log file: {getattr(self, 'train_log_path', None)}")
        self._close_train_log()

    def validation(self):
        print("Validation processing...")
        self.model.eval()
        if self.selector is not None:
            self.selector.eval()

        valid_total_losses = []
        valid_lm_losses = []
        valid_correct_emotions = 0
        valid_total_emotions = 0

        # 验证期 α 统计（均值 / 标准差 / 主模态分布），用于判断 MRD 是否在分化
        va_alpha_sum = torch.zeros(3)
        va_alpha_sq = torch.zeros(3)
        va_main_counts = torch.zeros(3, dtype=torch.long)
        va_n = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch

                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device),
                )
                imgs = imgs.to(self.args.device, dtype=torch.float32) if imgs is not None else None
                auds = auds.to(self.args.device, dtype=torch.float32) if auds is not None else None

                gate, main_idx = self._selector_forward(input_ids, imgs, auds)

                if gate is not None:
                    g = gate.detach().float().cpu()
                    va_alpha_sum += g.sum(dim=0)
                    va_alpha_sq += (g ** 2).sum(dim=0)
                    va_main_counts += torch.bincount(
                        main_idx.detach().view(-1).cpu(), minlength=3)[:3]
                    va_n += int(g.size(0))

                outputs = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids,
                    attention_mask=self._attn_mask(input_ids),
                    labels=lm_labels, emotion_labels=emotion_labels,
                    imgs=imgs,
                    auds=auds,
                    gated_weights=gate,
                    main_idx=main_idx,
                )

                valid_total_losses.append(outputs.loss.item())

                shift_logits, shift_labels = self._shift_for_lm(outputs.logits, lm_labels)
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

        # ---- α 统计（供 train() 写入日志）----
        if va_n > 0:
            mean = va_alpha_sum / va_n
            var = (va_alpha_sq / va_n - mean ** 2).clamp_min(0.0)
            self._val_alpha_stats = {
                "alpha": [float(v) for v in mean.tolist()],
                "std": float(var.sqrt().mean().item()),
                "main": [int(v) for v in va_main_counts.tolist()],
                "n": int(va_n),
            }
        else:
            self._val_alpha_stats = None

        return avg_valid_loss, valid_ppl, valid_acc

    def nucleus_sampling(self, input_ids, token_type_ids, input_len,
                         imgs=None, auds=None, gated_weights=None, main_idx=None):
        """自回归生成。imgs/auds/gated_weights/main_idx 为 None 时退化为纯文本生成。"""
        output_ids = []
        for pos in range(input_len, self.args.max_len):
            outputs = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=torch.ones_like(input_ids),
                imgs=imgs,
                auds=auds,
                gated_weights=gated_weights,
                main_idx=main_idx,
            )
            # 融合分支会在序列最前面插入伪 token，因此用 -1 取最后一位，避免位置偏移
            next_token_logits = outputs.logits[:, -1, :]

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

    @staticmethod
    def _shift_for_lm(logits, labels):
        """算 PPL 用的 shift_logits / shift_labels。

        模态融合会在序列最前面插入 1 个伪 token（不参与语言建模），
        此时 logits 比 labels 多一位，必须先把 labels 左侧补 -100 再 shift。
        """
        if logits.size(1) == labels.size(1) + 1:
            labels = torch.cat([labels.new_full((labels.size(0), 1), -100), labels], dim=1)
        elif logits.size(1) != labels.size(1):
            n = min(logits.size(1), labels.size(1))
            logits, labels = logits[:, -n:], labels[:, -n:]
        return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()

    def _attn_mask(self, input_ids):
        """padding 位置为 0，其余为 1。pad_id 用 <bos>，不会和真实 token 冲突。"""
        pad_id = getattr(self.args, 'pad_id', None)
        if pad_id is None:
            return None
        return (input_ids != int(pad_id)).long()

    def _per_sample_loss(self, out, lm_labels, emotion_labels):
        """单样本的 L_lm + L_emo（token 维取均值），用于构造 richness 的边际贡献。"""
        sl, st = self._shift_for_lm(out.logits, lm_labels)
        lm = nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), st.reshape(-1),
            reduction='none', ignore_index=-100,
        ).view(sl.size(0), -1)
        denom = (st != -100).view(st.size(0), -1).sum(1).clamp(min=1).float()
        lm = lm.sum(1) / denom
        em = nn.functional.cross_entropy(out.emotion_logits, emotion_labels, reduction='none')
        return lm + em

    def _richness_kl(self, input_ids, token_type_ids, lm_labels, emotion_labels,
                     imgs, auds, gate, main_idx, base_out=None):
        """方案 C：用"边际贡献"给 α 构造监督目标。

            Δ_T = L(关闭融合) − L(全模态)
            Δ_A = L(移除音频证据) − L(全模态)
            Δ_V = L(移除视频证据) − L(全模态)
            target = (1-s)·softmax(Δ/τ_r) + s/3
            loss_rich = CE(target, α)

        三次前向都在 no_grad 下完成（只需要标量目标），额外开销约 +1 个 forward。
        """
        with torch.no_grad():
            base = self._per_sample_loss(base_out, lm_labels, emotion_labels)
            forced = torch.zeros_like(main_idx)      # richness 估计统一用文本 hub，保证 Δ 语义一致
            deltas = []
            for ab in (3, 1, 2):
                o = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids,
                    attention_mask=self._attn_mask(input_ids),
                    labels=lm_labels, emotion_labels=emotion_labels,
                    imgs=imgs, auds=auds,
                    gated_weights=gate.detach(), main_idx=forced,
                    ablate=ab,
                )
                deltas.append(self._per_sample_loss(o, lm_labels, emotion_labels) - base)
            delta = torch.stack(deltas, dim=-1).clamp_min(0.0)          # (B,3)
            target = torch.softmax(delta / max(1e-6, self.args.richness_tau), dim=-1)
            s = float(self.args.richness_smooth)
            target = (1.0 - s) * target + s / 3.0
        logp = torch.log(gate.clamp_min(1e-8))
        return -(target * logp).sum(dim=-1).mean()

    def _pool_text_feat(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """用 GPT-2 的 token embedding 做均值池化，得到 (B, E) 的轻量文本特征供 selector 使用。

        两个要点：
          1) 只对真实 token 求均值（排除 padding），否则同一句在不同 batch 里特征不同；
          2) 只池化到第一个 </s> 为止 —— input_ids 里还带着参考回复，把它算进 selector 特征
             既稀释了上下文语义，也造成训练/推理分布不一致。
        """
        with torch.no_grad():
            emb = self.model.transformer.wte(input_ids)  # (B, L, E)
            if attention_mask is not None:
                m = attention_mask.to(emb.device).float()
            else:
                m = torch.ones_like(input_ids, dtype=torch.float32)
            # 上下文掩码：第一个 </s> 及其之前为 1
            eos_id = getattr(self.args, "eos_id", None)
            if eos_id is not None:
                ctx_mask = ((input_ids == int(eos_id)).cumsum(dim=1) <= 1).float()
                m = m * ctx_mask
            m = m.unsqueeze(-1).to(emb.dtype)
            feat = (emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return feat.float()

    def _pool_modal_feat(self, x, mask: torch.Tensor = None):
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

        # 有 attention mask 时按真实 token 长度池化，避免 padding 的 0 行稀释特征
        if mask is not None and torch.is_tensor(x) and x.dim() == 3:
            m = mask.to(x.device).unsqueeze(-1).to(x.dtype)
            return ((x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)).float()

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

        mask = self._attn_mask(input_ids)
        with torch.no_grad():
            text_feat = self._pool_text_feat(input_ids.to(self.args.device),
                                             mask.to(self.args.device) if mask is not None else None)
        audio_feat = self._pool_modal_feat(auds, mask)
        video_feat = self._pool_modal_feat(imgs, mask)

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
            # 传入各模态维度 -> 三塔各自编码自己的模态（否则回退到共享 backbone）
            d_text=int(text_dim),
            d_audio=int(audio_dim),
            d_video=int(video_dim),
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

        mask = self._attn_mask(input_ids)
        text_feat = self._pool_text_feat(input_ids, mask)  # (B, E)
        audio_feat = self._pool_modal_feat(auds, mask)
        video_feat = self._pool_modal_feat(imgs, mask)

        states = _build_states(text_feat.to(self.args.device), audio_feat.to(self.args.device) if audio_feat is not None else None,
                               video_feat.to(self.args.device) if video_feat is not None else None)
        # τ 温度：直接作用于 selector 的 gate logits（论文 Eq.6）。
        # τ=None / 1.0 时与旧行为完全一致；训练与推理都会生效。
        tau = getattr(self.args, 'tau', None)
        try:
            sel_out = self.selector(states, tau=tau)
        except TypeError:
            # 兼容不接受 tau 的 selector 实现
            sel_out = self.selector(states)
        gate = sel_out.get("gate")
        if gate is None:
            tau_val = float(tau) if (tau is not None and float(tau) > 0) else 1.0
            gate = torch.softmax(sel_out["logits"] / tau_val, dim=-1)

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
        test_correct_emotions = 0
        test_total_emotions = 0

        save_alpha = bool(getattr(self.args, 'save_alpha_log', False))
        alpha_records = []

        with torch.no_grad():
            if self.args.choose_use_test_or_val == 'val':
                data_loader = self.valid_loader
            elif self.args.choose_use_test_or_val == 'test':
                data_loader = self.test_loader
            for _, batch in enumerate(tqdm(data_loader)):
                input_ids, token_type_ids, lm_labels, imgs, auds, contexts, emotion_labels = batch

                input_ids, token_type_ids, lm_labels, emotion_labels = (
                    input_ids.to(self.args.device),
                    token_type_ids.to(self.args.device),
                    lm_labels.to(self.args.device),
                    torch.LongTensor(emotion_labels).to(self.args.device),
                )
                imgs = imgs.to(self.args.device, dtype=torch.float32) if imgs is not None else None
                auds = auds.to(self.args.device, dtype=torch.float32) if auds is not None else None

                gate, main_idx = self._selector_forward(input_ids, imgs, auds)

                for i in range(input_ids.size(0)):
                    current_input = input_ids[i].unsqueeze(0)
                    current_token_types = token_type_ids[i].unsqueeze(0)

                    # 生成前缀 = 上下文 + 结尾的 </s>。
                    # 注意：不能用 (input_ids != eos).sum()，那样会把参考回复也数进去，
                    # 等于把参考答案喂给模型（会严重高估 BERTScore）。
                    eos_pos = (current_input[0] == self.args.eos_id).nonzero(as_tuple=True)[0]
                    if eos_pos.numel() > 0:
                        input_len = int(eos_pos[0].item()) + 1
                    else:
                        input_len = int((current_input != self.args.pad_id).sum().item())
                    input_len = min(input_len, current_input.size(1))

                    sample_img = imgs[i:i + 1, :input_len] if imgs is not None else None
                    sample_aud = auds[i:i + 1, :input_len] if auds is not None else None

                    output_ids = self.nucleus_sampling(current_input[:, :input_len], current_token_types[:, :input_len],
                                                       input_len,
                                                       imgs=sample_img, auds=sample_aud,
                                                       gated_weights=(gate[i:i + 1] if gate is not None else None),
                                                       main_idx=(main_idx[i:i + 1] if main_idx is not None else None))
                    hypothesis_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    print("hyp: ", hypothesis_text)
                    all_hypotheses.append(hypothesis_text)

                    ref_ids = lm_labels[i][lm_labels[i] != -100]  # Filter out padding
                    reference_text = self.tokenizer.decode(ref_ids, skip_special_tokens=True)
                    all_references.append(reference_text)

                    all_true_labels.append(emotion_labels[i].item())

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=self._attn_mask(input_ids),
                    labels=lm_labels,
                    emotion_labels=emotion_labels,
                    imgs=imgs,
                    auds=auds,
                    gated_weights=gate,
                    main_idx=main_idx,
                )
                preds = torch.argmax(outputs.emotion_logits, dim=-1)
                test_correct_emotions += (preds == emotion_labels).sum().item()
                test_total_emotions += emotion_labels.size(0)

                if save_alpha:
                    B = input_ids.size(0)
                    base = len(all_hypotheses) - B
                    gate_cpu = gate.detach().float().cpu().tolist()
                    main_cpu = main_idx.detach().cpu().tolist()
                    pred_cpu = preds.detach().cpu().tolist()
                    for i in range(B):
                        alpha_records.append({
                            "sample_id": base + i,
                            "alpha": gate_cpu[i],
                            "main_modal": int(main_cpu[i]),
                            "pred_emo": int(pred_cpu[i]),
                            "true_emo": int(emotion_labels[i].item()),
                            "hypothesis": all_hypotheses[base + i] if base + i < len(all_hypotheses) else "",
                            "reference": all_references[base + i] if base + i < len(all_references) else "",
                        })

                shift_logits, shift_labels = self._shift_for_lm(outputs.logits, lm_labels)
                loss_fct_lm = nn.CrossEntropyLoss()
                lm_loss = loss_fct_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                all_losses.append(lm_loss.item())

        test_acc = (test_correct_emotions / test_total_emotions) * 100 if test_total_emotions > 0 else 0.0
        return all_hypotheses, all_references, all_true_labels, all_losses, test_acc, alpha_records

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
    parser.add_argument("--new_module_lr", type=float, default=1e-3,
                        help="LR for newly initialized modules (fusion / emotion head / selector).")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max-norm (<=0 disables).")
    parser.add_argument("--richness_lambda", type=float, default=0.1,
                        help="Weight λ of the richness KL term (<=0 disables, reverts to plain LM+emo).")
    parser.add_argument("--richness_tau", type=float, default=0.5,
                        help="Temperature τ_r: target = softmax(Δ/τ_r).")
    parser.add_argument("--richness_smooth", type=float, default=0.1,
                        help="Mix ratio of the uniform distribution into the richness target.")
    parser.add_argument("--richness_every", type=int, default=1,
                        help="Estimate the richness target every N steps (cost saving).")
    parser.add_argument("--richness_warmup_steps", type=int, default=500,
                        help="Linearly ramp the richness weight lambda over the first N steps.")
    parser.add_argument("--aux_floor", type=float, default=0.05,
                        help="Floor added to auxiliary α before within-auxiliary renormalization.")
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
    parser.add_argument("--dataset", type=str, default="ERGM", choices=["ERGM", "IEMOCAP", "MELD"],
                        help="Choose data pipeline. IEMOCAP uses PKL+JSON via IEMOCAPDialoguePKLDataset. MELD uses JSON+separate A/V PKLs.")
    parser.add_argument("--train_pkls", type=str, default=None,
                        help="(IEMOCAP) path to train PKL or a comma-separated list of PKLs")
    parser.add_argument("--val_pkls", type=str, default=None,
                        help="(IEMOCAP) path to val PKL or a comma-separated list of PKLs")
    parser.add_argument("--iemocap_text_json", type=str, default=None,
                        help="(IEMOCAP) path to JSON holding raw text/dialogue info to be encoded")
    parser.add_argument("--meld_text_json", type=str, default=None,
                        help="(MELD) path to meld_diadict.json")
    parser.add_argument("--meld_text_json_val", type=str, default=None,
                        help="(MELD) path to meld_diadict_val.json")
    parser.add_argument("--meld_text_json_test", type=str, default=None,
                        help="(MELD) path to meld_diadict_test.json")
    parser.add_argument("--meld_aud_pkl", type=str, default=None,
                        help="(MELD) path to audio features PKL with keys {'0','1'}")
    parser.add_argument("--meld_img_pkl", type=str, default=None,
                        help="(MELD) path to image/video features PKL with keys {'0','1'}")
    parser.add_argument("--meld_aud_pkl_test", type=str, default=None,
                        help="(MELD) path to audio features PKL with keys {'0','1'}")
    parser.add_argument("--meld_img_pkl_test", type=str, default=None,
                        help="(MELD) path to image/video features PKL with keys {'0','1'}")
    parser.add_argument("--tau", type=float, default=None,
                        help="推理时 MRD 温度缩放 τ；不传则默认 1.0（等价于原始行为）")
    parser.add_argument("--save_alpha_log", action="store_true",
                        help="推理时将每个样本的 RaMRA α (gate) 落盘到 alpha_log.jsonl（仅影响 infer）")
    parser.add_argument("--choose_use_test_or_val", type=str, default='val', choices=["val", "test"],
                        help="Choose use which dataset split to infer.")

    args = parser.parse_args()

    model_name = args.model_type.split("/")[-1]
    args.data_dir = os.path.join(args.data_dir, model_name, args.dataset.lower())
    args.ckpt_dir = os.path.join(args.ckpt_dir, model_name, args.dataset.lower())

    if args.mode == 'train':
        manager = Manager(args)
        manager.train()
    elif args.mode == 'infer':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint using --ckpt_name."
        manager = Manager(args)

        if args.dataset == "IEMOCAP":
            baseline_path = "/root/autodl-tmp/ERGM-main/tools/models/roberta-large/roberta-large.tsv"
        elif args.dataset == "MELD":
            baseline_path = "/root/autodl-tmp/ERGM-main/tools/models/roberta-large/roberta-large-meld.tsv"

        evaluator = Evaluator(device=manager.args.device,
                              bert_model_path='/root/autodl-tmp/ERGM-main/tools/models/roberta-large',
                              baseline_path=baseline_path,
                              num_layers=24,
                              rescale_with_baseline=False, )

        hypotheses, references, true_labels, losses, test_acc, alpha_records = manager.test()
        infer_dump_path = os.path.join(args.output_dir, args.dataset.lower(), f"{args.ckpt_name}_infer_outputs.pkl")
        os.makedirs(os.path.dirname(infer_dump_path), exist_ok=True)

        if args.save_alpha_log and len(alpha_records) > 0:
            alpha_log_path = os.path.join(args.output_dir, args.dataset.lower(), "alpha_log.jsonl")
            with open(alpha_log_path, "w", encoding="utf-8") as f:
                for rec in alpha_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[Alpha] saved {len(alpha_records)} records to {alpha_log_path}")
        with open(infer_dump_path, "wb") as f:
            pickle.dump(
                {
                    "hypotheses": hypotheses,
                    "references": references,
                    "true_labels": true_labels,
                    "losses": losses,
                    "test_acc": test_acc,
                },
                f,
            )

        final_metrics = evaluator.evaluate_all(
            hypotheses=hypotheses,
            references=references
        )

        test_avg_losses = np.mean(losses)
        test_ppl = math.exp(test_avg_losses)

        print(f"PPL: {test_ppl:.4f}")
        print(f"Emotion Acc: {test_acc:.2f}%")

        print("\n--- Final Evaluation Results ---")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{metric.upper():<10}: {value:.4f}")
            else:
                print(f"{metric.upper():<10}: {value}")
        print("--------------------------------")

        os.makedirs(args.data_dir, exist_ok=True)
        results_file_path = os.path.join(args.data_dir, f"{args.ckpt_name}_evaluation_results.txt")
        with open(results_file_path, "w", encoding="utf-8") as f:
            for metric, value in final_metrics.items():
                f.write(f"{metric}: {value}\n")

        # 供 analysis/tau_sensitivity_plot.py 读取（它期望 <output_dir>/metrics.json）
        metrics_payload = {
            "ppl": float(test_ppl),
            "bertscore": float(final_metrics.get("f1", float("nan"))),
            "emo_acc": float(test_acc),
            "dist_1": float(final_metrics.get("dist_1", float("nan"))),
            "dist_2": float(final_metrics.get("dist_2", float("nan"))),
            "tau": float(args.tau) if getattr(args, "tau", None) else 1.0,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
        print(f"[Metrics] saved to {os.path.join(args.output_dir, 'metrics.json')}")
