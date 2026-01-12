import os
import pickle
from typing import List, Dict
import numpy as np
from tqdm import tqdm

import torch
import librosa
import torch.nn.functional as F
from PIL import Image
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    BartForConditionalGeneration,
    BartTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaModel,
    BartTokenizerFast,
    BartModel,
    RobertaTokenizerFast,
    GPT2TokenizerFast,
    GPT2Model,
    GPT2Config,
    GPT2LMHeadModel,
    HubertModel,
    CLIPProcessor,
    CLIPModel,
    AutoFeatureExtractor,
)


# Text
class TextEncoder:
    def __init__(self, task: str, egc_model: str = 'facebook/bart-base', cls_model: str = 'roberta-base',
                 device: str = 'cpu'):
        self.task = task
        self.device = device
        self.out_dim = None  # will be set after loading encoder
        if task == 'EGC':
            if 'bart' in egc_model.lower():
                if BartTokenizerFast is None or BartModel is None:
                    raise ImportError("transformers (Bart) not available.")
                self.tok = BartTokenizerFast.from_pretrained(egc_model)
                self.enc = BartModel.from_pretrained(egc_model).to(device)
                self.enc.eval()
                self.out_dim = self.enc.config.hidden_size
            elif 'gpt2' in egc_model.lower():
                if GPT2TokenizerFast is None or GPT2Model is None:
                    raise ImportError("transformers (GPT2) not available.")
                self.tok = GPT2TokenizerFast.from_pretrained(egc_model)
                special_tokens = {
                    "additional_special_tokens": ["<sp1>", "<sp2>"],
                }
                num_new_tokens = self.tok.add_special_tokens(special_tokens)
                # GPT‑2 没有默认 pad_token，设为 eos 以便批处理/attention_mask 正确
                if self.tok.pad_token is None:
                    self.tok.pad_token = self.tok.eos_token
                self.enc = GPT2Model.from_pretrained(egc_model).to(device)
                try:
                    self.enc.resize_token_embeddings(len(self.tok))
                except Exception:
                    pass
                self.enc.eval()
                self.out_dim = self.enc.config.hidden_size  # 通常 768
            else:
                raise ValueError(f"Unsupported EGC text backbone: {egc_model}")
        else:
            if RobertaTokenizerFast is None or RobertaModel is None:
                raise ImportError("transformers (RoBERTa) not available.")
            self.tok = RobertaTokenizerFast.from_pretrained(cls_model)
            self.enc = RobertaModel.from_pretrained(cls_model).to(device)
            self.enc.eval()
            self.out_dim = self.enc.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Return (B, D) mean‑pooled encoder features with padding‑aware masking."""
        enc = self.tok(texts, padding=True, truncation=True, return_tensors='pt')
        # 将输入移到与模型相同的 device
        for k in enc:
            enc[k] = enc[k].to(self.device)
        outputs = self.enc(**enc)
        # 统一使用 last_hidden_state + attention_mask 做 mean pooling
        last = outputs.last_hidden_state  # (B, L, D)
        attn = enc.get('attention_mask', None)
        if attn is None:
            # 保险：若无 mask，构造全 1
            attn = torch.ones(last.size()[:2], dtype=last.dtype, device=last.device)
        attn = attn.unsqueeze(-1)  # (B, L, 1)
        summed = (last * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1.0)
        mean_pooled = summed / counts
        return mean_pooled.detach().cpu()

# Audio
def extract_audio_features(audio_path, model_name: str = "facebook/hubert-large-ll60k", device: str = "cpu"):
    """
    Extract audio features with HuBERT (default: facebook/hubert-large-ll60k).
    Returns last_hidden_state (T, D) on CPU or None on failure.
    """
    try:
        processor = AutoFeatureExtractor.from_pretrained(model_name)
        model = HubertModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"[audio] failed to load HuBERT ({model_name}): {e}")
        return None

    try:
        audio_input, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"[audio] failed to load audio {audio_path}: {e}")
        return None
    if isinstance(audio_input, np.ndarray):
        audio_input = np.ascontiguousarray(audio_input, dtype=np.float32)
    if audio_input is None or (isinstance(audio_input, np.ndarray) and audio_input.size == 0):
        print(f"[audio] empty waveform for {audio_path}")
        return None

    inputs = processor(
        audio_input,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model(**inputs).last_hidden_state

    return features.cpu()


def extract_audio_dir(
    audio_dir: str,
    pkl_path: str,
    mode: int,
    model_name: str = "facebook/hubert-large-ll60k",
    device: str = "cuda:0",
    pool: str = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Batch-extract audio embeddings for all .wav files under `audio_dir` and store in a PKL.
    - mode: 0=train, 1=valid (used as top-level key: "0" / "1")
    - pkl_path: PKL will hold {"0": {...}, "1": {...}}, each inner dict maps basename -> embedding tensor.
    - pool: if "mean", mean-pool over time; if "raw", store full sequence.
    Returns the full dict after update.
    """
    mode_key = str(int(mode))
    audio_dir = os.path.abspath(audio_dir)
    os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)

    # load existing or init
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            store = pickle.load(f)
    else:
        store = {"0": {}, "1": {}}

    if "0" not in store:
        store["0"] = {}
    if "1" not in store:
        store["1"] = {}

    # prepare model once
    try:
        processor = AutoFeatureExtractor.from_pretrained(model_name)
        model = HubertModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load HuBERT model {model_name}: {e}") from e

    # iterate wav files
    wav_files = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith(".wav")
    ]
    wav_files.sort()

    for wav in tqdm(wav_files, desc="audio", ncols=80):
        key = os.path.splitext(os.path.basename(wav))[0]  # e.g., dia0_utt0
        if key in store[mode_key]:
            # already processed; skip to save time
            continue
        try:
            audio_input, sample_rate = librosa.load(wav, sr=16000, mono=True)
        # ensure contiguous float32 array
            if isinstance(audio_input, np.ndarray):
                audio_input = np.ascontiguousarray(audio_input, dtype=np.float32)
            if audio_input is None or (isinstance(audio_input, np.ndarray) and audio_input.size == 0):
                raise RuntimeError(f"[audio] empty waveform for {wav}")
        except Exception as e:
            print(f"[audio] skip {wav} due to load error: {e}")
            continue

        inputs = processor(
            audio_input,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = model(**inputs).last_hidden_state  # (1, T, D)

            if pool == "mean":
                feats = feats.mean(dim=1).squeeze(0)  # (D,)
            elif pool == "adaptive":
                L = 64  # 你可以改成 32/64/128，建议和视频帧数/你想对齐的 token 数一致
                feats = feats.squeeze(0)  # (T, D)

                feats = F.adaptive_avg_pool1d(
                    feats.transpose(0, 1).unsqueeze(0),  # (1, D, T)
                    L
                ).squeeze(0).transpose(0, 1)  # (L, D)
            else:
                feats = feats.squeeze(0)  # (T, D)

            feats = feats.cpu()

        store[mode_key][key] = feats.numpy().astype("float64")
        # print(f"[audio] processed {key} -> {feats.shape}")

    with open(pkl_path, "wb") as pf:
        pickle.dump(store, pf)
    return store


def extract_img_dir(
    img_dir: str,
    pkl_path: str,
    mode: int,
    model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda:0",
    batch_size: int = 8,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Batch-extract image embeddings for each utterance folder under `img_dir`.
    - Expects structure: img_dir/<utt_id>/frame_*.jpg (e.g., dia0_utt0 with 32 imgs).
    - Stores into PKL: {"0": {...}, "1": {...}}, inner dict maps utt_id -> ndarray (#frames, D).
    """
    mode_key = str(int(mode))
    img_dir = os.path.abspath(img_dir)
    os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)

    # load existing or init
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            store = pickle.load(f)
    else:
        store = {"0": {}, "1": {}}
    store.setdefault("0", {})
    store.setdefault("1", {})

    # prepare model once
    try:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP model {model_name}: {e}") from e
    feat_dim = getattr(model.config, "projection_dim", 768)

    utt_dirs = [
        os.path.join(img_dir, d)
        for d in os.listdir(img_dir)
        if os.path.isdir(os.path.join(img_dir, d))
    ]
    utt_dirs.sort()

    for utt_path in tqdm(utt_dirs, desc="image", ncols=80):
        key = os.path.basename(utt_path)
        if key in store[mode_key]:
            continue

        img_files = [
            os.path.join(utt_path, f)
            for f in os.listdir(utt_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        img_files.sort()
        if not img_files:
            # fallback: no frames found, fill zeros
            zeros = np.zeros((32, feat_dim), dtype=np.float32)
            store[mode_key][key] = zeros
            print(f"[image] no frames under {utt_path}, fill zeros {zeros.shape}")
            continue

        feats_list = []
        # batch to save memory
        for i in range(0, len(img_files), batch_size):
            batch_paths = img_files[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    print(f"[image] skip frame {p}: {e}")
            if not imgs:
                continue
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feat = model.get_image_features(**inputs)  # (B, D)
                feats_list.append(feat.cpu())

        if not feats_list:
            feats_np = np.zeros((32, feat_dim), dtype=np.float32)
            print(f"[image] no valid frames for {utt_path}, fill zeros {feats_np.shape}")
        else:
            feats = torch.cat(feats_list, dim=0)  # (N, D)
            feats_np = feats.numpy().astype(np.float32)
        store[mode_key][key] = feats_np
        print(f"[image] processed {key} -> {store[mode_key][key].shape}")

    with open(pkl_path, "wb") as pf:
        pickle.dump(store, pf)
    return store


# Visual
def extract_image_features(image_path, model_name: str = "openai/clip-vit-large-patch14", device: str = "cpu"):
    """
    Extract image features with CLIP ViT-L/14. Returns last_hidden_state or None on failure.
    """
    try:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"[image] failed to load CLIP ({model_name}): {e}")
        return None

    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[image] failed to open {image_path}: {e}")
        return None

    inputs = processor(images=raw_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        image_features = vision_outputs.last_hidden_state

    return image_features.cpu()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features for MELD assets.")
    parser.add_argument("--audio_dir", type=str, help="Path to folder of wav files to batch-extract.")
    parser.add_argument("--image_dir", type=str, help="Path to folder of wav files to batch-extract.")
    parser.add_argument("--audio_pkl", type=str, default="data/audio_feats.pkl", help="Where to save/load the audio features PKL.")
    parser.add_argument("--image_pkl", type=str, default="data/image_feats.pkl", help="Where to save/load the image features PKL.")
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1], help="0=train, 1=valid (used as dict key).")
    parser.add_argument("--audio_model", type=str, default="E:\\CODE\\ERGM\\tools\\hubert-large", help="HuBERT model name.")
    parser.add_argument("--visual_model", type=str, default="E:\\CODE\\ERGM\\tools\\clip-vit-large-patch14", help="Clip-VIT model name.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for extraction, e.g., cpu or cuda:0.")
    args = parser.parse_args()

    if args.audio_dir:
        extract_audio_dir(
            audio_dir=args.audio_dir,
            pkl_path=args.audio_pkl,
            mode=args.mode,
            model_name=args.audio_model,
            device=args.device,
            # pool="mean",
        )

    if args.image_dir:
        extract_img_dir(
            img_dir=args.image_dir,
            pkl_path=args.image_pkl,
            mode=args.mode,
            model_name=args.audio_model,
            device=args.device
        )
