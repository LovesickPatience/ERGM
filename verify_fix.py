"""CPU-only 校验：新的 collate + 修复后的融合分支是否能正确跑通。
不加载真实 GPT-2 权重，用一个极小配置的模型验证形状/逻辑。
"""
import sys
import types

sys.path.insert(0, r"E:\code\ERGM\ERGM")

import torch
from transformers import GPT2Tokenizer, GPT2Config

from src.custom_dataset import PadCollate
from selector.data_preprocess import IEMOCAPDialoguePKLDataset
from src.model.model_with_fusion import GPT2LMHeadModel

MODEL_DIR = r"E:\code\ERGM\ERGM\models\gpt2"
JSON_PATH = r"E:\code\ERGM\ERGM\data\iemocap\iemocap_diadict_with_emocause.json"
PKL_PATH = r"E:\code\ERGM\ERGM\data\iemocap\iemocap_data_0610.pkl"


def build_args(tok):
    a = types.SimpleNamespace()
    a.sp1_token, a.sp2_token = "<sp1>", "<sp2>"
    a.sp1_id = tok.convert_tokens_to_ids("<sp1>")
    a.sp2_id = tok.convert_tokens_to_ids("<sp2>")
    a.img_token, a.aud_token = "<img>", "<aud>"
    a.img_token_id = tok.convert_tokens_to_ids("<img>")
    a.aud_token_id = tok.convert_tokens_to_ids("<aud>")
    a.eos_id = tok.eos_token_id
    a.bos_id = tok.convert_tokens_to_ids("<bos>")
    a.pad_id = a.bos_id          # 与 main_with_selector 一致
    a.max_len = 1024
    a.emo_map = None
    return a


def main():
    tok = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    tok.add_special_tokens({
        "bos_token": "<bos>",
        "additional_special_tokens": ["<sp1>", "<sp2>", "<img>", "<aud>"],
    })
    args = build_args(tok)

    ds = IEMOCAPDialoguePKLDataset(json_path=JSON_PATH, pkl_path=PKL_PATH,
                                   split="train", max_turns=10)
    ppd = PadCollate(eos_id=args.eos_id, args=args)
    collate = ppd.iemocap_collate(tok)

    batch = [ds[i] for i in (0, 3, 500, 2500)]
    input_ids, tt, labels, imgs, auds, contexts, emo = collate(batch)

    print("=== collate 输出 ===")
    print("input_ids     ", tuple(input_ids.shape), input_ids.dtype)
    print("token_type_ids", tuple(tt.shape))
    print("labels        ", tuple(labels.shape))
    print("imgs (video)  ", None if imgs is None else tuple(imgs.shape))
    print("auds (audio)  ", None if auds is None else tuple(auds.shape))
    print("emotion_labels", emo)

    assert torch.is_tensor(imgs) and imgs.dim() == 3, "imgs 必须是 (B,L,D) 张量"
    assert imgs.size(0) == len(batch) and imgs.size(1) == input_ids.size(1)

    # padding 是否用 <bos>
    pad_rows = (input_ids == args.pad_id).sum().item()
    print("pad token (<bos>) 数量:", pad_rows, "-> 期望 > 0（说明右填充生效）")

    # 每条样本里"不同特征行"的数量，应约等于该样本的上下文句数（而不是 1）
    print("\n=== 逐句对齐检查 ===")
    for i, s in enumerate(batch):
        L = int((input_ids[i] != args.pad_id).sum())
        a_rows = auds[i][:L]
        uniq = torch.unique(a_rows, dim=0).size(0)
        print(f"  sample[{i}] L={L}  ctx_utts={len(s['ctx_utt_texts'])}  "
              f"不同音频特征行数={uniq}")

    # 解码第一条，确认格式
    print("\n=== 解码 sample[0] ===")
    L0 = int((input_ids[0] != args.pad_id).sum())
    print(tok.decode(input_ids[0][:L0], skip_special_tokens=False))

    # ---------- 极小模型前向 ----------
    print("\n=== 融合分支前向（tiny model, CPU）===")
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=len(tok),
                     n_positions=1024, add_cross_attention=True)
    cfg.audio_dim = 64
    cfg.video_dim = 64
    cfg.img_id = args.img_token_id
    cfg.aud_id = args.aud_token_id
    model = GPT2LMHeadModel(cfg)
    model.eval()

    attn = (input_ids != args.pad_id).long()
    B = input_ids.size(0)
    gate = torch.softmax(torch.randn(B, 3), dim=-1)
    main_idx = torch.argmax(gate, dim=-1)

    with torch.no_grad():
        out = model(input_ids=input_ids, token_type_ids=tt, attention_mask=attn,
                    labels=labels, emotion_labels=torch.tensor(emo),
                    imgs=imgs, auds=auds, gated_weights=gate, main_idx=main_idx)

    print("logits       ", tuple(out.logits.shape), "(应为 L，逐位置增强不改变长度)")
    print("emotion_logits", tuple(out.emotion_logits.shape))
    print("loss         ", float(out.loss))
    assert out.logits.size(1) == input_ids.size(1), "融合不应改变序列长度"
    assert torch.isfinite(out.loss), "loss 非有限值"

    # 关掉融合（main_idx=None）
    with torch.no_grad():
        out2 = model(input_ids=input_ids, token_type_ids=tt, attention_mask=attn,
                     labels=labels, emotion_labels=torch.tensor(emo))
    print("无融合 logits", tuple(out2.logits.shape))

    # 残差 + 零初始化：初始化时融合必须是恒等映射（loss 与非融合完全一致）
    d0 = abs(float(out.loss) - float(out2.loss))
    print(f"初始化时 融合loss={float(out.loss):.6f}  无融合loss={float(out2.loss):.6f}  diff={d0:.2e}")
    assert d0 < 1e-6, "初始化时融合不是恒等映射 -> aug_proj 零初始化失效"

    # 扰动 aug_proj 后，融合必须真正影响输出
    with torch.no_grad():
        model.transformer.aug_proj.weight.normal_(0, 0.02)
        out3 = model(input_ids=input_ids, token_type_ids=tt, attention_mask=attn,
                     labels=labels, emotion_labels=torch.tensor(emo),
                     imgs=torch.randn_like(imgs), auds=torch.randn_like(auds),
                     gated_weights=gate, main_idx=main_idx)
    diff = abs(float(out.loss) - float(out3.loss))
    print(f"扰动 aug_proj 后 loss={float(out3.loss):.6f}  diff={diff:.6f}")
    assert diff > 1e-6, "扰动后更换模态特征 loss 不变 -> 融合分支未生效"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
