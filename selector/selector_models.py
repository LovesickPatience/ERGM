from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SelectorConfig:
    input_dim: int                 # concatenated feature dim (text/audio/video + handcrafted)
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    mode: str = "discrete"        # "discrete" or "continuous"
    num_actions: int = 3           # required if mode == "discrete"
    num_modalities: int = 3        # required if mode == "continuous" (e.g., T/A/V)
    # 各模态的原始特征维度；三者都给定时，GatedTriTower 使用"三塔各自吃自己模态"的结构
    d_text: int = None
    d_audio: int = None
    d_video: int = None


@dataclass
class TriTowerCfg:
    d_text: int = 768
    d_aud: int = 64
    d_vid: int = 64
    hid_t: int = 256
    hid_a: int = 128
    hid_v: int = 128
    dropout: float = 0.2
    num_actions: int = 3  # {T, A, V} 或你的小集合


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        d_in = in_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(d_in, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d_in = hidden_dim
        layers += [nn.Linear(d_in, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseSelector(nn.Module):
    def forward(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class MLPSelector(BaseSelector):
    """
    If mode == "discrete": returns dict with keys {"logits"} for an action set of size num_actions.
    If mode == "continuous": returns dict with keys {"weights"} representing a simplex over num_modalities.
    """
    def __init__(self, cfg: SelectorConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.mode == "discrete":
            self.head = MLP(cfg.input_dim, cfg.hidden_dim, cfg.num_actions, cfg.num_layers, cfg.dropout)
        elif cfg.mode == "continuous":
            self.head = MLP(cfg.input_dim, cfg.hidden_dim, cfg.num_modalities, cfg.num_layers, cfg.dropout)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    def forward(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.head(states)
        if self.cfg.mode == "discrete":
            return {"logits": out}  # to be consumed by a DiscretePolicy
        else:
            # project to simplex via softmax (leave temperature to the policy if needed)
            weights = F.softmax(out, dim=-1)
            return {"weights": weights}


class TriTowerSelector(nn.Module):
    """Three-tower scorer: independently score T/A/V then concatenate into logits (B,3 or B,A).
    When used with a custom action set (A actions), we still output (B, A) by a small head on top of [s_t, s_a, s_v].
    If you want exact {T,A,V} scoring, set action set to 3-way and the mapping can be identity.
    """
    def __init__(self, d_text: int, d_aud: int, d_vid: int, hidden_t: int = 256, hidden_a: int = 128, hidden_v: int = 128,
                 dropout: float = 0.1, num_actions: int = 3):
        super().__init__()
        self.ln_t = nn.LayerNorm(d_text)
        self.ln_a = nn.LayerNorm(d_aud)
        self.ln_v = nn.LayerNorm(d_vid)
        self.tower_t = nn.Sequential(
            nn.Linear(d_text, hidden_t),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_t, 1)
        )
        self.tower_a = nn.Sequential(
            nn.Linear(d_aud, hidden_a),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_a, 1)
        )
        self.tower_v = nn.Sequential(
            nn.Linear(d_vid, hidden_v),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_v, 1)
        )
        # head 将 3 维打分映射到动作集合大小（例如 small6 / mix3）
        self.head = nn.Linear(3, num_actions)

    def forward(self, tf: torch.Tensor, af: torch.Tensor, vf: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 允许某些模态为 None（用 0 向量代替）
        B = None
        if tf is not None:
            B = tf.size(0)
        elif af is not None:
            B = af.size(0)
        elif vf is not None:
            B = vf.size(0)
        else:
            raise ValueError("All modalities are None")
        if tf is None:
            tf = tf.new_zeros((B, self.ln_t.normalized_shape[0])) if af is not None else torch.zeros((B, self.ln_t.normalized_shape[0]), device=af.device if af is not None else vf.device)
        if af is None:
            af = torch.zeros((B, self.ln_a.normalized_shape[0]), device=tf.device)
        if vf is None:
            vf = torch.zeros((B, self.ln_v.normalized_shape[0]), device=tf.device)

        t = self.ln_t(tf)
        a = self.ln_a(af)
        v = self.ln_v(vf)
        s_t = self.tower_t(t)  # (B,1)
        s_a = self.tower_a(a)  # (B,1)
        s_v = self.tower_v(v)  # (B,1)
        scores3 = torch.cat([s_t, s_a, s_v], dim=-1)  # (B,3)
        logits = self.head(scores3)  # (B, A)
        return {"logits": logits, "scores3": scores3}

class GatedTriTowerSelector(BaseSelector):
    """
    三塔门控 selector（RaMRA 的 MRD）。

    推荐结构（cfg 同时给出 d_text/d_audio/d_video 时启用）：
      1) 三个**相互独立**的模态塔（每个塔只吃自己的模态）：
             h_m = φ(LN(W_m x_m + b_m))                      （论文 Eq.2）
      2) 乘性门控，逐维度抑制不可靠通道：
             g_m = σ(W^g_m x_m + b^g_m),  h̃_m = g_m ⊙ h_m     （论文 Eq.3-4）
      3) 另加一路**全局上下文**编码（只喂给 GateNet，不参与 α 的维度）：
             h_ctx = MLP([x_T ; x_A ; x_V])
         这样 gate 既能看到单模态证据，也能看到跨模态上下文，而三个塔本身不被混合污染。
      4) GateNet + 温度归一化：
             κ = GateNet([h̃_T ; h̃_A ; h̃_V ; h_ctx]) ∈ R^3
             α = softmax(κ / τ)                              （论文 Eq.6）
      5) 融合：h_fused = Σ_m α_m h̃_m,  logits = out(h_fused)

    回退结构（未提供模态维度时）：共享 backbone + 三个线性头，仅为向后兼容。
    """

    def __init__(self, cfg: SelectorConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim
        self.d_text, self.d_audio, self.d_video = cfg.d_text, cfg.d_audio, cfg.d_video
        self.use_modality_inputs = all(
            d is not None and int(d) > 0 for d in (self.d_text, self.d_audio, self.d_video)
        )

        if self.use_modality_inputs:
            self.d_text, self.d_audio, self.d_video = int(self.d_text), int(self.d_audio), int(self.d_video)
            # 1) 三个独立的模态塔
            self.tower_text = self._make_tower(self.d_text, H, cfg.dropout)
            self.tower_audio = self._make_tower(self.d_audio, H, cfg.dropout)
            self.tower_video = self._make_tower(self.d_video, H, cfg.dropout)
            # 2) 逐维乘性门控
            self.gate_text = nn.Linear(self.d_text, H)
            self.gate_audio = nn.Linear(self.d_audio, H)
            self.gate_video = nn.Linear(self.d_video, H)
            # 3) 第四路：全局上下文（跨模态），只提供给 GateNet
            self.ctx_enc = MLP(
                in_dim=cfg.input_dim,
                hidden_dim=H,
                out_dim=H,
                num_layers=max(2, cfg.num_layers),
                dropout=cfg.dropout,
            )
            # 4) GateNet: 4 路 -> 3 个模态 logit
            self.gate_net = nn.Sequential(
                nn.Linear(H * 4, H),
                nn.Tanh(),
                nn.Linear(H, 3),
            )
        else:
            # 回退：旧的共享 backbone 结构
            self.backbone = MLP(
                in_dim=cfg.input_dim,
                hidden_dim=H,
                out_dim=H,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            )
            self.text_head = nn.Linear(H, H)
            self.audio_head = nn.Linear(H, H)
            self.video_head = nn.Linear(H, H)
            self.gate_net = nn.Sequential(
                nn.Linear(H * 3, H),
                nn.Tanh(),
                nn.Linear(H, 3),
            )

        # 5) 输出头
        self.out = nn.Linear(H, cfg.num_actions)

    @staticmethod
    def _make_tower(in_dim: int, hidden: int, dropout: float) -> nn.Module:
        """单模态塔：LN(W x + b) -> 非线性（论文 Eq.2）。"""
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _split_states(self, states: torch.Tensor):
        """从拼接特征中切出三个模态（顺序 = _build_states: text, audio, video）。"""
        d_t, d_a = self.d_text, self.d_audio
        x_t = states[:, :d_t]
        x_a = states[:, d_t:d_t + d_a]
        x_v = states[:, d_t + d_a:d_t + d_a + self.d_video]
        return x_t, x_a, x_v

    def forward(self, states: torch.Tensor, tau: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            states: (B, input_dim) = [text_feat ; audio_feat ; video_feat]
            tau:    温度系数 τ（论文 Eq.6），作用于 gate logits。None / 1.0 等价于原行为。
        Returns:
            dict: {"logits": (B, num_actions), "gate": (B, 3), "gate_logits": (B, 3)}
        """
        tau_val = float(tau) if (tau is not None and float(tau) > 0) else 1.0

        if self.use_modality_inputs:
            x_t, x_a, x_v = self._split_states(states)
            h_t = self.gate_text(x_t).sigmoid() * self.tower_text(x_t)
            h_a = self.gate_audio(x_a).sigmoid() * self.tower_audio(x_a)
            h_v = self.gate_video(x_v).sigmoid() * self.tower_video(x_v)
            h_ctx = self.ctx_enc(states)
            gate_in = torch.cat([h_t, h_a, h_v, h_ctx], dim=-1)   # (B, 4H)
        else:
            h_shared = self.backbone(states)
            h_t = torch.tanh(self.text_head(h_shared))
            h_a = torch.tanh(self.audio_head(h_shared))
            h_v = torch.tanh(self.video_head(h_shared))
            gate_in = torch.cat([h_t, h_a, h_v], dim=-1)          # (B, 3H)

        gate_logits = self.gate_net(gate_in)                      # (B, 3)
        gate_weights = F.softmax(gate_logits / tau_val, dim=-1)   # (B, 3) = α

        h_stack = torch.stack([h_t, h_a, h_v], dim=1)             # (B, 3, H)
        h_fused = (gate_weights.unsqueeze(-1) * h_stack).sum(dim=1)

        logits = self.out(h_fused)                                # (B, num_actions)

        return {
            "logits": logits,
            "gate": gate_weights,
            "gate_logits": gate_logits,
        }


# Registry for future models
SELECTOR_REGISTRY = {
    "MLP": MLPSelector,
    "TriTower": TriTowerSelector,
    "GatedTriTower": GatedTriTowerSelector,
}


def build_selector(name: str, cfg: SelectorConfig) -> nn.Module:
    if name not in SELECTOR_REGISTRY:
        raise KeyError(f"Unknown selector model: {name}. Available: {list(SELECTOR_REGISTRY.keys())}")
    return SELECTOR_REGISTRY[name](cfg)