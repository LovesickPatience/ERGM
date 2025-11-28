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


# Registry for future models
SELECTOR_REGISTRY = {
    "MLP": MLPSelector,
}


def build_selector(name: str, cfg: SelectorConfig) -> nn.Module:
    if name not in SELECTOR_REGISTRY:
        raise KeyError(f"Unknown selector model: {name}. Available: {list(SELECTOR_REGISTRY.keys())}")
    return SELECTOR_REGISTRY[name](cfg)