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


# Registry for future models
SELECTOR_REGISTRY = {
    "MLP": MLPSelector,
}


def build_selector(name: str, cfg: SelectorConfig) -> nn.Module:
    if name not in SELECTOR_REGISTRY:
        raise KeyError(f"Unknown selector model: {name}. Available: {list(SELECTOR_REGISTRY.keys())}")
    return SELECTOR_REGISTRY[name](cfg)