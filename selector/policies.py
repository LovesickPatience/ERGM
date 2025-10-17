# -*- coding: utf-8 -*-
"""Policies for action sampling and constraints.
- DiscretePolicy: over a finite action set; supports temperature, sampling, and log-probs.
- ContinuousSimplex: projects to simplex and supports straight-through sampling (optional).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class DiscretePolicyOutput:
    logits: torch.Tensor   # (B, A)
    probs: torch.Tensor    # (B, A)


class DiscretePolicy:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> DiscretePolicyOutput:
        if self.temperature != 1.0:
            logits = logits / self.temperature
        probs = F.softmax(logits, dim=-1)
        return DiscretePolicyOutput(logits=logits, probs=probs)

    @staticmethod
    def sample(probs: torch.Tensor) -> torch.Tensor:
        """Returns actions (B,) sampled from probs."""
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def greedy(probs: torch.Tensor) -> torch.Tensor:
        return probs.argmax(dim=-1)

    @staticmethod
    def log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def log_probs_pair(logits: torch.Tensor,
                       a_pos: torch.Tensor,
                       a_neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return log-probs for two action tensors under the same logits.

        Args:
            logits: (B, A)
            a_pos: (B,) preferred action indices (int64/long)
            a_neg: (B,) dispreferred action indices (int64/long)
        Returns:
            lp_pos, lp_neg: both shape (B,)
        """
        if a_pos.dtype != torch.long:
            a_pos = a_pos.long()
        if a_neg.dtype != torch.long:
            a_neg = a_neg.long()

        logp = F.log_softmax(logits, dim=-1)  # (B, A)
        lp_pos = logp.gather(-1, a_pos.unsqueeze(-1)).squeeze(-1)  # (B,)
        lp_neg = logp.gather(-1, a_neg.unsqueeze(-1)).squeeze(-1)  # (B,)
        return lp_pos, lp_neg


class ContinuousSimplex:
    """Utility for continuous weights over K modalities.
    We use softmax for projection. Sampling is optional and not typically needed.
    """
    @staticmethod
    def project(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature != 1.0:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)

    @staticmethod
    def gumbel_softmax_sample(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)