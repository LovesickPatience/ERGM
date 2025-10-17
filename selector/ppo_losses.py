# -*- coding: utf-8 -*-
"""PPO-style clipped surrogate for selector training (discrete actions).
We use advantages (A) computed from rewards with a simple baseline.
"""
from typing import Tuple
import torch
import torch.nn.functional as F


def ppo_clipped_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    reduce: str = "mean",
) -> torch.Tensor:
    """Compute PPO clipped surrogate loss (to minimize => negate objective).
    Args:
        logp_new, logp_old: (B,) log-prob of taken actions under new/old policy
        advantages: (B,)
    """
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    obj = torch.min(unclipped, clipped)
    # We minimize negative objective
    if reduce == "mean":
        return -obj.mean()
    elif reduce == "sum":
        return -obj.sum()
    else:
        return -obj


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    ent = -(probs * logp).sum(-1)
    return ent