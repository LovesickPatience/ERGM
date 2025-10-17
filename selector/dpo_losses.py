# -*- coding: utf-8 -*-
"""DPO-style losses for selector training.
We assume a *discrete* action set with probabilities from a DiscretePolicy.
For each (state s), we have preferred action a+ and dispreferred a-.

Loss: -log sigma( beta * ( log pi(a+|s) - log pi(a-|s) ) )
"""
from typing import Dict
import torch
import torch.nn.functional as F


def dpo_loss(logp_pos: torch.Tensor, logp_neg: torch.Tensor, beta: float = 0.1, reduce: str = "mean") -> torch.Tensor:
    """
    Args:
        logp_pos: (B,) log-prob of preferred actions
        logp_neg: (B,) log-prob of dispreferred actions
        beta: temperature scaling
        reduce: mean or sum
    Returns:
        scalar loss
    """
    assert logp_pos.shape == logp_neg.shape
    margin = beta * (logp_pos - logp_neg)
    # -log sigmoid(margin)
    loss = F.softplus(-margin)  # = -log(sigmoid(margin))
    if reduce == "mean":
        return loss.mean()
    elif reduce == "sum":
        return loss.sum()
    else:
        return loss