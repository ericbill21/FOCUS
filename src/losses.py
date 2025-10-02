from typing import List, Optional, Tuple, Union

import math
from tabulate import tabulate

import numpy as np

import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses


def controller_loss(emb, label_groups) -> torch.Tensor:
    def jensen_shannon_divergence(Q: torch.Tensor) -> torch.Tensor:
        eps = 1e-10

        T, _ = Q.shape

        # Empirically the distributions have to many almost-zero values
        # which domintate the KL divergence. To mitigate this, we cube
        # the values to make the distribution more peaky. (This is a pure empirical fix.)
        Q = Q**3
        Q = Q / Q.sum(dim=-1, keepdim=True).clamp_min(eps)

        M = Q.mean(dim=0, keepdim=True)
        logQ = (Q + eps).log()
        logM = (M + eps).log()

        KL = (Q * (logQ - logM)).sum(dim=-1).mean() 
        return KL / math.log(T)

    js_pos = 0.0
    js_neg = 0.0
        
    Q = []
    n_js_pos = 0
    for group in label_groups:
        # Get prob distribution for each token in the group
        P = emb[group]

        # Intra-group Coherence: minimize the Jensen-Shannon divergence
        # between the probability distributions of the tokens in the group
        if len(group) > 1:
            js_pos += jensen_shannon_divergence(P)
            n_js_pos += 1

        # Claculate the mixture distribution for the group
        p_mix = torch.mean(P, dim=0)
        Q.append(p_mix)

    # Inter-group Separation: maximize the Jensen-Shannon divergence
    # between the probability distributions of the subject groups
    if len(Q) > 1:
        Q = torch.stack(Q, dim=0)
        js_neg += 1 - jensen_shannon_divergence(Q)

    # Normalize the loss components
    if n_js_pos > 0:
        js_pos /= n_js_pos

    return (js_pos + js_neg)/2


def conform_loss(emb, prev_emb, label_groups) -> torch.Tensor:
    if prev_emb is None or len(prev_emb) == 0:
        return emb.sum() * 0.0

    X = torch.cat([emb, prev_emb], dim=0)

    label_map = {}
    idxs = torch.unique(torch.tensor([i for g in label_groups for i in g], device=emb.device)).long()
    for gi, g in enumerate(label_groups):
        for idx in g:
            label_map[int(idx)] = gi
    y = torch.tensor([label_map[int(i)] for i in idxs.tolist()], device=emb.device, dtype=torch.long)
    y = torch.cat([y, y], dim=0)

    ntx = losses.NTXentLoss()
    return ntx(X, y)


def attend_and_excite_loss(emb, label_groups, tau=1.0) -> torch.Tensor:
    attn = emb / emb.sum(dim=0, keepdim=True)

    group_maxes = []
    for group in label_groups:
        g_map = attn[group].mean(dim=0)
        group_maxes.append(g_map.max())

    m = torch.stack(group_maxes).min()
    return F.relu(tau - m)


def divide_and_bind_loss(emb, label_groups) -> torch.Tensor:
    HW = emb.shape[-1]

    dim = int(math.sqrt(HW))
    assert dim * dim == HW
    attn_hw = emb.view(emb.shape[0], dim, dim)

    tv_vals = []
    for g in label_groups:
        g_map = attn_hw[g].mean(dim=0) # (H, W)
        g_map = (g_map / g_map.sum()) * HW

        tv_h = (g_map[1:, :] - g_map[:-1, :]).abs().mean()
        tv_w = (g_map[:, 1:] - g_map[:, :-1]).abs().mean()
        tv_vals.append(tv_h + tv_w)

    return -1.0 * torch.stack(tv_vals).min()

def jedi_loss(emb, label_groups, block_upper=16, block_lower=4, lambda_reg=0.01) -> torch.Tensor:
    eps = 1e-10

    def jensen_shannon_divergence(Q: torch.Tensor) -> torch.Tensor:
        # Empirically the distributions have to many almost-zero values
        # which domintate the KL divergence. To mitigate this, we cube
        # the values to make the distribution more peaky. (This is a pure empirical fix.)
        Q = Q**3

        _, T, _ = Q.shape
        prob_Q = Q / Q.sum(dim=-1, keepdim=True)
        log_Q = torch.log(prob_Q + eps)
        prob_M = torch.mean(prob_Q, dim=1, keepdim=True)
        log_M = torch.log(prob_M + eps)

        KL = torch.sum(prob_Q * (log_Q - log_M), dim=-1)
        return torch.mean(KL, dim=-1) / math.log(T)

    def entropy(Q: torch.Tensor) -> torch.Tensor:
        # Empirically the distributions have to many almost-zero values
        # which domintate the KL divergence. To mitigate this, we cube
        # the values to make the distribution more peaky. (This is a pure empirical fix.)
        Q = Q**3

        prob_Q = Q / Q.sum(dim=1, keepdim=True)
        log_Q = torch.log(prob_Q + eps)

        return -1.0 * torch.sum(prob_Q * log_Q, dim=-1) / math.log(Q.shape[-1])

    device = emb.device
    num_blocks = block_upper - block_lower
    js_neg = torch.zeros(num_blocks, device=device)
    js_pos = torch.zeros(num_blocks, device=device)
    reg    = torch.zeros(num_blocks, device=device)

    Q = []
    n_js_pos = 0
    for group in label_groups:
        P = torch.stack([emb[block_lower:block_upper, tok] for tok in group], dim=1)
        if P.shape[1] > 1:
            js_pos += jensen_shannon_divergence(P)
            n_js_pos += 1

        p_mix = torch.mean(P, dim=1)
        Q.append(p_mix)
        reg += 1 - entropy(p_mix)

    if len(Q) > 1:
        Q = torch.stack(Q, dim=1)
        js_neg += 1 - jensen_shannon_divergence(Q)

    # Normalize the loss components
    if n_js_pos > 0:
        js_pos /= n_js_pos 
    reg /= len(label_groups)

    loss_per_block = js_pos + js_neg + lambda_reg * reg
    return loss_per_block.mean()