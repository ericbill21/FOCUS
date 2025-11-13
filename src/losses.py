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

def enmmdt_loss(emb: torch.Tensor, label_groups) -> torch.Tensor:
    """
    Enhancing MMDiT (Wei et al., 2024) ambiguity loss:
      L = λba * Lba + λta * Lta + λol * Lol
    - Lba (block alignment): align early blocks to later blocks per encoder (late maps are detached as targets).
    - Lta (text-encoder alignment): align CLIP and T5 late maps.
    - Lol (overlap): penalize pairwise overlap between different subjects' normalized maps.
    
    Supported emb layouts:
      [T, HW]
      [B, T, HW]
      [E, B, T, HW]   (E>=2 enables text-encoder alignment)
    
    label_groups: List[List[int]] — token indices per subject (same style as your other losses).
    """
    eps = 1e-12
    device = emb.device

    def cos1(a, b):
        # flatten to 1D, cosine similarity
        return F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0, eps=eps)

    # --- standardize dims to [E, B, T, HW] ---
    if emb.dim() == 2:        # [T, HW]
        emb = emb.unsqueeze(0).unsqueeze(0)  # [1,1,T,HW]
    elif emb.dim() == 3:      # [B, T, HW]
        emb = emb.unsqueeze(0)               # [1,B,T,HW]
    elif emb.dim() == 4:      # [E, B, T, HW]
        pass
    else:
        raise ValueError(f"enmmdt_loss: unsupported emb shape {tuple(emb.shape)}")

    E, B, T, HW = emb.shape

    # Split blocks into "early" and "late" (paper uses 5–8 vs 9–12; fall back if B is small)  :contentReference[oaicite:1]{index=1}
    if B >= 12:
        early_idx = torch.arange(4, min(8, B), device=device)  # 0-based 4..7  => blocks 5..8
        late_idx  = torch.arange(8, min(12, B), device=device) # 0-based 8..11 => blocks 9..12
    else:
        mid = max(1, B // 2)
        early_idx = torch.arange(0, mid, device=device)
        late_idx  = torch.arange(mid, B, device=device)
        if late_idx.numel() == 0:  # degenerate single-block case
            late_idx = early_idx

    # Build subject-level maps per encoder and block: A[e, b, s, HW]
    S = len(label_groups)
    if S == 0:
        return emb.sum() * 0.0

    subjects = []
    idx_tensor = torch.arange(T, device=device)
    for g in label_groups:
        g_idx = torch.as_tensor(g, device=device, dtype=torch.long)
        # guard invalid indices
        g_idx = g_idx[(g_idx >= 0) & (g_idx < T)]
        if g_idx.numel() == 0:
            # empty group -> use zeros to avoid NaNs
            subjects.append(torch.zeros(E, B, HW, device=device, dtype=emb.dtype))
        else:
            # average token maps for this subject
            # emb: [E,B,T,HW] -> select tokens -> mean over token dim -> [E,B,HW]
            subj_maps = emb[:, :, g_idx, :].mean(dim=2)
            subjects.append(subj_maps)
    # stack subjects: [E,B,S,HW]
    A = torch.stack(subjects, dim=2)

    # Aggregate early/late per subject & encoder: [E,S,HW]
    A_early = A[:, early_idx, :, :].mean(dim=1) if early_idx.numel() > 0 else A.mean(dim=1)
    A_late  = A[:,  late_idx, :, :].mean(dim=1) if  late_idx.numel() > 0 else A.mean(dim=1)

    # -----------------------
    # Block Alignment Loss Lba (Eq. 1): detach late maps as targets  :contentReference[oaicite:2]{index=2}
    # -----------------------
    Lba_terms = []
    if B > 1:
        for e in range(E):
            for s in range(S):
                ce = 1.0 - cos1(A_early[e, s], A_late[e, s].detach())
                Lba_terms.append(ce)
        if len(Lba_terms) > 0:
            # paper uses 1/(2N) when E=2; here average by (#encoders_used * N) for stability
            Lba = torch.stack(Lba_terms).mean()
        else:
            Lba = emb.sum() * 0.0
    else:
        Lba = emb.sum() * 0.0  # no block structure → skip

    # -----------------------
    # Text-Encoder Alignment Loss Lta (Eq. 2): late CLIP vs late T5  :contentReference[oaicite:3]{index=3}
    # -----------------------
    if E >= 2:
        # take the first two encoders as (CLIP, T5)
        e0, e1 = 0, 1
        Lta_terms = [(1.0 - cos1(A_late[e0, s], A_late[e1, s])) for s in range(S)]
        Lta = torch.stack(Lta_terms).mean() if len(Lta_terms) > 0 else emb.sum() * 0.0
    else:
        Lta = emb.sum() * 0.0

    # -----------------------
    # Overlap Loss Lol (Eq. 3): pairwise overlap of normalized maps  :contentReference[oaicite:4]{index=4}
    # -----------------------
    # Normalize late maps per (e,s)
    Ahat = A_late.clamp_min(0)
    Ahat = Ahat / (Ahat.sum(dim=-1, keepdim=True).clamp_min(eps))  # [E,S,HW], L1-normalized

    pair_terms = []
    if S >= 2:
        for i in range(S):
            for j in range(i + 1, S):
                if E >= 2:
                    # 4 terms: clip-clip, t5-t5, clip-t5, t5-clip  (use first two encoders)
                    e0, e1 = 0, 1
                    pair_terms.append(torch.dot(Ahat[e0, i], Ahat[e0, j]))
                    pair_terms.append(torch.dot(Ahat[e1, i], Ahat[e1, j]))
                    pair_terms.append(torch.dot(Ahat[e0, i], Ahat[e1, j]))
                    pair_terms.append(torch.dot(Ahat[e1, i], Ahat[e0, j]))
                else:
                    # single-encoder case: just one overlap term
                    pair_terms.append(torch.dot(Ahat[0, i], Ahat[0, j]))
        # average across all pair-terms for scale stability
        Lol = torch.stack(pair_terms).mean()
    else:
        Lol = emb.sum() * 0.0

    # -----------------------
    # Combine with paper weights (Eq. 4)  :contentReference[oaicite:5]{index=5}
    # -----------------------
    lambda_ba = 1.0
    lambda_ta = 0.2
    lambda_ol = 0.001
    return lambda_ba * Lba + lambda_ta * Lta + lambda_ol * Lol

def self_cross_guidance_loss(
    cross: torch.Tensor,               # [T,HW] or [B,T,HW]  (token → patch probs)
    label_groups,                      # List[List[int]]: token ids per subject (already sliced)
    self_attn: torch.Tensor = None,    # [HW,HW] or [B,HW,HW]  (image self-attn). REQUIRED for the main term.
    lambda_cross: float = 0.0,         # Attend&Excite-style cross response weight (Eq. 8)
    use_otsu: bool = True,             # Paper’s masked aggregation (Otsu) (Sec. 4)
    topk_fallback: int = 16,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Self-Cross diffusion guidance loss, faithful to the paper:

      As_i  = weighted aggregation of self-attn rows for patches selected by Otsu on Ac_i  (Eq. 5)
      g(i,j) = sum min(As_i, Ac_j) + sum min(Ac_i, As_j)                                   (Eq. 6)
      S_self-cross = average over pairs (i<j)                                              (Eq. 7)
      L_total = S_self-cross + lambda_cross * S_cross-attn                                 (Eq. 8)

    If `self_attn` is None, we can’t compute S_self-cross; the function then returns only the
    cross response term (useful fallback but not the full paper loss).
    """
    def _ensure_bt(x, kind):
        # cross: [T,HW] → [1,T,HW]; [B,T,HW] stays
        # self : [HW,HW] → [1,HW,HW]; [B,HW,HW] stays
        if x is None: return None
        if x.dim() == 2 and kind == "cross":
            return x.unsqueeze(0)
        if x.dim() == 2 and kind == "self":
            return x.unsqueeze(0)
        if (kind == "cross" and x.dim() == 3) or (kind == "self" and x.dim() == 3):
            return x
        raise ValueError(f"{kind} has invalid shape: {tuple(x.shape)}")

    def _normalize_last(x, dim=-1):
        return x / x.sum(dim=dim, keepdim=True).clamp_min(eps)

    def _otsu_thr_01(v, bins=256):
        # v in [0,1]; compute Otsu threshold (no grad)
        h = torch.histc(v.detach().clamp(0,1), bins=bins, min=0.0, max=1.0)
        p = h / h.sum().clamp_min(1e-12)
        omega = torch.cumsum(p, 0)
        idx = torch.arange(1, bins+1, device=v.device, dtype=v.dtype)
        mu = torch.cumsum(p*idx, 0)
        mu_t = mu[-1].clamp_min(1e-12)
        num = (mu_t*omega - mu).pow(2)
        den = (omega*(1-omega)).clamp_min(1e-12)
        k = torch.argmax(num/den).item()
        return float((k + 0.5) / bins)

    cross = _ensure_bt(cross, "cross")      # [B,T,HW]
    B, T, HW = cross.shape
    cross = _normalize_last(cross.clamp_min(0.0), dim=-1)

    # Attend&Excite style cross response term (Eq. 3–4)
    K_all = []
    for g in label_groups:
        if len(g) > 0:
            K_all.extend(g)
    if len(K_all) > 0:
        K_idx = torch.as_tensor(K_all, device=cross.device, dtype=torch.long)
        Scross_attn = (1.0 - cross[:, K_idx].amax(dim=-1)).max(dim=-1).values.mean()
    else:
        Scross_attn = cross.sum() * 0.0

    # If no self-attention provided, we can’t do the main Self-Cross term → fallback
    if self_attn is None:
        return lambda_cross * Scross_attn

    self_attn = _ensure_bt(self_attn, "self")  # [B,HW,HW]
    if self_attn.shape[0] != B:
        # broadcast if needed
        if self_attn.shape[0] == 1:
            self_attn = self_attn.expand(B, *self_attn.shape[1:])
        else:
            raise ValueError("Batch mismatch between cross and self_attn.")

    # Row-normalize self-attention (each query row sums to 1)
    self_attn = _normalize_last(self_attn.clamp_min(0.0), dim=-1)  # [B,HW,HW]

    def _aggregate_self_for_group(cg: torch.Tensor, S: torch.Tensor):
        """
        cg: [Tg,HW] cross maps for one subject (already prob-normalized)
        S : [HW,HW] row-stochastic self-attn
        returns As: [HW], prob-normalized (Eq. 5)
        """
        g = cg.mean(0)                           # [HW]
        g_min, g_max = g.min(), g.max()
        if (g_max - g_min) < 1e-12:
            return torch.full_like(g, 1.0 / g.numel())  # degenerate: uniform

        g01 = (g - g_min) / (g_max - g_min + eps)
        if use_otsu:
            thr = _otsu_thr_01(g01)
            mask = g01 >= thr
            if mask.sum() == 0:
                k = min(topk_fallback, g.numel())
                mask = torch.zeros_like(g, dtype=torch.bool); mask[torch.topk(g, k).indices] = True
        else:
            k = min(topk_fallback, g.numel())
            mask = torch.zeros_like(g, dtype=torch.bool); mask[torch.topk(g, k).indices] = True

        w = _normalize_last(g[mask].unsqueeze(0), dim=-1).squeeze(0)  # weights over selected rows
        As = (w.unsqueeze(1) * S[mask]).sum(0)                        # weighted sum of rows → [HW]
        return _normalize_last(As.unsqueeze(0), dim=-1).squeeze(0)

    # Build per-subject {Ac_i, As_i}, then compute S_self-cross (Eq. 6–7)
    pair_terms = []
    for b in range(B):
        subj_cross = []
        subj_self  = []
        for g in label_groups:
            if len(g) == 0:
                continue
            idx = torch.as_tensor(g, device=cross.device, dtype=torch.long)
            Ac_i = cross[b, idx]                    # [Tg,HW]
            As_i = _aggregate_self_for_group(Ac_i, self_attn[b])  # [HW]
            subj_cross.append(_normalize_last(Ac_i.mean(0).unsqueeze(0), dim=-1).squeeze(0))
            subj_self.append(As_i)

        S = len(subj_cross)
        if S >= 2:
            # average over unordered pairs i<j (Eq. 7)
            acc, cnt = 0.0, 0
            for i in range(S):
                for j in range(i+1, S):
                    acc += torch.minimum(subj_self[i],  subj_cross[j]).sum() \
                         + torch.minimum(subj_cross[i], subj_self[j]).sum()
                    cnt += 1
            pair_terms.append(acc / max(cnt, 1))
        else:
            pair_terms.append(cross.new_tensor(0.0))

    S_self_cross = torch.stack(pair_terms).mean()
    return S_self_cross + lambda_cross * Scross_attn