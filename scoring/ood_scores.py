"""
Canonical OOD scoring functions.

All scores take pre-computed logits (v), embeddings (z), and head parameters (W, b).
Combination scores use z-score normalization (zero mean, unit variance) by default.
Logit normalization (LN) operates on the logit vector v only -- never on embeddings or weights.

Usage:
    from ood_scores import compute_all_scores, auroc, fpr95

    scores_id = compute_all_scores(logits_id, z_id, W, b)
    scores_ood = compute_all_scores(logits_ood, z_ood, W, b)

    for name in scores_id:
        print(f"{name}: AUROC={auroc(scores_id[name], scores_ood[name]):.4f}")
"""

import numpy as np
from scipy.special import logsumexp

def score_mls(logits):
    """Maximum Logit Score: max_k v_k"""
    return logits.max(axis=1)

def score_energy(logits):
    """Energy score: log sum_k exp(v_k)"""
    return logsumexp(logits, axis=1)

def score_cosine(z, W):
    """True cosine similarity: max_k cos(z, w_k) = max_k (z * w_k) / (||z||*||w_k||).

    Both the embedding z and each weight row w_k are L2-normalized, so this
    measures angular similarity independent of magnitude. This is the cosine
    that goes into CSKD-DC = z(cosine) + z(energy).
    """
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return (zn @ Wn.T).max(axis=1)

def score_max_proj(z, W):
    """Maximum raw projection: max_k z * w_k (no normalization).

    Equivalent to MLS without the bias. Kept for diagnostics; CSKD-DC uses
    score_cosine (true angular similarity), not this.
    """
    return (z @ W.T).max(axis=1)

def score_ln_mls(logits):
    """Logit-normalized MLS: max_k v_k / ||v||_2.
    L2 normalization operates on the logit vector only."""
    v_norm = logits / (np.linalg.norm(logits, axis=1, keepdims=True) + 1e-12)
    return v_norm.max(axis=1)

def score_ln_energy(logits):
    """Logit-normalized Energy: logsumexp of v / ||v||_2.
    L2 normalization operates on the logit vector only."""
    v_norm = logits / (np.linalg.norm(logits, axis=1, keepdims=True) + 1e-12)
    return logsumexp(v_norm, axis=1)

def score_msp(logits):
    """Maximum Softmax Probability."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    softmax = exp_v / exp_v.sum(axis=1, keepdims=True)
    return softmax.max(axis=1)

def _znorm(x):
    """Z-score normalization: (x - mean) / std."""
    return (x - x.mean()) / (x.std() + 1e-12)

def _minmax(x):
    """Min-max normalization: (x - min) / (max - min)."""
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def score_cosene(cos_scores, ene_scores, norm="zscore"):
    """Cos+Energy combination score (CSKD-DC when norm='zscore').

    Combines cosine alignment and energy into a single score.
    Both are normalized to comparable scales before summing.

    Args:
        cos_scores: cosine scores (all samples: ID + OOD concatenated)
        ene_scores: energy scores (all samples: ID + OOD concatenated)
        norm: "zscore" (default) or "minmax"

    Returns:
        combined scores (same length as inputs)
    """
    if norm == "zscore":
        return _znorm(cos_scores) + _znorm(ene_scores)
    elif norm == "minmax":
        return _minmax(cos_scores) + _minmax(ene_scores)
    else:
        raise ValueError(f"Unknown norm: {norm}. Use 'zscore' or 'minmax'.")

def score_cskd_dc(cos_scores, ene_scores):
    """CSKD-DC: z_score(cosine) + z_score(energy). Alias for score_cosene(norm='zscore')."""
    return score_cosene(cos_scores, ene_scores, norm="zscore")

def score_mls_cos(mls_scores, cos_scores):
    """MLS+cosine: z_score(MLS) + z_score(cosine).

    Sibling of CSKD-DC that uses the max logit instead of energy as the
    magnitude-driven component. cosine should be true cos(z, w_k) -- see
    score_cosine.
    """
    return _znorm(mls_scores) + _znorm(cos_scores)

def compute_all_scores(logits, z, W, b=None):
    """Compute all individual OOD scores for a set of samples.

    Args:
        logits: (N, K) pre-softmax outputs. If None, computed from z, W, b.
        z: (N, D) penultimate embeddings.
        W: (K, D) classification head weights.
        b: (K,) classification head biases. If None, assumed zero.

    Returns:
        dict of score_name -> (N,) arrays
    """
    if logits is None:
        logits = z @ W.T
        if b is not None:
            logits = logits + b

    return {
        "mls": score_mls(logits),
        "energy": score_energy(logits),
        "cosine": score_cosine(z, W),
        "ln_mls": score_ln_mls(logits),
        "ln_energy": score_ln_energy(logits),
        "msp": score_msp(logits),
    }

def compute_cosene(scores_id, scores_ood, norm="zscore"):
    """Compute Cos+Energy combination from pre-computed individual scores.

    Args:
        scores_id: dict from compute_all_scores (ID samples)
        scores_ood: dict from compute_all_scores (OOD samples)
        norm: "zscore" (default) or "minmax"

    Returns:
        (cosene_id, cosene_ood) score arrays
    """
    cos_all = np.concatenate([scores_id["cosine"], scores_ood["cosine"]])
    ene_all = np.concatenate([scores_id["energy"], scores_ood["energy"]])
    N_id = len(scores_id["cosine"])

    combined = score_cosene(cos_all, ene_all, norm=norm)
    return combined[:N_id], combined[N_id:]

_trapz = getattr(np, "trapezoid", None) or np.trapz

def auroc(id_scores, ood_scores):
    """Area Under ROC Curve. Higher ID scores = better."""
    Ni, No = len(id_scores), len(ood_scores)
    s = np.concatenate([id_scores, ood_scores]).astype(np.float64)
    l = np.concatenate([np.ones(Ni), np.zeros(No)])
    o = np.argsort(-s, kind="stable")
    sl, ss = l[o], s[o]
    tp, fp = np.cumsum(sl), np.cumsum(1 - sl)
    c = np.where(np.concatenate([ss[:-1] != ss[1:], [True]]))[0]
    fpr = np.concatenate([[0.], fp[c] / No])
    tpr = np.concatenate([[0.], tp[c] / Ni])
    return float(_trapz(tpr, fpr))

def fpr95(id_scores, ood_scores):
    """False Positive Rate at 95% True Positive Rate."""
    Ni, No = len(id_scores), len(ood_scores)
    s = np.concatenate([id_scores, ood_scores]).astype(np.float64)
    l = np.concatenate([np.ones(Ni), np.zeros(No)])
    o = np.argsort(-s, kind="stable")
    sl, ss = l[o], s[o]
    tp, fp = np.cumsum(sl), np.cumsum(1 - sl)
    c = np.where(np.concatenate([ss[:-1] != ss[1:], [True]]))[0]
    fpr_a = np.concatenate([[0.], fp[c] / No])
    tpr_a = np.concatenate([[0.], tp[c] / Ni])
    i = np.searchsorted(tpr_a, 0.95, side="left")
    if i >= len(fpr_a):
        return float(fpr_a[-1])
    if i > 0 and tpr_a[i] != tpr_a[i - 1]:
        a = (0.95 - tpr_a[i - 1]) / (tpr_a[i] - tpr_a[i - 1])
        return float(fpr_a[i - 1] + a * (fpr_a[i] - fpr_a[i - 1]))
    return float(fpr_a[i])
