"""
ASH (Activation Shaping) variants -- Djurisic et al., ICLR 2023.

Operates on a frozen pooled embedding z  in  R^D (post-ReLU). Three flavours:
    ASH-P : prune everything below the percentile p (zero out small values)
    ASH-B : prune below percentile p, then binarize survivors to a constant
    ASH-S : prune below percentile p, then rescale survivors so their sum
            matches the original L1 sum (preserving energy)

The percentile threshold is chosen *per-sample* (the original paper) -- i.e.
each row of z is shaped against its own quantile.

After shaping, score with any standard logit-based detector
(MLS / Energy / Cosine / CSKD-DC).
"""
import numpy as np

def _percentile_threshold(z, p):
    """Per-row p-th percentile (returns shape (N, 1))."""
    return np.percentile(z, p, axis=1, keepdims=True)

def ash_p(z, p=90.0):
    """Prune below the per-sample p-th percentile."""
    thr = _percentile_threshold(z, p)
    return np.where(z >= thr, z, 0.0).astype(z.dtype)

def ash_b(z, p=90.0):
    """Prune below percentile p; binarize survivors so each row's L1 mass
    equals the original L1 sum."""
    thr = _percentile_threshold(z, p)
    keep = z >= thr
    n_keep = keep.sum(axis=1, keepdims=True).clip(min=1)
    s_orig = np.abs(z).sum(axis=1, keepdims=True)
    fill = s_orig / n_keep
    return np.where(keep, fill, 0.0).astype(z.dtype)

def ash_s(z, p=90.0):
    """Prune below percentile p; scale survivors so total L1 is preserved."""
    thr = _percentile_threshold(z, p)
    keep = z >= thr
    s_orig = np.abs(z).sum(axis=1, keepdims=True)
    s_keep = np.where(keep, z, 0.0).sum(axis=1, keepdims=True)
    scale = np.where(s_keep > 0, s_orig / s_keep, 1.0)
    return (np.where(keep, z, 0.0) * scale).astype(z.dtype)

SHAPINGS = {
    "ash_p": ash_p,
    "ash_b": ash_b,
    "ash_s": ash_s,
}

def apply(z, name="ash_s", p=90.0):
    """Apply an ASH shaping by name. Returns reshaped embeddings of same shape."""
    if name not in SHAPINGS:
        raise ValueError(f"Unknown ASH variant: {name}. Choose from {list(SHAPINGS)}")
    return SHAPINGS[name](z, p=p)
