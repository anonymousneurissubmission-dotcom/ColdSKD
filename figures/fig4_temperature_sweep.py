#!/usr/bin/env python3
"""
Figure 3: Temperature sweep -- two panels side by side.
  Left:  No-LS models (V1) -- smooth response
  Right: LS-trained models -- U-shaped AUROC

Uses precomputed logit decomposition data or computes from pooled features + heads.

Usage:
    source dl_env/bin/activate
    python fig3_temperature_sweep.py
"""
import numpy as np
import torch
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import POOLED_FEATURES, FIGURES_DIR, ensure_dirs
ensure_dirs()

POOLED = POOLED_FEATURES / "imagenet1k"
_BUNDLED_TSWEEP = ROOT / "trained_heads" / "imagenet1k_tsweep"
TSWEEP = Path(os.environ.get(
    "COLDSKD_TSWEEP_DIR", str(_BUNDLED_TSWEEP)))
OUT_DIR = FIGURES_DIR
OOD_DATASETS = ["openimage_o", "inaturalist", "imagenet_o", "dtd", "ninco", "ssb_hard"]

TDIRS = ["T0p10", "T0p25", "T0p50", "T0p75", "T1p00",
         "T1p50", "T2p00", "T3p00", "T4p00", "T6p00", "T8p00"]
T_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

NO_LS_MODELS = ["resnet50_v1", "resnet101_v1", "resnet152_v1"]
LS_MODELS = ["resnet50_v2", "resnet101_v2", "resnet152_v2",
             "swin_t_v1", "swin_t_v2",
             "efficientnet_b1_v1",
             "regnet_y_16gf_v2",
             "inception_v3",
             "vit_b_16_v1", "vit_b_16_v2"]

MODEL_LABELS = {
    "resnet50_v1":        "RN50-V1",
    "resnet101_v1":       "RN101-V1",
    "resnet152_v1":       "RN152-V1",
    "resnet50_v2":        "RN50-V2",
    "resnet101_v2":       "RN101-V2",
    "resnet152_v2":       "RN152-V2",
    "swin_t_v1":          "Swin-T-V1",
    "swin_t_v2":          "Swin-T-V2",
    "efficientnet_b1_v1": "EffNet-B1",
    "regnet_y_16gf_v2":   "RegNet-V2",
    "inception_v3":       "Inc-V3",
    "vit_b_16_v1":        "ViT-B16-V1",
    "vit_b_16_v2":        "ViT-B16-SWAG",
}

MODEL_COLORS = {
    "resnet50_v1":  "#1f77b4", "resnet50_v2":  "#1f77b4",
    "resnet101_v1": "#ff7f0e", "resnet101_v2": "#ff7f0e",
    "resnet152_v1": "#2ca02c", "resnet152_v2": "#2ca02c",
    "swin_t_v1":    "#d62728", "swin_t_v2":    "#e377c2",
    "efficientnet_b1_v1": "#9467bd",
    "regnet_y_16gf_v2":   "#8c564b",
    "inception_v3":       "#bcbd22",
    "vit_b_16_v1":        "#17becf",
    "vit_b_16_v2":        "#7f7f7f",
}

MODEL_MARKERS = {
    "resnet50_v1":  "o", "resnet50_v2":  "o",
    "resnet101_v1": "s", "resnet101_v2": "s",
    "resnet152_v1": "^", "resnet152_v2": "^",
    "swin_t_v1":    "D", "swin_t_v2":    "d",
    "efficientnet_b1_v1": "v",
    "regnet_y_16gf_v2":   "P",
    "inception_v3":       "X",
    "vit_b_16_v1":        "h",
    "vit_b_16_v2":        "H",
}

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

def load_head(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["weight"].numpy().astype(np.float32), d["bias"].numpy().astype(np.float32)

def auroc_fn(id_s, ood_s):
    Ni, No = len(id_s), len(ood_s)
    s = np.concatenate([id_s.astype(np.float64), ood_s.astype(np.float64)])
    l = np.concatenate([np.ones(Ni, np.int32), np.zeros(No, np.int32)])
    o = np.argsort(-s, kind="stable"); sl, ss = l[o], s[o]
    tp, fp = np.cumsum(sl), np.cumsum(1 - sl)
    c = np.where(np.concatenate([ss[:-1] != ss[1:], [True]]))[0]
    fpr = np.concatenate([[0.], fp[c] / No])
    tpr = np.concatenate([[0.], tp[c] / Ni])
    return float(_trapz(tpr, fpr))

def compute_tsweep(model):
    """Compute MLS AUROC at each T for a model."""
    z_id = np.load(POOLED / model / "imagenet1k_val_emb.npy").astype(np.float32)
    z_oods = []
    for od in OOD_DATASETS:
        p = POOLED / "ood" / od / model / f"{od}_emb.npy"
        if p.exists():
            z_oods.append(np.load(p).astype(np.float32))
    z_ood = np.concatenate(z_oods)

    results = {}

    ce_path = TSWEEP / model / "ce_only" / "final.pth"
    if ce_path.exists():
        W, b = load_head(ce_path)
        mls_id = (z_id @ W.T + b).max(axis=1)
        mls_ood = (z_ood @ W.T + b).max(axis=1)
        results["ce_only"] = auroc_fn(mls_id, mls_ood)

    for td, tv in zip(TDIRS, T_VALUES):
        fp = TSWEEP / model / td / "final.pth"
        if not fp.exists():
            continue
        W, b = load_head(fp)
        mls_id = (z_id @ W.T + b).max(axis=1)
        mls_ood = (z_ood @ W.T + b).max(axis=1)
        results[tv] = auroc_fn(mls_id, mls_ood)

    return results

def make_plot(all_data):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.patch.set_facecolor("white")

    all_vals = []
    for mn, data in all_data.items():
        all_vals.extend([v for k, v in data.items() if isinstance(k, (int, float))])
        if "ce_only" in data:
            all_vals.append(data["ce_only"])
    y_min = min(all_vals) - 0.02
    y_max = max(all_vals) + 0.02

    for ax, models, title in [
        (ax_l, NO_LS_MODELS, "No Label Smoothing (V1)"),
        (ax_r, LS_MODELS, "LS-Trained Models"),
    ]:
        ax.set_facecolor("white")

        for mn in models:
            if mn not in all_data:
                continue
            data = all_data[mn]
            label = MODEL_LABELS.get(mn, mn)
            color = MODEL_COLORS.get(mn, "gray")
            marker = MODEL_MARKERS.get(mn, "o")

            ts = [tv for tv in T_VALUES if tv in data]
            aurocs = [data[tv] for tv in ts]
            ax.plot(ts, aurocs, f"-{marker}", color=color, lw=2, ms=5,
                    label=label, zorder=3, markeredgecolor="white",
                    markeredgewidth=0.5)

            if "ce_only" in data:
                ax.axhline(data["ce_only"], color=color, ls="--", lw=1.0,
                           alpha=0.5, zorder=2)

        ax.axvline(0.5, color="#cccccc", ls=":", lw=0.8, zorder=1)
        ax.axvline(1.0, color="#cccccc", ls=":", lw=0.8, zorder=1)
        ax.text(0.5, y_min + 0.002, "T=0.5", ha="center", fontsize=7,
                color="#999999")
        ax.text(1.0, y_min + 0.002, "T=1", ha="center", fontsize=7,
                color="#999999")

        ax.set_xlabel("Distillation Temperature $T$")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, 8.5)
        ax.grid(True, alpha=0.15, linestyle="--")

        if ax == ax_l:
            ax.legend(loc="lower right", frameon=True, facecolor="white",
                      edgecolor="#CCCCCC", framealpha=0.95)
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=True, facecolor="white",
                      edgecolor="#CCCCCC", framealpha=0.95,
                      borderpad=0.5)

    ax_l.set_ylabel("MLS AUROC (mean over 6 OOD datasets)")

    ax_r.axvspan(1.0, 2.0, alpha=0.08, color="#d62728", zorder=0)
    ax_r.annotate("Collapse Zone\n($1.0 < T < 2.0$)",
                  xy=(3.0, y_min + 0.01), xytext=(3., y_min + 0.01),
                  fontsize=8, color="#d62728", fontstyle="italic",
                  ha="center", va="bottom",
                  bbox=dict(boxstyle="round,pad=0.3",
                            facecolor="#fff0f0", edgecolor="#d62728",
                            alpha=0.9))

    ax_r.axvspan(0.0, 1.0, alpha=0.08, color="#1f77b4", zorder=0)
    ax_r.annotate("Cold Distillation\n($0.0 < T < 1.0$)",
                  xy=(0.5, y_max - 0.01), xytext=(0.5, y_max - 0.01),
                  fontsize=8, color="#1f77b4", fontstyle="italic",
                  ha="center", va="top",
                  bbox=dict(boxstyle="round,pad=0.3",
                            facecolor="#f0f0ff", edgecolor="#1f77b4",
                            alpha=0.9))

    fig.suptitle("Temperature Sweep: Self-Distillation AUROC vs $T$\n"
                 "Dashed lines = CE-only (AugDelete) baseline per model",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig3_temperature_sweep.pdf"
    out_png = OUT_DIR / "fig3_temperature_sweep.png"
    fig.savefig(out_pdf, facecolor="white")
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")

if __name__ == "__main__":
    all_data = {}
    for mn in NO_LS_MODELS + LS_MODELS:
        if not (POOLED / mn).exists() or not (TSWEEP / mn).exists():
            print(f"  {mn}: SKIPPED")
            continue
        data = compute_tsweep(mn)
        all_data[mn] = data
        ce = data.get("ce_only", float("nan"))
        best_t = max(T_VALUES, key=lambda t: data.get(t, 0))
        best_v = data.get(best_t, 0)
        print(f"  {mn}: CE={ce:.4f}  best T={best_t} AUROC={best_v:.4f}")

    make_plot(all_data)
    print("\nDone.")
