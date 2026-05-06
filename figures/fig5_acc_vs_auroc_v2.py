#!/usr/bin/env python3
"""
Figure 1: Accuracy vs MLS AUROC scatter for LS-trained (v2/modern) models.
Shows Pretrained -> AugDelete -> CSKD improvement path.

Adapted from FIGURE_1_PLOT_AUROC_VS_ACC.py for Linux paths.
Uses pooled features + trained heads directly.

Usage:
    source dl_env/bin/activate
    python fig1_auroc_vs_accuracy.py
"""
import os, json, copy, sys
import numpy as np
import torch
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent / "scoring"))
from ood_scores import score_cosine, score_energy, score_cskd_dc  # noqa: E402

import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import POOLED_FEATURES, FIGURES_DIR, ensure_dirs
ensure_dirs()

POOLED = POOLED_FEATURES / "imagenet1k"
_BUNDLED_TSWEEP = ROOT / "trained_heads" / "imagenet1k_tsweep"
_BUNDLED_TTEST = ROOT / "trained_heads" / "imagenet1k_ttest"
TSWEEP = Path(os.environ.get(
    "COLDSKD_TSWEEP_DIR", str(_BUNDLED_TSWEEP)))
TTEST_DIR = Path(os.environ.get(
    "COLDSKD_TTEST_DIR", str(_BUNDLED_TTEST)))
OUT_DIR = FIGURES_DIR

CSKD_LAM_DIR = None
CSKD_LAM_VALUE = None
CSKD_LAM_T = 0.5
STRICT_CSKD = False

MODEL_LIST = [
    "resnet50_v2", "resnet101_v2", "resnet152_v2",
    "inception_v3",
    "efficientnet_b1_v1",
    "swin_t_v1", "swin_t_v2",
    "regnet_y_16gf_v2",
    "vit_b_16_v1", "vit_b_16_v2",
]

V1_MODELS = [
    "resnet50_v1", "resnet101_v1", "resnet152_v1",
]

MODEL_LABELS = {
    "resnet50_v2":        "RN50-V2",
    "resnet101_v2":       "RN101-V2",
    "resnet152_v2":       "RN152-V2",
    "inception_v3":       "Inc-V3",
    "efficientnet_b1_v1": "EffNet-B1-V1",
    "efficientnet_b1_v2": "EffNet-B1-V2",
    "swin_t_v1":          "Swin-T-V1",
    "swin_t_v2":          "Swin-T-V2",
    "regnet_y_16gf_v2":   "RegNet-V2",
    "vit_b_16_v1":        "ViT-B16-V1",
    "vit_b_16_v2":        "ViT-B16-SWAG",
    "resnet50_v1":        "RN50-V1",
    "resnet101_v1":       "RN101-V1",
    "resnet152_v1":       "RN152-V1",
    "regnet_y_16gf_v1":   "RegNet-V1",
}

OOD_DATASETS = ["openimage_o", "inaturalist", "imagenet_o", "dtd", "ninco", "ssb_hard"]

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

def auroc(id_s, ood_s):
    Ni, No = len(id_s), len(ood_s)
    s = np.concatenate([id_s.astype(np.float64), ood_s.astype(np.float64)])
    l = np.concatenate([np.ones(Ni, np.int32), np.zeros(No, np.int32)])
    o = np.argsort(-s, kind="stable"); sl, ss = l[o], s[o]
    tp, fp = np.cumsum(sl), np.cumsum(1 - sl)
    c = np.where(np.concatenate([ss[:-1] != ss[1:], [True]]))[0]
    fpr = np.concatenate([[0.], fp[c] / No])
    tpr = np.concatenate([[0.], tp[c] / Ni])
    return float(_trapz(tpr, fpr))

def load_head(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["weight"].numpy().astype(np.float32), d["bias"].numpy().astype(np.float32)

def normalize_01(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-12)

def compute_auroc_from_head(W, b, z_id, z_oods, labels):
    """Compute accuracy and mean MLS AUROC over OOD datasets."""
    logits_id = z_id @ W.T + b
    pred = logits_id.argmax(1)
    acc = float((pred == labels).mean())
    id_mls = logits_id.max(axis=1)

    aurocs = []
    for z_ood in z_oods.values():
        logits_ood = z_ood @ W.T + b
        ood_mls = logits_ood.max(axis=1)
        aurocs.append(auroc(id_mls, ood_mls))

    return acc, float(np.mean(aurocs))

def compute_cos_energy_auroc(W, b, z_id, z_oods, labels):
    """CSKD-DC AUROC: z_score(true_cosine) + z_score(energy), averaged per OOD.

    Uses the canonical scoring functions from scoring/ood_scores.py so that
    fig5 and scoring/score_models.py produce identical numbers.
    """
    logits_id = z_id @ W.T + b
    acc = float((logits_id.argmax(1) == labels).mean())

    cos_id = score_cosine(z_id, W)
    ene_id = score_energy(logits_id)
    n_id = len(z_id)

    aurocs = []
    for z_ood in z_oods.values():
        logits_ood = z_ood @ W.T + b
        cos_ood = score_cosine(z_ood, W)
        ene_ood = score_energy(logits_ood)
        cos_all = np.concatenate([cos_id, cos_ood])
        ene_all = np.concatenate([ene_id, ene_ood])
        combined = score_cskd_dc(cos_all, ene_all)
        aurocs.append(auroc(combined[:n_id], combined[n_id:]))

    return acc, float(np.mean(aurocs))

def collect_data():
    data = []

    for mn in MODEL_LIST:
        label = MODEL_LABELS.get(mn, mn)

        id_path = POOLED / mn / "imagenet1k_val_emb.npy"
        lbl_path = POOLED / mn / "imagenet1k_val_lbl.npy"
        if not id_path.exists():
            print(f"  {mn}: NO EMBEDDINGS")
            continue

        z_id = np.load(id_path).astype(np.float32)
        labels = np.load(lbl_path).astype(np.int32)

        z_oods = {}
        for od in OOD_DATASETS:
            p = POOLED / "ood" / od / mn / f"{od}_emb.npy"
            if p.exists():
                z_oods[od] = np.load(p).astype(np.float32)

        if not z_oods:
            print(f"  {mn}: NO OOD DATA")
            continue

        print(f"\n  {mn} ({label}):")

        pre_path = POOLED / mn / "trained_fc.pth"
        if pre_path.exists():
            W, b = load_head(pre_path)
            acc, mean_auroc = compute_auroc_from_head(W, b, z_id, z_oods, labels)
            data.append((label, "Pretrained", acc * 100, mean_auroc, 0.0, 0.0))
            print(f"    Pretrained:  Acc={acc*100:.2f}%  AUROC={mean_auroc:.4f}")

        ce_path = TSWEEP / mn / "ce_only" / "final.pth"
        if ce_path.exists():
            W, b = load_head(ce_path)
            acc, mean_auroc = compute_auroc_from_head(W, b, z_id, z_oods, labels)
            data.append((label, "AugDelete", acc * 100, mean_auroc, 0.0, 0.0))
            print(f"    AugDelete:   Acc={acc*100:.2f}%  AUROC={mean_auroc:.4f}")

        ttest_path = TTEST_DIR / mn / f"{mn}_ttest_results.json"
        if ttest_path.exists():
            ttest = json.load(open(ttest_path))
            heads = ttest.get("heads", {})

            if "linear_ce" in heads and heads["linear_ce"]:
                entries = heads["linear_ce"]
                accs = [r["accuracy"] for r in entries if "accuracy" in r]
                aurocs_list = [r["mls_mean_auroc"] for r in entries if "mls_mean_auroc" in r]
                if accs and aurocs_list:
                    acc_m = np.mean(accs)
                    aur_m = np.mean(aurocs_list)
                    acc_s = np.std(accs) if len(accs) > 1 else 0.0
                    aur_s = np.std(aurocs_list) if len(aurocs_list) > 1 else 0.0
                    acc_pct = acc_m * 100 if acc_m <= 1.0 else acc_m
                    acc_spct = acc_s * 100 if acc_m <= 1.0 else acc_s
                    data = [(l, m, a, r, sa, sr) for l, m, a, r, sa, sr in data
                            if not (l == label and m == "AugDelete")]
                    data.append((label, "AugDelete", acc_pct, aur_m, acc_spct, aur_s))
                    print(f"    AugDelete*:  Acc={acc_pct:.2f}+/-{acc_spct:.2f}%  "
                          f"AUROC={aur_m:.4f}+/-{aur_s:.4f}  ({len(aurocs_list)} seeds)")

            override_path = (CSKD_LAM_DIR / mn /
                             f"lam{CSKD_LAM_VALUE}_T{CSKD_LAM_T}.pth"
                             if CSKD_LAM_DIR is not None else None)
            if override_path is not None and override_path.is_file():
                W, b = load_head(override_path)
                acc, mean_auroc = compute_auroc_from_head(W, b, z_id, z_oods, labels)
                data.append((label, "CSKD T=0.5", acc * 100, mean_auroc, 0.0, 0.0))
                print(f"    CSKD T=0.5(lambda={CSKD_LAM_VALUE}):  Acc={acc*100:.2f}%  AUROC={mean_auroc:.4f}  [override]")
            elif "linear_kd" in heads and heads["linear_kd"]:
                if STRICT_CSKD and CSKD_LAM_DIR is not None:
                    print(f"    [skip {mn}] strict mode: missing {override_path}")
                    continue
                entries = heads["linear_kd"]
                accs = [r["accuracy"] for r in entries if "accuracy" in r]
                aurocs_list = [r["mls_mean_auroc"] for r in entries if "mls_mean_auroc" in r]
                if accs and aurocs_list:
                    acc_m = np.mean(accs)
                    aur_m = np.mean(aurocs_list)
                    acc_s = np.std(accs) if len(accs) > 1 else 0.0
                    aur_s = np.std(aurocs_list) if len(aurocs_list) > 1 else 0.0
                    acc_pct = acc_m * 100 if acc_m <= 1.0 else acc_m
                    acc_spct = acc_s * 100 if acc_m <= 1.0 else acc_s
                    data.append((label, "CSKD T=0.5", acc_pct, aur_m, acc_spct, aur_s))
                    print(f"    CSKD T=0.5:  Acc={acc_pct:.2f}+/-{acc_spct:.2f}%  "
                          f"AUROC={aur_m:.4f}+/-{aur_s:.4f}  ({len(aurocs_list)} seeds)")
            else:
                if STRICT_CSKD and CSKD_LAM_DIR is not None:
                    print(f"    [skip {mn}] strict mode: missing {override_path}")
                    continue
                ck_path = TSWEEP / mn / "T0p50" / "final.pth"
                if ck_path.exists():
                    W, b = load_head(ck_path)
                    acc, mean_auroc = compute_auroc_from_head(W, b, z_id, z_oods, labels)
                    data.append((label, "CSKD T=0.5", acc * 100, mean_auroc, 0.0, 0.0))
                    print(f"    CSKD T=0.5:  Acc={acc*100:.2f}%  AUROC={mean_auroc:.4f}")
        else:
            override_path = (CSKD_LAM_DIR / mn /
                             f"lam{CSKD_LAM_VALUE}_T{CSKD_LAM_T}.pth"
                             if CSKD_LAM_DIR is not None else None)
            if override_path is not None and override_path.is_file():
                W, b = load_head(override_path)
                acc, mean_auroc = compute_auroc_from_head(W, b, z_id, z_oods, labels)
                data.append((label, "CSKD T=0.5", acc * 100, mean_auroc, 0.0, 0.0))
                print(f"    CSKD T=0.5(lambda={CSKD_LAM_VALUE}):  Acc={acc*100:.2f}%  AUROC={mean_auroc:.4f}  [override]")
            elif STRICT_CSKD and CSKD_LAM_DIR is not None:
                print(f"    [skip {mn}] strict mode: missing {override_path}")
                continue
            else:
                ck_path = TSWEEP / mn / "T0p50" / "final.pth"
                if ck_path.exists():
                    W, b = load_head(ck_path)
                    acc, mean_auroc = compute_auroc_from_head(W, b, z_id, z_oods, labels)
                    data.append((label, "CSKD T=0.5", acc * 100, mean_auroc, 0.0, 0.0))
                    print(f"    CSKD T=0.5:  Acc={acc*100:.2f}%  AUROC={mean_auroc:.4f}")

        ck_head_path = (CSKD_LAM_DIR / mn /
                        f"lam{CSKD_LAM_VALUE}_T{CSKD_LAM_T}.pth"
                        if CSKD_LAM_DIR is not None else Path("/dev/null"))
        if not ck_head_path.is_file():
            ck_head_path = TTEST_DIR / mn / "linear_kd_seed42.pth"
        if not ck_head_path.is_file():
            ck_head_path = TSWEEP / mn / "T0p50" / "final.pth"
        if ck_head_path.is_file():
            W, b = load_head(ck_head_path)
            acc, ce_auroc = compute_cos_energy_auroc(W, b, z_id, z_oods, labels)
            data.append((label, "CSKD Decomp", acc * 100, ce_auroc, 0.0, 0.0))
            print(f"    CSKD Decomp: Acc={acc*100:.2f}%  C+E AUROC={ce_auroc:.4f}")

    for mn in V1_MODELS:
        label = MODEL_LABELS.get(mn, mn)
        id_path = POOLED / mn / "imagenet1k_val_emb.npy"
        pre_path = POOLED / mn / "trained_fc.pth"
        if not id_path.exists() or not pre_path.exists():
            continue

        z_id = np.load(id_path).astype(np.float32)
        labels = np.load(POOLED / mn / "imagenet1k_val_lbl.npy").astype(np.int32)
        z_oods = {}
        for od in OOD_DATASETS:
            p = POOLED / "ood" / od / mn / f"{od}_emb.npy"
            if p.exists():
                z_oods[od] = np.load(p).astype(np.float32)
        if not z_oods:
            continue

        W, b = load_head(pre_path)
        acc, mean_auroc = compute_auroc_from_head(W, b, z_id, z_oods, labels)
        data.append((label, "V1 Pretrained", acc * 100, mean_auroc, 0.0, 0.0))
        print(f"\n  {mn} ({label}): V1 Pretrained Acc={acc*100:.2f}%  AUROC={mean_auroc:.4f}")

    return data

METHOD_STYLE = {
    "Pretrained":    {"marker": "o", "color": "#d62728", "size": 90,
                      "zorder": 3, "edgecolor": "white", "linewidth": 0.8},
    "AugDelete":     {"marker": "s", "color": "#ff7f0e", "size": 90,
                      "zorder": 4, "edgecolor": "white", "linewidth": 0.8},
    "CSKD T=0.5":    {"marker": "*", "color": "#2ca02c", "size": 180,
                      "zorder": 5, "edgecolor": "white", "linewidth": 0.6},
    "CSKD Decomp":   {"marker": "D", "color": "#1f77b4", "size": 90,
                      "zorder": 6, "edgecolor": "white", "linewidth": 0.8},
    "V1 Pretrained": {"marker": "^", "color": "#7f7f7f", "size": 70,
                      "zorder": 2, "edgecolor": "white", "linewidth": 0.8},
}
METHOD_ORDER = ["V1 Pretrained", "Pretrained", "AugDelete", "CSKD T=0.5", "CSKD Decomp"]

def make_plot(data):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    model_points = defaultdict(list)
    for model, method, acc, aur, acc_s, aur_s in data:
        model_points[model].append((method, acc, aur, acc_s, aur_s))

    for model, points in model_points.items():
        pts = {m: (a, r) for m, a, r, _, _ in points}
        ordered = [m for m in METHOD_ORDER if m in pts]
        for i in range(len(ordered) - 1):
            m1, m2 = ordered[i], ordered[i + 1]
            x1, y1 = pts[m1]
            x2, y2 = pts[m2]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color="#BBBBBB",
                                        lw=1.2, shrinkA=4, shrinkB=4),
                        zorder=1)

    for model, method, acc, aur, acc_s, aur_s in data:
        s = METHOD_STYLE[method]
        if (acc_s > 0) or (aur_s > 0):
            ax.errorbar(acc, aur,
                        xerr=acc_s if acc_s > 0 else None,
                        yerr=aur_s if aur_s > 0 else None,
                        fmt="none", ecolor=s["color"], elinewidth=1.0,
                        capsize=3, capthick=0.8, alpha=0.6,
                        zorder=s["zorder"] - 1)
        ax.scatter(acc, aur, marker=s["marker"], s=s["size"],
                   c=s["color"], edgecolors=s["edgecolor"],
                   linewidths=s["linewidth"], zorder=s["zorder"])

    LABEL_CONFIG = {
        "RN50-V2":       {"offset": (8, -12),  "ha": "left"},
        "RN101-V2":      {"offset": (8, -20),    "ha": "left"},
        "RN152-V2":      {"offset": (0, -16),    "ha": "left"},
        "Inc-V3":        {"offset": (-8, -12), "ha": "right"},
        "EffNet-B1-V1":  {"offset": (-8, -12), "ha": "right"},
        "EffNet-B1-V2":  {"offset": (0, -12),  "ha": "left"},
        "Swin-T-V1":     {"offset": (-20, -12),    "ha": "left"},
        "Swin-T-V2":     {"offset": (8, -12),  "ha": "left"},
        "RegNet-V2":     {"offset": (8, -12),  "ha": "left"},
        "ViT-B16-V1":    {"offset": (-8, -16),  "ha": "left"},
        "ViT-B16-SWAG":  {"offset": (-30, -20),    "ha": "left"},
        "RN50-V1":       {"offset": (0, -12),  "ha": "center"},
        "RN101-V1":      {"offset": (8, -12),  "ha": "center"},
        "RN152-V1":      {"offset": (-12, -12),  "ha": "center"},
    }

    for model, points in model_points.items():
        for pref in ["Pretrained", "V1 Pretrained"]:
            match = [(a, r) for m, a, r, _, _ in points if m == pref]
            if match:
                px, py = match[0]
                break
        else:
            px, py = points[0][1], points[0][2]

        cfg = LABEL_CONFIG.get(model, {"offset": (8, -2), "ha": "left"})
        is_v1 = any(m == "V1 Pretrained" for m, _, _, _, _ in points)
        ax.annotate(
            model, xy=(px, py), xytext=cfg["offset"],
            textcoords="offset points",
            fontsize=6 if is_v1 else 6.5,
            color="#7f7f7f" if is_v1 else "#333333",
            fontweight="normal" if is_v1 else "bold",
            fontstyle="italic" if is_v1 else "normal",
            ha=cfg["ha"], va="center",
            arrowprops=dict(arrowstyle="-",
                            color="#bbbbbb" if is_v1 else "#999999",
                            lw=0.4, shrinkA=0, shrinkB=2),
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_xlabel("Top-1 Accuracy (%)")
    ax.set_ylabel("MLS AUROC (mean over 6 OOD datasets)")
    path_methods = [m for m in METHOD_ORDER if m != "V1 Pretrained"]
    subtitle = " -> ".join(path_methods) if path_methods else ""
    ax.set_title("Accuracy vs OOD Detection" +
                 (f"\n{subtitle}" if subtitle else ""),
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.2, linestyle="--")

    legend_elements = []
    for method in METHOD_ORDER:
        s = METHOD_STYLE[method]
        ms = 10 if s["marker"] != "*" else 14
        legend_elements.append(
            Line2D([0], [0], marker=s["marker"], color="w",
                   markerfacecolor=s["color"], markersize=ms,
                   markeredgecolor=s["edgecolor"],
                   markeredgewidth=s["linewidth"],
                   label=method, linestyle="None"))
    legend_elements.append(
        Line2D([0], [0], color="#BBBBBB", lw=1.5, linestyle="-",
               label="Improvement path", marker=">", markersize=4,
               markerfacecolor="#BBBBBB"))

    ax.legend(handles=legend_elements, loc="center right",
              bbox_to_anchor=(1.0, 0.35),
              frameon=True, facecolor="white", edgecolor="#CCCCCC",
              framealpha=0.95, borderpad=0.6, handletextpad=0.4,
              fontsize=8)

    fig.tight_layout()

    out_pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    out_png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(out_pdf, facecolor="white")
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of methods to plot. "
                        "Choices: 'V1 Pretrained' Pretrained AugDelete "
                        "'CSKD T=0.5' 'CSKD Decomp'. "
                        "Default = all.")
    p.add_argument("--out-name", type=str, default=None,
                   help="Output filename stem (default: auto from --methods)")
    p.add_argument("--cskd-lambda-dir", type=str, default=None,
                   help="Override CSKD T=0.5 head source. Expects layout "
                        "{dir}/{model}/lam{X}_T{Y}.pth.")
    p.add_argument("--cskd-lambda", type=str, default=None,
                   help="lambda value to read inside --cskd-lambda-dir, e.g. 0.9")
    p.add_argument("--cskd-T", type=str, default="0.5",
                   help="T value to read inside --cskd-lambda-dir (default 0.5)")
    p.add_argument("--strict-cskd", action="store_true",
                   help="Skip models whose override head is missing "
                        "(otherwise fall back to ttest/tsweep).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Limit to these V2 model names.")
    args = p.parse_args()

    if args.cskd_lambda_dir:
        CSKD_LAM_DIR = Path(args.cskd_lambda_dir)
        CSKD_LAM_VALUE = args.cskd_lambda
        CSKD_LAM_T = args.cskd_T
        STRICT_CSKD = args.strict_cskd
        if not CSKD_LAM_VALUE:
            print("ERROR: --cskd-lambda-dir given without --cskd-lambda value")
            sys.exit(2)
    if args.models:
        MODEL_LIST = [m for m in MODEL_LIST if m in args.models]
        V1_MODELS = [m for m in V1_MODELS if m in args.models]

    data = collect_data()
    if not data:
        print("\nERROR: No data found.")
        sys.exit(1)

    if args.methods:
        keep = set(args.methods)
        unknown = keep - set(METHOD_ORDER)
        if unknown:
            print(f"ERROR: unknown methods {unknown}. "
                  f"Choices: {METHOD_ORDER}")
            sys.exit(2)
        before = len(data)
        data = [r for r in data if r[1] in keep]
        METHOD_ORDER = [m for m in METHOD_ORDER if m in keep]
        print(f"\n  Filtered: {before} -> {len(data)} data points "
              f"(methods kept: {METHOD_ORDER})")

    if args.out_name:
        OUT_NAME = args.out_name
    elif args.methods:
        slug = "_".join(m.replace(" ", "").replace("=", "").replace(".", "p")
                        for m in METHOD_ORDER).lower()
        OUT_NAME = f"fig1_acc_vs_auroc_{slug}"
    else:
        OUT_NAME = "fig1_auroc_vs_accuracy"

    print(f"\n  Collected {len(data)} data points")
    make_plot(data)
    print("\nDone.")
