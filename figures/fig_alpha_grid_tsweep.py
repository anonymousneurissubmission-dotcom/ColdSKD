"""
2x3 grid: temperature sweep of MLS AUROC for CSKD heads.

Rows    = dataset    (CIFAR-100, ImageNet-200)
Columns = alpha (LS) (0.1, 0.2, 0.3)
Curves  = lambda values strictly inside (0, 1):  {0.3, 0.5, 0.7, 0.9}
          (AugDelete / lambda=0 and pure-KD / lambda=1 omitted)

Each curve = mean over 3 seeds at every shared temperature, with shaded
+/-1 std band. Reads pre-extracted pooled features and the bundled grid
heads at trained_heads/{cifar100,imagenet200}/{backbone}/lam{L}_T{T}.pth.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import POOLED_FEATURES, FIGURES_DIR, ensure_dirs
ensure_dirs()
sys.path.insert(0, str(ROOT / "scoring"))
from ood_scores import score_mls, auroc

ALPHAS = ["0.1", "0.2", "0.3"]
SEEDS = [0, 1, 2]
LAMBDAS = [0.3, 0.5, 0.7, 0.9]
TEMPERATURES = [0.01, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0,
                 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

DATASETS = [
    {
        "name": "CIFAR-100",
        "key": "cifar100",
        "id_test_tag": "cifar100_test",
        "ood_sets": ["cifar10", "svhn", "textures",
                      "places", "lsun_c", "lsun_r", "isun"],
        "backbone_fmt": "resnet18_s{seed}_ls_{alpha}",
        "head_fmt": "resnet18_s{seed}_ls_{alpha}_grid",
    },
    {
        "name": "ImageNet-200",
        "key": "imagenet200",
        "id_test_tag": "imagenet200_test",
        "ood_sets": ["inaturalist", "openimage_o", "textures",
                      "ssb_hard", "ninco"],
        "backbone_fmt": "ls_{alpha}_s{seed}",
        "head_fmt": "ls_{alpha}_s{seed}",
    },
]

HEADS_ROOT = ROOT / "trained_heads"


def fmt_t(t):
    return str(float(t))


def fmt_lam(lam):
    return str(float(lam))


def load_features(dataset, backbone):
    base = POOLED_FEATURES / dataset["key"] / backbone
    z_id = np.load(base / f"{dataset['id_test_tag']}_emb.npy").astype(np.float32)
    z_oods = []
    for od in dataset["ood_sets"]:
        p = (POOLED_FEATURES / dataset["key"] / "ood" / od / backbone /
             f"{od}_emb.npy")
        if p.is_file():
            z_oods.append(np.load(p).astype(np.float32))
    return z_id, z_oods


def load_head(path):
    h = torch.load(path, map_location="cpu", weights_only=True)
    return h["weight"].numpy().astype(np.float32), h["bias"].numpy().astype(np.float32)


def mean_auroc(W, b, z_id, z_oods):
    mls_id = score_mls(z_id @ W.T + b)
    aurocs = []
    for z_ood in z_oods:
        mls_ood = score_mls(z_ood @ W.T + b)
        aurocs.append(auroc(mls_id, mls_ood))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def collect_curve(dataset, alpha, lam):
    """Return dict T -> list of per-seed AUROCs."""
    by_T = {T: [] for T in TEMPERATURES}
    for seed in SEEDS:
        backbone = dataset["backbone_fmt"].format(seed=seed, alpha=alpha)
        head_dir = dataset["head_fmt"].format(seed=seed, alpha=alpha)
        z_id, z_oods = load_features(dataset, backbone)
        for T in TEMPERATURES:
            head_path = HEADS_ROOT / dataset["key"] / head_dir / \
                        f"lam{fmt_lam(lam)}_T{fmt_t(T)}.pth"
            if not head_path.is_file():
                continue
            W, b = load_head(head_path)
            by_T[T].append(mean_auroc(W, b, z_id, z_oods))
    return by_T


def render():
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(len(DATASETS), len(ALPHAS),
                              figsize=(15.0, 7.5),
                              squeeze=False)

    cmap = plt.get_cmap("viridis")
    lam_color = {lam: cmap(i / max(1, len(LAMBDAS) - 1))
                 for i, lam in enumerate(LAMBDAS)}

    for r, dataset in enumerate(DATASETS):
        for c, alpha in enumerate(ALPHAS):
            ax = axes[r][c]
            print(f"\n[{dataset['name']}  alpha={alpha}]")
            for lam in LAMBDAS:
                by_T = collect_curve(dataset, alpha, lam)
                Ts = [T for T in TEMPERATURES if len(by_T[T]) > 0]
                means = [float(np.mean(by_T[T])) for T in Ts]
                stds = [float(np.std(by_T[T])) for T in Ts]
                if not Ts:
                    print(f"  lambda={lam}: no heads found")
                    continue
                col = lam_color[lam]
                ax.plot(Ts, means, "-o", color=col, lw=1.8, ms=4,
                         label=fr"$\lambda$={lam}")
                idx_best = int(np.argmax(means))
                print(f"  lambda={lam}: peak AUROC={means[idx_best]:.4f}  "
                      f"at T={Ts[idx_best]}  ({len(Ts)} T points)")

            ax.set_xscale("log")
            ax.axvline(1.0, color="grey", lw=0.7, ls=":", alpha=0.5)
            ax.set_title(fr"{dataset['name']}  $\alpha$={alpha}",
                          fontweight="bold")
            ax.grid(True, alpha=0.25, linestyle="--")
            if c == 0:
                ax.set_ylabel("MLS AUROC (mean over OOD)")
            if r == len(DATASETS) - 1:
                ax.set_xlabel("Distillation Temperature $T$ (log scale)")
            if (r, c) == (0, 0):
                ax.legend(loc="lower center", frameon=True,
                           facecolor="white", edgecolor="#cccccc",
                           framealpha=0.95)

    fig.suptitle("CSKD Temperature Sweep: MLS AUROC vs $T$  "
                 r"(0 < $\lambda$ < 1, mean over 3 seeds)",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_pdf = FIGURES_DIR / "fig_alpha_grid_tsweep.pdf"
    out_png = FIGURES_DIR / "fig_alpha_grid_tsweep.png"
    fig.savefig(out_pdf, facecolor="white")
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    render()
