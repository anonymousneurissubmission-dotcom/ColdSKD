"""1x3 alpha sweep figure for CIFAR-100 only.

Layout: 1 row x 3 columns, alpha in {0.1, 0.2, 0.3}.

Each panel plots the signed gradient-norm balance:
  -(1-lam) * ||dL_CE/dW||_F   (CE pull, dashed, negative)
  +lam     * ||dL_KL/dW||_F   (KD pull, solid+marker, positive)
KD loss = T^2 * KL(p_s_T || p_tau_T) (matching the training objective).
Multi-seed averaged across {0,1,2}. y-axis shared across the 3 alpha cols.

Output: results/plot_1x3_cifar100_alpha_sweep.pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import POOLED_FEATURES, HEADS_DIR, FIGURES_DIR, ensure_dirs
ensure_dirs()
RES = FIGURES_DIR
import os as _os
_BUNDLED = ROOT / "trained_heads" / "cifar100"
HEADS_CIFAR = Path(_os.environ.get("COLDSKD_HEADS_CIFAR",
                                    str(_BUNDLED if _BUNDLED.is_dir()
                                        else HEADS_DIR / "cifar100")))

T_MIN, T_MAX = 0.1, 50.0
ALPHAS = [0.1, 0.2, 0.3]
LAMS = [0.3, 0.5, 0.7, 0.9]
TS = [0.001, 0.01, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0,
      1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]
SEEDS = [0, 1, 2]
CMAP = plt.get_cmap("viridis")
LAM_COLOR = {l: CMAP(i / (len(LAMS) - 1)) for i, l in enumerate(LAMS)}

CFG = dict(
    title="CIFAR-100  RN18  (K=100)",
    K=100,
    pool_root=str(POOLED_FEATURES / "cifar100"),
    pool_fmt="resnet18_s{seed}_ls_{alpha}",
    emb="cifar100_train_emb.npy",
    lbl="cifar100_train_lbl.npy",
    heads_root=str(HEADS_CIFAR),
    heads_fmt="resnet18_s{seed}_ls_{alpha}_grid",
)

def compute_one_backbone(cfg, seed, alpha, device, batch_size=2048):
    pool = Path(cfg["pool_root"]) / cfg["pool_fmt"].format(
        seed=seed, alpha=alpha)
    heads = Path(cfg["heads_root"]) / cfg["heads_fmt"].format(
        seed=seed, alpha=alpha)
    z = np.load(pool / cfg["emb"]).astype(np.float32)
    y = np.load(pool / cfg["lbl"]).astype(np.int64)
    fc = torch.load(pool / "trained_fc.pth", map_location="cpu",
                    weights_only=False)
    z = torch.from_numpy(z).to(device)
    y = torch.from_numpy(y).to(device)
    W_t = fc["weight"].float().to(device)
    b_t = fc.get("bias")
    if b_t is not None:
        b_t = b_t.float().to(device)
    n = z.shape[0]
    with torch.no_grad():
        v_tau_full = z @ W_t.T
        if b_t is not None:
            v_tau_full = v_tau_full + b_t

    rows = []
    for lam in LAMS:
        for T in TS:
            ck = torch.load(heads / f"lam{lam}_T{T}.pth",
                            map_location=device, weights_only=False)
            W = ck["weight"].to(device).requires_grad_(True)
            b = ck["bias"].to(device).requires_grad_(True)

            grad_W_CE = torch.zeros_like(W)
            grad_W_KL = torch.zeros_like(W)
            for i in range(0, n, batch_size):
                j = min(i + batch_size, n)
                z_b = z[i:j]; y_b = y[i:j]; v_tau = v_tau_full[i:j]
                v_s = z_b @ W.T + b
                logp_s_T1 = F.log_softmax(v_s, dim=-1)
                ce = -logp_s_T1.gather(1, y_b.unsqueeze(1)).squeeze(1).mean()
                logp_s_T = F.log_softmax(v_s / T, dim=-1)
                logp_t_T = F.log_softmax(v_tau / T, dim=-1)
                kl_loss = (T * T) * (
                    logp_s_T.exp() * (logp_s_T - logp_t_T)
                ).sum(-1).mean()
                w_chunk = (j - i) / n
                grad_W_CE += w_chunk * torch.autograd.grad(
                    ce, W, retain_graph=True)[0]
                grad_W_KL += w_chunk * torch.autograd.grad(
                    kl_loss, W, retain_graph=False)[0]
                W.grad = None; b.grad = None

            rows.append(dict(
                seed=seed, alpha=alpha, lam=lam, T=T,
                grad_CE_fro=float(grad_W_CE.norm()),
                grad_KL_fro=float(grad_W_KL.norm()),
            ))
    return pd.DataFrame(rows)

def compute_avg(cfg, alpha, device):
    parts = [compute_one_backbone(cfg, s, alpha, device) for s in SEEDS]
    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["lam", "T"], as_index=False).agg(
        grad_CE_fro=("grad_CE_fro", "mean"),
        grad_KL_fro=("grad_KL_fro", "mean"),
    )

def plot_panel(ax, df):
    df = df[(df["T"] >= T_MIN) & (df["T"] <= T_MAX)]
    for lam in LAMS:
        sub = df[df.lam == lam].sort_values("T")
        color = LAM_COLOR[lam]
        ax.plot(sub["T"], -(1.0 - lam) * sub["grad_CE_fro"], "--",
                color=color, lw=1.4, alpha=0.85,
                label=fr"CE ($\lambda = {lam}$)")
        ax.plot(sub["T"], lam * sub["grad_KL_fro"], "-", marker="o", ms=3,
                color=color, lw=1.4, alpha=0.95,
                label=fr"KD ($\lambda = {lam}$)")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.axvline(1.0, color="grey", lw=0.5, ls=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlim(T_MIN, T_MAX)
    ax.grid(alpha=0.25)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    data = {}
    for a in ALPHAS:
        print(f"=== cifar100 alpha={a} ===")
        data[a] = compute_avg(CFG, a, device)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6),
                              sharey=True, sharex=True)

    for c, a in enumerate(ALPHAS):
        ax = axes[c]
        plot_panel(ax, data[a])
        ax.set_title(fr"$\alpha={a}$", fontsize=12)
        ax.set_xlabel("T")
        if c == 0:
            ax.set_ylabel(
                f"{CFG['title']}\nsigned grad norm",
                fontsize=9,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.1),
               ncol=len(handles), frameon=False, fontsize=12)
    out_pdf = RES / "plot_1x3_cifar100_alpha_sweep_v2.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_pdf}")

    txt = RES / "plot_1x3_cifar100_alpha_sweep.txt"
    txt.write_text(
        "Script: plot_1x3_cifar100_alpha_sweep.py\n"
        "Layout: 1 row x 3 cols (alpha=0.1, 0.2, 0.3) for CIFAR-100 RN18.\n"
        "Multi-seed averaged across {0,1,2}. y shared across cols.\n"
        "Per panel: signed gradient norms\n"
        "  -(1-lam)*||G_CE||_F (dashed) and +lam*||G_KL||_F (solid+marker).\n"
        "KD loss = T^2 * KL(p_s_T||p_tau_T). T axis [0.1, 50] log.\n"
        f"Lambdas: {LAMS}\n"
    )
    print(f"wrote {txt}")

if __name__ == "__main__":
    main()
