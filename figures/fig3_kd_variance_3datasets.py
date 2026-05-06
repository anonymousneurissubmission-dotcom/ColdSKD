"""3-panel plot of teacher tempered Bernoulli variance vs T for
CIFAR-100, ImageNet-200, and ImageNet-1k.

Only the Bernoulli variance (green) is plotted -- empirical markers from
the actual teacher head softmax outputs, plus the symmetric-bg theory
(dashed line). Subplot title = dataset/model name only.

Output: results/plot_kd_variance_3datasets.pdf
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import POOLED_FEATURES, FIGURES_DIR, ensure_dirs
ensure_dirs()
RES = FIGURES_DIR

def gstar(alpha, K):
    return float(np.log(K * (1 - alpha) / alpha + 1))

def measure_teacher_at_T(z, y, W, b, T, K, sample_cap=None):
    """Return mean p_y, mean H, sigma^2 = p_y*(1-p_y) at temperature T."""
    if sample_cap is not None and len(z) > sample_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(z), size=sample_cap, replace=False)
        z_use = z[idx]; y_use = y[idx]
    else:
        z_use = z; y_use = y
    Wt = torch.from_numpy(W); bt = torch.from_numpy(b) if b is not None else None
    with torch.no_grad():
        v = torch.from_numpy(z_use) @ Wt.T + (bt if bt is not None else 0)
        v = v.float()
        pred = v.argmax(dim=1).numpy()
        correct = pred == y_use
        if correct.sum() == 0:
            return float("nan")
        v_c = v[correct].numpy()
        v_T = v_c / T
        v_T_shift = v_T - v_T.max(axis=1, keepdims=True)
        e = np.exp(v_T_shift)
        p = e / e.sum(axis=1, keepdims=True)
        y_c = y_use[correct]
        p_y = p[np.arange(p.shape[0]), y_c]
    return float((p_y * (1 - p_y)).mean()), float(p_y.mean())

def per_sample_gaps(z, y, W, b, K, sample_cap=None):
    """Per-sample effective gaps g_i, fit from p^{tau,1}_{y,i}.
    g_i = log((K-1) * p_y / (1 - p_y)) at T=1 per correct sample.
    Returns: g_array (N_correct,), p_y_T1_mean (for sym-bg theory).
    """
    if sample_cap is not None and len(z) > sample_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(z), size=sample_cap, replace=False)
        z_use = z[idx]; y_use = y[idx]
    else:
        z_use = z; y_use = y
    Wt = torch.from_numpy(W); bt = torch.from_numpy(b) if b is not None else None
    with torch.no_grad():
        v = torch.from_numpy(z_use) @ Wt.T + (bt if bt is not None else 0)
        v = v.float()
        pred = v.argmax(dim=1).numpy()
        correct = pred == y_use
        v_c = v[correct].numpy()
        y_c = y_use[correct]
        v1 = v_c - v_c.max(axis=1, keepdims=True)
        e1 = np.exp(v1)
        p1 = e1 / e1.sum(axis=1, keepdims=True)
        p_y = p1[np.arange(p1.shape[0]), y_c]
    p_y = np.clip(p_y, 1e-8, 1 - 1e-8)
    V_i = (1.0 - p_y) / p_y
    V_i = np.maximum(V_i, 1e-12)
    g_i = np.log((K - 1) / V_i)
    g_i = np.maximum(g_i, 1e-6)
    return g_i, float(p_y.mean())

def sigma2_optionC(g_array, T_grid, K):
    """Population-averaged sigma^2 over per-sample gaps."""
    out = np.empty(len(T_grid))
    for j, T in enumerate(T_grid):
        V = (K - 1) * np.exp(-g_array / T)
        out[j] = float((V / (1.0 + V) ** 2).mean())
    return out

def per_sample_leader_bulk(z, y, W, b, K, sample_cap=None):
    """Option D: per-sample leader + bulk decomposition.
    Fit at T=1:
      g1_i  = log(p_y_i / p_(2)_i)             [near-miss gap]
      gbg_i = log((K-2) * p_y_i / residual_i)  [background gap]
    where p_(2)_i is the largest non-target probability and residual_i
    is the remaining non-target mass (1 - p_y_i - p_(2)_i).
    Returns: (g1, gbg) arrays of length N_correct.
    """
    if sample_cap is not None and len(z) > sample_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(z), size=sample_cap, replace=False)
        z_use = z[idx]; y_use = y[idx]
    else:
        z_use = z; y_use = y
    Wt = torch.from_numpy(W); bt = torch.from_numpy(b) if b is not None else None
    with torch.no_grad():
        v = torch.from_numpy(z_use) @ Wt.T + (bt if bt is not None else 0)
        v = v.float()
        pred = v.argmax(dim=1).numpy()
        correct = pred == y_use
        v_c = v[correct].numpy()
        y_c = y_use[correct]
        v1 = v_c - v_c.max(axis=1, keepdims=True)
        e1 = np.exp(v1)
        p1 = e1 / e1.sum(axis=1, keepdims=True)
        N = p1.shape[0]
        rows = np.arange(N)
        p_y = p1[rows, y_c]
        p_masked = p1.copy()
        p_masked[rows, y_c] = -1.0
        p2 = p_masked.max(axis=1)
    p_y = np.clip(p_y, 1e-8, 1 - 1e-8)
    p2 = np.clip(p2, 1e-12, 1 - 1e-8)
    residual = np.clip(1.0 - p_y - p2, 1e-12, 1.0)
    g1 = np.log(p_y / p2)
    g1 = np.maximum(g1, 1e-6)
    if K > 2:
        gbg = np.log((K - 2) * p_y / residual)
        gbg = np.maximum(gbg, 1e-6)
    else:
        gbg = np.full_like(g1, 1e6)
    return g1, gbg

def sigma2_optionD(g1, gbg, T_grid, K):
    """Population-averaged sigma^2 with leader + bulk per-sample model."""
    out = np.empty(len(T_grid))
    for j, T in enumerate(T_grid):
        V = np.exp(-g1 / T) + (K - 2) * np.exp(-gbg / T)
        out[j] = float((V / (1.0 + V) ** 2).mean())
    return out

DATASETS = [
    ("CIFAR-100  RN18 ls0.1", "cifar100", 100,
     str(POOLED_FEATURES / "cifar100" / "resnet18_s0_ls_0.1"),
     "cifar100_train_emb.npy", "cifar100_train_lbl.npy", None),
    ("ImageNet-200  RN18 ls0.1", "imagenet200", 200,
     str(POOLED_FEATURES / "imagenet200" / "ls_0.1_s0"),
     "imagenet200_train_emb.npy", "imagenet200_train_lbl.npy", None),
    ("ImageNet-1k  RN50 V2", "imagenet1k", 1000,
     str(POOLED_FEATURES / "imagenet1k" / "resnet50_v2"),
     "imagenet1k_val_emb.npy", "imagenet1k_val_lbl.npy", 50000),
]

def main():
    T_grid = np.geomspace(0.1, 30.0, 60)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), sharey=False)

    for ax, (title, ds, K, root, emb_name, lbl_name, cap) in zip(axes, DATASETS):
        if not os.path.isdir(root):
            ax.text(0.5, 0.5, f"missing: {root}",
                    transform=ax.transAxes, ha="center")
            ax.set_title(title); continue
        try:
            z = np.load(os.path.join(root, emb_name)).astype(np.float32)
            y = np.load(os.path.join(root, lbl_name))
            fc = torch.load(os.path.join(root, "trained_fc.pth"),
                            map_location="cpu", weights_only=False)
            W = fc["weight"].float().numpy()
            b = (fc["bias"].float().numpy()
                 if fc.get("bias") is not None else None)
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha="center")
            ax.set_title(title); continue

        sigma2_emp = []
        py_emp = []
        for T in T_grid:
            s2, py = measure_teacher_at_T(z, y, W, b, T, K, sample_cap=cap)
            sigma2_emp.append(s2); py_emp.append(py)
        sigma2_emp = np.array(sigma2_emp)
        py_emp = np.array(py_emp)

        idx_T1 = int(np.argmin(np.abs(T_grid - 1.0)))
        py_T1 = py_emp[idx_T1]
        alpha_emp = K * (1.0 - py_T1) / (K - 1)
        g = gstar(alpha_emp, K)
        T_peak = g / np.log(K - 1)

        V_grid = (K - 1) * np.exp(-g / T_grid)
        sigma2_theo = V_grid / (1.0 + V_grid) ** 2

        g_i, _ = per_sample_gaps(z, y, W, b, K, sample_cap=cap)
        sigma2_C = sigma2_optionC(g_i, T_grid, K)
        g_mean = float(g_i.mean())
        g_std = float(g_i.std())

        g1_i, gbg_i = per_sample_leader_bulk(z, y, W, b, K, sample_cap=cap)
        sigma2_D = sigma2_optionD(g1_i, gbg_i, T_grid, K)
        g1_mean = float(g1_i.mean())
        gbg_mean = float(gbg_i.mean())

        idx_peak = int(np.argmax(sigma2_emp))
        T_peak_emp = T_grid[idx_peak]
        mask_cold = T_grid < 1.0
        ax.fill_between(T_grid, 0.0, sigma2_emp,
                        where=mask_cold, interpolate=True,
                        color="#9bb7e8", alpha=0.55, zorder=1)

        ax.plot(T_grid, sigma2_emp, "o-", color="C2", ms=5, lw=2.0,
                label=r"empirical $\sigma^2(T)=p_y^{\tau,T}\,(1-p_y^{\tau,T})$")
        ax.axhline(0.25, color="grey", lw=0.7, ls=":", alpha=0.7,
                   label="0.25 (Bernoulli max)")
        ax.axvline(T_peak_emp, color="C2", lw=1.0, ls="--", alpha=0.7,
                   label=fr"empirical peak $T={T_peak_emp:.2f}$")

        ax.text(T_peak_emp * 0.5, sigma2_emp[idx_peak] * 0.25, "coldKD",
                ha="center", va="center", fontsize=14,
                fontweight="bold", color="#1f3a93",
                bbox=dict(boxstyle="round,pad=0.45", fc="white",
                          ec="#1f3a93", lw=1.4, alpha=0.95),
                zorder=5)

        ax.set_xscale("log")
        ax.set_xlabel("T", fontsize=14)
        ax.set_ylabel(r"$p^{\tau,T}_y\,(1{-}p^{\tau,T}_y)$", fontsize=14)
        ax.set_title(title, fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=11, frameon=False, loc="upper left")
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)

        print(f"{title}: alpha_emp={alpha_emp:.4f}  g*={g:.3f}  "
              f"T_peak={T_peak:.3f}")

    fig.suptitle("Teacher tempered Bernoulli variance vs T (empirical)",
                  y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = RES / "plot_kd_variance_3datasets.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
