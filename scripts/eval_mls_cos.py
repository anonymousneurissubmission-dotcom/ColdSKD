"""
Compare MLS, CSKD-DC (= z(cos)+z(ene)) and MLS+cosine (= z(mls)+z(cos))
across IM1K-V2/SWAG models using their CSKD T=0.5 heads.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import POOLED_FEATURES
sys.path.insert(0, str(ROOT / "scoring"))
from ood_scores import (score_mls, score_cosine, score_energy,
                         score_cskd_dc, score_mls_cos, auroc)

POOLED = POOLED_FEATURES / "imagenet1k"
HEADS = ROOT / "trained_heads" / "imagenet1k_tsweep"

MODELS = ["resnet50_v2", "resnet101_v2", "resnet152_v2",
          "inception_v3", "efficientnet_b1_v1",
          "swin_t_v1", "swin_t_v2",
          "regnet_y_16gf_v2",
          "vit_b_16_v1", "vit_b_16_v2"]
OOD_SETS = ["openimage_o", "inaturalist", "imagenet_o", "dtd", "ninco", "ssb_hard"]

def evaluate(mn):
    z_id = np.load(POOLED / mn / "imagenet1k_val_emb.npy").astype(np.float32)
    fc = torch.load(HEADS / mn / "T0p50" / "final.pth", weights_only=True)
    W = fc["weight"].numpy().astype(np.float32)
    b = fc["bias"].numpy().astype(np.float32)

    v_id = z_id @ W.T + b
    mls_id = score_mls(v_id)
    cos_id = score_cosine(z_id, W)
    ene_id = score_energy(v_id)
    n_id = len(z_id)

    a_mls, a_cskd, a_mlscos = [], [], []
    for od in OOD_SETS:
        p = POOLED / "ood" / od / mn / f"{od}_emb.npy"
        if not p.is_file():
            continue
        z_ood = np.load(p).astype(np.float32)
        v_ood = z_ood @ W.T + b
        mls_ood = score_mls(v_ood)
        cos_ood = score_cosine(z_ood, W)
        ene_ood = score_energy(v_ood)

        a_mls.append(auroc(mls_id, mls_ood))

        cos_all = np.concatenate([cos_id, cos_ood])
        ene_all = np.concatenate([ene_id, ene_ood])
        s = score_cskd_dc(cos_all, ene_all)
        a_cskd.append(auroc(s[:n_id], s[n_id:]))

        mls_all = np.concatenate([mls_id, mls_ood])
        s = score_mls_cos(mls_all, cos_all)
        a_mlscos.append(auroc(s[:n_id], s[n_id:]))

    return (float(np.mean(a_mls)), float(np.mean(a_cskd)),
            float(np.mean(a_mlscos)))

def main():
    print(f"{'model':<22} {'MLS':>8} {'CSKD-DC':>10} {'MLS+cos':>10} {'Delta(MLS+cos - MLS)':>18}")
    print("-" * 70)
    rows = []
    for mn in MODELS:
        if not (HEADS / mn / "T0p50" / "final.pth").is_file():
            print(f"{mn:<22} (no head)")
            continue
        mls, cskd, mlscos = evaluate(mn)
        delta = mlscos - mls
        rows.append((mn, mls, cskd, mlscos, delta))
        print(f"{mn:<22} {mls:>8.4f} {cskd:>10.4f} {mlscos:>10.4f} {delta:>+18.4f}")
    print("-" * 70)
    if rows:
        m_mls = np.mean([r[1] for r in rows])
        m_cskd = np.mean([r[2] for r in rows])
        m_mlscos = np.mean([r[3] for r in rows])
        print(f"{'mean':<22} {m_mls:>8.4f} {m_cskd:>10.4f} {m_mlscos:>10.4f} "
              f"{m_mlscos - m_mls:>+18.4f}")

if __name__ == "__main__":
    main()
