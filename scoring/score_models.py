"""
Score every (backbone x head x OOD) combination with MLS, ASH, CSKD-DC.

Walks POOLED_FEATURES/{dataset}/, loads each backbone's pooled features and
classification head (or any (lambda,T) head under HEADS_DIR/{dataset}/{backbone}/),
and writes a long-format CSV: one row per (backbone, head, ood_set, detector).

Detectors:
    mls           -- Maximum Logit Score
    ash_s_mls     -- MLS on ASH-S(z) embeddings
    ash_b_mls     -- MLS on ASH-B(z) embeddings
    ash_p_mls     -- MLS on ASH-P(z) embeddings
    cskd_dc       -- z_score(cosine) + z_score(energy)

Filter backbones / heads via --include or --exclude (substring match).

Run:
    python scoring/score_models.py --dataset imagenet200 \
        --include resnet50 --ash-percentile 90
"""
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import POOLED_FEATURES, HEADS_DIR, RESULTS_DIR, ensure_dirs
sys.path.insert(0, str(ROOT / "scoring"))
from ood_scores import (score_mls, score_energy, score_cosine,
                         score_cskd_dc, auroc, fpr95)
import ash as ash_mod

ID_TAGS = {
    "cifar100":    "cifar100_test",
    "imagenet200": "imagenet200_test",
    "imagenet1k":  "imagenet1k_val",
}

def discover_backbones(dataset, include, exclude):
    base = POOLED_FEATURES / dataset
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.iterdir()):
        if not p.is_dir() or p.name == "ood":
            continue
        name = p.name
        if include and not any(s in name for s in include):
            continue
        if exclude and any(s in name for s in exclude):
            continue
        out.append(name)
    return out

def discover_ood_sets(dataset):
    base = POOLED_FEATURES / dataset / "ood"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())

def load_backbone(dataset, backbone, ood_sets):
    base = POOLED_FEATURES / dataset / backbone
    z_id = np.load(base / f"{ID_TAGS[dataset]}_emb.npy").astype(np.float32)
    fc = torch.load(base / "trained_fc.pth", map_location="cpu", weights_only=True)
    W_orig = fc["weight"].numpy().astype(np.float32)
    b_orig = fc["bias"].numpy().astype(np.float32)
    z_oods = {}
    for od in ood_sets:
        p = POOLED_FEATURES / dataset / "ood" / od / backbone / f"{od}_emb.npy"
        if p.is_file():
            z_oods[od] = np.load(p).astype(np.float32)
    return z_id, W_orig, b_orig, z_oods

def list_heads(dataset, backbone):
    """Yield (head_name, W, b) -- always includes the original FC."""
    base = POOLED_FEATURES / dataset / backbone
    fc = torch.load(base / "trained_fc.pth", map_location="cpu", weights_only=True)
    yield "trained_fc", fc["weight"].numpy().astype(np.float32), fc["bias"].numpy().astype(np.float32)

    head_root = HEADS_DIR / dataset / backbone
    if head_root.is_dir():
        for p in sorted(head_root.glob("lam*_T*.pth")):
            d = torch.load(p, map_location="cpu", weights_only=True)
            yield p.stem, d["weight"].numpy().astype(np.float32), d["bias"].numpy().astype(np.float32)

def _logits(z, W, b):
    return z @ W.T + b

def score_one(z_id, z_ood, W, b, ash_p):
    """Return dict of detector -> (auroc, fpr95)."""
    out = {}

    v_id = _logits(z_id, W, b)
    v_ood = _logits(z_ood, W, b)

    s_id, s_ood = score_mls(v_id), score_mls(v_ood)
    out["mls"] = (auroc(s_id, s_ood), fpr95(s_id, s_ood))

    for variant in ("ash_s", "ash_b", "ash_p"):
        z_id_s = ash_mod.apply(z_id, name=variant, p=ash_p)
        z_ood_s = ash_mod.apply(z_ood, name=variant, p=ash_p)
        v_id_s = _logits(z_id_s, W, b)
        v_ood_s = _logits(z_ood_s, W, b)
        s_id, s_ood = score_mls(v_id_s), score_mls(v_ood_s)
        out[f"{variant}_mls"] = (auroc(s_id, s_ood), fpr95(s_id, s_ood))

    cos_all = np.concatenate([score_cosine(z_id, W), score_cosine(z_ood, W)])
    ene_all = np.concatenate([score_energy(v_id), score_energy(v_ood)])
    combined = score_cskd_dc(cos_all, ene_all)
    n_id = len(z_id)
    out["cskd_dc"] = (auroc(combined[:n_id], combined[n_id:]),
                      fpr95(combined[:n_id], combined[n_id:]))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(ID_TAGS))
    p.add_argument("--include", nargs="*", default=None,
                   help="Only score backbones whose name contains any of these substrings")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="Skip backbones whose name contains any of these substrings")
    p.add_argument("--ash-percentile", type=float, default=90.0)
    p.add_argument("--ood-sets", nargs="+", default=None,
                   help="By default, all OOD sets present under POOLED_FEATURES/{dataset}/ood/")
    p.add_argument("--out-csv", type=str, default=None)
    args = p.parse_args()

    ensure_dirs()
    backbones = discover_backbones(args.dataset, args.include, args.exclude)
    ood_sets = args.ood_sets or discover_ood_sets(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Backbones ({len(backbones)}): {backbones}")
    print(f"OOD sets ({len(ood_sets)}): {ood_sets}")
    if not backbones or not ood_sets:
        print("Nothing to score. Check POOLED_FEATURES paths.")
        return

    out_csv = Path(args.out_csv) if args.out_csv else (
        RESULTS_DIR / f"scores_{args.dataset}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backbone", "head", "ood_set", "detector", "auroc", "fpr95"])

        for bb in backbones:
            print(f"\n[{bb}]")
            z_id, _, _, z_oods = load_backbone(args.dataset, bb, ood_sets)
            for head_name, W, b in list_heads(args.dataset, bb):
                for od in ood_sets:
                    if od not in z_oods:
                        continue
                    res = score_one(z_id, z_oods[od], W, b, args.ash_percentile)
                    for det, (a, fp) in res.items():
                        writer.writerow([bb, head_name, od, det,
                                         f"{a:.6f}", f"{fp:.6f}"])
                    summary = "  ".join(f"{d}={v[0]:.3f}" for d, v in res.items())
                    print(f"  {head_name:<20} {od:<12}  {summary}")

    print(f"\nResults -> {out_csv}")

if __name__ == "__main__":
    main()
