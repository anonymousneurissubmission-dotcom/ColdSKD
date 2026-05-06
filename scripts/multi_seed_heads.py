"""
Train multi-seed (AugDelete + CSKD T=0.5) heads on pooled features for
CIFAR-100 / ImageNet-200, and aggregate into a `_ttest_results.json`
matching the schema used for the ImageNet-1K t-test artefacts shipped
under trained_heads/imagenet1k_ttest/.

Per backbone, trains:
    linear_ce  (lambda=0,   AugDelete baseline)
    linear_kd  (lambda=0.7, T=0.5, CSKD)
across N seeds (default {42, 43, 44}).

Outputs go to:
    trained_heads/{dataset}_ttest/{backbone}/
        linear_ce_seed{S}.pth
        linear_kd_seed{S}.pth
        {backbone}_ttest_results.json

Usage:
    python scripts/multi_seed_heads.py --gpu 0 --dataset cifar100
    python scripts/multi_seed_heads.py --gpu 0 --dataset imagenet200
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import POOLED_FEATURES
sys.path.insert(0, str(ROOT / "scoring"))
from ood_scores import compute_all_scores, auroc, fpr95

DEST = ROOT / "trained_heads"

DATASETS = {
    "cifar100": {
        "num_classes": 100,
        "id_train_tag": "cifar100_train",
        "id_test_tag": "cifar100_test",
        "ood_sets": ["cifar10", "svhn", "textures",
                     "places", "lsun_c", "lsun_r", "isun"],
        "default_backbones": [f"resnet18_s{s}_ls_{a}"
                              for s in (0, 1, 2)
                              for a in ("0.0", "0.1", "0.2", "0.3")],
    },
    "imagenet200": {
        "num_classes": 200,
        "id_train_tag": "imagenet200_train",
        "id_test_tag": "imagenet200_test",
        "ood_sets": ["inaturalist", "openimage_o", "textures",
                     "ssb_hard", "ninco"],
        "default_backbones": (
            [f"baseline_s{s}" for s in (0, 1, 2)] +
            [f"ls_{a}_s{s}" for s in (0, 1, 2)
                            for a in ("0.1", "0.2", "0.3")] +
            [f"mixup_1.0_s{s}" for s in (0, 1, 2)] +
            [f"cutmix_1.0_s{s}" for s in (0, 1, 2)] +
            [f"ls_0.1_cutmix_1.0_s{s}" for s in (0, 1, 2)]
        ),
    },
}

DEFAULT_CONFIG = {
    "epochs": 50,
    "lr": 0.1,
    "cos_lr": 0.05,
    "kd_alpha": 0.7,
    "kd_temp": 0.5,
}

def load_backbone(dataset, backbone, ood_sets):
    base = POOLED_FEATURES / dataset / backbone
    cfg = DATASETS[dataset]
    z_train = np.load(base / f"{cfg['id_train_tag']}_emb.npy").astype(np.float32)
    y_train = np.load(base / f"{cfg['id_train_tag']}_lbl.npy").astype(np.int64)
    z_val = np.load(base / f"{cfg['id_test_tag']}_emb.npy").astype(np.float32)
    fc = torch.load(base / "trained_fc.pth", map_location="cpu", weights_only=True)
    teacher_W = fc["weight"].numpy().astype(np.float32)
    teacher_b = fc["bias"].numpy().astype(np.float32)
    z_oods = {}
    for od in ood_sets:
        p = POOLED_FEATURES / dataset / "ood" / od / backbone / f"{od}_emb.npy"
        if p.is_file():
            z_oods[od] = np.load(p).astype(np.float32)
        else:
            print(f"    [skip OOD] missing {p}")
    return z_train, y_train, z_val, teacher_W, teacher_b, z_oods

def train_one_head(z_train, y_train, teacher_W, teacher_b, num_classes,
                   alpha, T, seed, device, epochs, lr, cos_lr, batch_size=512):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    N, D = z_train.shape

    z_t = torch.from_numpy(z_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    W_t = torch.from_numpy(teacher_W).to(device)
    b_t = torch.from_numpy(teacher_b).to(device)

    head = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=cos_lr)
    ce = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)

    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(N)).to(device)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            emb = z_t[idx]
            lbl = y_t[idx]
            with torch.no_grad():
                t_logits = emb @ W_t.T + b_t
            s_logits = head(emb)
            loss_ce = ce(s_logits, lbl)
            if alpha > 0:
                log_s = F.log_softmax(s_logits / T, dim=1)
                soft_t = F.softmax(t_logits / T, dim=1)
                loss_kd = F.kl_div(log_s, soft_t, reduction="batchmean") * (T ** 2)
                loss = (1 - alpha) * loss_ce + alpha * loss_kd
            else:
                loss = loss_ce
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    head.cpu().eval()
    return head.weight.detach().numpy(), head.bias.detach().numpy()

def eval_head(W, b, z_id, y_id, z_oods):
    raw_id = z_id @ W.T + b
    scores_id = compute_all_scores(raw_id, z_id, W, b)
    accuracy = float((raw_id.argmax(1) == y_id).mean())
    per_ood = {}
    mls_aurocs, mls_fprs, ene_aurocs = [], [], []
    for od, z_ood in z_oods.items():
        raw_ood = z_ood @ W.T + b
        scores_ood = compute_all_scores(raw_ood, z_ood, W, b)
        a_mls = auroc(scores_id["mls"], scores_ood["mls"])
        f_mls = fpr95(scores_id["mls"], scores_ood["mls"])
        a_ene = auroc(scores_id["energy"], scores_ood["energy"])
        per_ood[od] = {
            "mls_auroc": a_mls,
            "mls_fpr95": f_mls,
            "energy_auroc": a_ene,
        }
        mls_aurocs.append(a_mls)
        mls_fprs.append(f_mls)
        ene_aurocs.append(a_ene)
    return {
        "accuracy": accuracy,
        "mls_mean_auroc": float(np.mean(mls_aurocs)) if mls_aurocs else None,
        "mls_mean_fpr95": float(np.mean(mls_fprs)) if mls_fprs else None,
        "energy_mean_auroc": float(np.mean(ene_aurocs)) if ene_aurocs else None,
        "per_ood": per_ood,
    }

def run_backbone(dataset, backbone, seeds, device, cfg):
    ds = DATASETS[dataset]
    out_dir = DEST / f"{dataset}_ttest" / backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{backbone}_ttest_results.json"

    if json_path.is_file():
        results = json.load(open(json_path))
        results.setdefault("heads", {})
    else:
        results = {
            "model": backbone,
            "dataset": dataset,
            "n_seeds": len(seeds),
            "seeds": list(seeds),
            "config": cfg,
            "heads": {"linear_ce": [], "linear_kd": []},
        }

    z_train, y_train, z_val, tW, tb, z_oods = load_backbone(
        dataset, backbone, ds["ood_sets"])
    y_val = np.load(POOLED_FEATURES / dataset / backbone /
                    f"{ds['id_test_tag']}_lbl.npy").astype(np.int64)

    head_specs = [
        ("linear_ce", 0.0, 1.0),
        ("linear_kd", cfg["kd_alpha"], cfg["kd_temp"]),
    ]

    for head_name, alpha, T in head_specs:
        results["heads"].setdefault(head_name, [])
        already_done_seeds = {
            s for s, entry in zip(results["seeds"], results["heads"][head_name])
            if isinstance(entry, dict)
        }
        for seed in seeds:
            ck_path = out_dir / f"{head_name}_seed{seed}.pth"
            if ck_path.is_file() and seed in already_done_seeds:
                print(f"    {head_name} seed={seed}: cached")
                continue
            t0 = time.time()
            W, b = train_one_head(
                z_train, y_train, tW, tb, ds["num_classes"],
                alpha, T, seed, device,
                cfg["epochs"], cfg["lr"], cos_lr=cfg["cos_lr"])
            torch.save({"weight": torch.from_numpy(W),
                        "bias": torch.from_numpy(b),
                        "alpha": alpha, "T": T, "seed": seed}, ck_path)
            metrics = eval_head(W, b, z_val, y_val, z_oods)
            metrics["seed"] = seed
            replaced = False
            for i, e in enumerate(results["heads"][head_name]):
                if isinstance(e, dict) and e.get("seed") == seed:
                    results["heads"][head_name][i] = metrics
                    replaced = True
                    break
            if not replaced:
                results["heads"][head_name].append(metrics)
            print(f"    {head_name} seed={seed}  acc={metrics['accuracy']:.4f}  "
                  f"MLS={metrics['mls_mean_auroc']:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    for hn in results["heads"]:
        results["heads"][hn].sort(key=lambda e: e.get("seed", 0))

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> {json_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--backbones", nargs="+", default=None,
                   help="Default = canonical list for the dataset")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--cos-lr", type=float, default=DEFAULT_CONFIG["cos_lr"])
    p.add_argument("--kd-alpha", type=float, default=DEFAULT_CONFIG["kd_alpha"])
    p.add_argument("--kd-temp", type=float, default=DEFAULT_CONFIG["kd_temp"])
    args = p.parse_args()

    cfg = {"epochs": args.epochs, "lr": args.lr, "cos_lr": args.cos_lr,
           "kd_alpha": args.kd_alpha, "kd_temp": args.kd_temp}
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  config: {cfg}  seeds: {args.seeds}")

    backbones = args.backbones or DATASETS[args.dataset]["default_backbones"]
    print(f"Dataset: {args.dataset}  backbones ({len(backbones)}): {backbones}")

    for bb in backbones:
        base = POOLED_FEATURES / args.dataset / bb
        if not base.is_dir():
            print(f"\n[skip {bb}] {base} missing")
            continue
        print(f"\n[{bb}]")
        run_backbone(args.dataset, bb, args.seeds, device, cfg)

    print("\nDone.")

if __name__ == "__main__":
    main()
