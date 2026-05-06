"""
lambda x T grid sweep on pooled features (parallel heads).

Trains one head per (lambda, T) pair simultaneously: shared batch, per-head loss,
single backward over the concatenated parameter list. Resumable: heads already
on disk are loaded from cache.

Run:
    python heads/lambda_T_sweep.py --gpu 0 \
        --dataset imagenet200 --backbone resnet50_ls_0.1
"""
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
from config import POOLED_FEATURES, HEADS_DIR, RESULTS_DIR, ensure_dirs
sys.path.insert(0, str(ROOT / "heads"))
from train_head_lambda_T import load_pooled, evaluate_head, DEFAULTS

LAMBDAS_DEFAULT = [0.3, 0.5, 0.7, 0.9]
TEMPS_DEFAULT = [0.01, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dataset", required=True, choices=list(DEFAULTS.keys()))
    p.add_argument("--backbone", required=True)
    p.add_argument("--ood-sets", nargs="+",
                   default=["inaturalist", "openimage_o", "textures", "ssb_hard"])
    p.add_argument("--lambdas", nargs="+", type=float, default=LAMBDAS_DEFAULT)
    p.add_argument("--temperatures", nargs="+", type=float, default=TEMPS_DEFAULT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ensure_dirs()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    cfg = DEFAULTS[args.dataset]
    head_dir = HEADS_DIR / args.dataset / args.backbone
    head_dir.mkdir(parents=True, exist_ok=True)
    results_json = RESULTS_DIR / f"lambda_T_sweep_{args.dataset}_{args.backbone}.json"

    print(f"Backbone: {args.backbone}  Device: {device}")
    z_tr, y_tr, z_val, y_val, z_oods, tW, tb = load_pooled(
        args.dataset, args.backbone, args.ood_sets)
    print(f"Train {z_tr.shape}  Val {z_val.shape}  OOD {list(z_oods)}")

    N, D = z_tr.shape
    z_t = torch.from_numpy(z_tr).to(device)
    y_t = torch.from_numpy(y_tr).to(device)
    W_t = torch.from_numpy(tW).to(device)
    b_t = torch.from_numpy(tb).to(device)

    grid = [(0.0, 1.0)] + [(l, T) for l in args.lambdas for T in args.temperatures]

    cached, train_grid = {}, []
    for (lam, T) in grid:
        path = head_dir / f"lam{lam}_T{T}.pth"
        if path.is_file():
            d = torch.load(path, map_location="cpu", weights_only=True)
            cached[(lam, T)] = (d["weight"].numpy(), d["bias"].numpy())
        else:
            train_grid.append((lam, T))
    print(f"Cache: {len(cached)} hits / {len(train_grid)} to train")

    if train_grid:
        torch.manual_seed(args.seed)
        heads = nn.ModuleList(
            [nn.Linear(D, cfg["num_classes"]) for _ in train_grid]).to(device)
        with torch.no_grad():
            ref_w = heads[0].weight.clone()
            ref_b = heads[0].bias.clone()
            for h in heads[1:]:
                h.weight.copy_(ref_w)
                h.bias.copy_(ref_b)
        opt = torch.optim.SGD(heads.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        ce = nn.CrossEntropyLoss()
        rng = np.random.RandomState(args.seed)

        print(f"Training {len(heads)} heads in parallel for {args.epochs} epochs...")
        for ep in range(args.epochs):
            ep_t = time.time()
            perm = torch.from_numpy(rng.permutation(N)).to(device)
            for s in range(0, N, args.batch_size):
                idx = perm[s:s + args.batch_size]
                emb = z_t[idx]
                lbl = y_t[idx]
                with torch.no_grad():
                    t_logits = emb @ W_t.T + b_t

                total = 0.0
                for h, (lam, T) in zip(heads, train_grid):
                    logits = h(emb)
                    loss_ce = ce(logits, lbl)
                    if lam > 0:
                        log_s = F.log_softmax(logits / T, dim=1)
                        soft_t = F.softmax(t_logits / T, dim=1)
                        loss_kd = F.kl_div(log_s, soft_t, reduction="batchmean") * (T ** 2)
                        total = total + (1 - lam) * loss_ce + lam * loss_kd
                    else:
                        total = total + loss_ce
                opt.zero_grad()
                total.backward()
                opt.step()
            sched.step()
            print(f"  epoch {ep+1}/{args.epochs}  ({time.time()-ep_t:.1f}s)")

        for h, (lam, T) in zip(heads, train_grid):
            h_cpu = h.cpu().eval()
            W = h_cpu.weight.detach()
            b = h_cpu.bias.detach()
            torch.save({"weight": W, "bias": b, "lambda": lam, "T": T},
                       head_dir / f"lam{lam}_T{T}.pth")
            cached[(lam, T)] = (W.numpy(), b.numpy())

    print("\nEvaluating...")
    results = json.loads(results_json.read_text()) if results_json.is_file() else {}

    avg_t, per_ood_t = evaluate_head(tW, tb, z_val, z_oods)
    results["teacher_orig"] = {"lambda": None, "T": None,
                               "avg": avg_t, "per_ood": per_ood_t}
    print(f"  teacher_orig         MLS={avg_t['mls']:.4f}  Ene={avg_t['energy']:.4f}  "
          f"Cos={avg_t['cosine']:.4f}  CSKD-DC={avg_t['cskd_dc']:.4f}")

    for (lam, T) in grid:
        W, b = cached[(lam, T)]
        avg, per_ood = evaluate_head(W, b, z_val, z_oods)
        results[f"lam{lam}_T{T}"] = {"lambda": lam, "T": T,
                                     "avg": avg, "per_ood": per_ood}
        print(f"  lambda={lam}  T={T:<5}  MLS={avg['mls']:.4f}  Ene={avg['energy']:.4f}  "
              f"Cos={avg['cosine']:.4f}  CSKD-DC={avg['cskd_dc']:.4f}")

    results_json.write_text(json.dumps(results, indent=2))
    print(f"\nDone. Total: {time.time()-t0:.0f}s")
    print(f"Heads:   {head_dir}")
    print(f"Results: {results_json}")

if __name__ == "__main__":
    main()
