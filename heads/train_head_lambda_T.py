"""
Train a single classification head with cold self-distillation:

    L = (1 - lambda) * CE(student, y) + lambda * T^2 * KL(softmax(student/T) || softmax(teacher/T))

Reads pre-extracted pooled features from POOLED_FEATURES/{dataset}/{backbone}/.
Teacher = the original `trained_fc.pth` (frozen). Student = a fresh nn.Linear.

Run:
    python heads/train_head_lambda_T.py --gpu 0 \
        --dataset imagenet200 --backbone resnet50_ls_0.1 --lam 0.7 --T 0.1
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
sys.path.insert(0, str(ROOT / "scoring"))
from ood_scores import compute_all_scores, compute_cosene, auroc

DEFAULTS = {
    "cifar100":    {"num_classes": 100, "id_train": "cifar100_train",    "id_test": "cifar100_test"},
    "imagenet200": {"num_classes": 200, "id_train": "imagenet200_train", "id_test": "imagenet200_test"},
    "imagenet1k":  {"num_classes": 1000, "id_train": "imagenet1k_train",  "id_test": "imagenet1k_val"},
}

def load_pooled(dataset, backbone, ood_sets):
    base = POOLED_FEATURES / dataset / backbone
    cfg = DEFAULTS[dataset]

    z_train = np.load(base / f"{cfg['id_train']}_emb.npy").astype(np.float32)
    y_train = np.load(base / f"{cfg['id_train']}_lbl.npy").astype(np.int64)
    z_val = np.load(base / f"{cfg['id_test']}_emb.npy").astype(np.float32)
    y_val = np.load(base / f"{cfg['id_test']}_lbl.npy").astype(np.int64)

    fc = torch.load(base / "trained_fc.pth", map_location="cpu", weights_only=True)

    z_oods = {}
    for od in ood_sets:
        p = POOLED_FEATURES / dataset / "ood" / od / backbone / f"{od}_emb.npy"
        if p.is_file():
            z_oods[od] = np.load(p).astype(np.float32)
        else:
            print(f"  [skip OOD] missing {p}")
    return z_train, y_train, z_val, y_val, z_oods, fc["weight"].numpy(), fc["bias"].numpy()

def evaluate_head(W, b, z_id, z_oods):
    raw_id = z_id @ W.T + b
    scores_id = compute_all_scores(raw_id, z_id, W, b)
    out = {}
    for od, z_ood in z_oods.items():
        raw_ood = z_ood @ W.T + b
        scores_ood = compute_all_scores(raw_ood, z_ood, W, b)
        cos_id, cos_ood = compute_cosene(scores_id, scores_ood)
        out[od] = {
            "mls":     auroc(scores_id["mls"],     scores_ood["mls"]),
            "energy":  auroc(scores_id["energy"],  scores_ood["energy"]),
            "cosine":  auroc(scores_id["cosine"],  scores_ood["cosine"]),
            "cskd_dc": auroc(cos_id, cos_ood),
        }
    avg = {k: float(np.mean([v[k] for v in out.values()]))
           for k in ["mls", "energy", "cosine", "cskd_dc"]}
    return avg, out

def train_head(z_train, y_train, teacher_W, teacher_b, num_classes, lam, T,
               epochs, lr, bs, seed, device):
    torch.manual_seed(seed)
    N, D = z_train.shape
    z_t = torch.from_numpy(z_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    W_t = torch.from_numpy(teacher_W).to(device)
    b_t = torch.from_numpy(teacher_b).to(device)

    head = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)

    print(f"  training head lambda={lam} T={T} for {epochs} epochs ({N} samples, D={D})")
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(N)).to(device)
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            emb = z_t[idx]
            lbl = y_t[idx]
            with torch.no_grad():
                t_logits = emb @ W_t.T + b_t
            s_logits = head(emb)
            loss_ce = ce(s_logits, lbl)
            if lam > 0:
                log_s = F.log_softmax(s_logits / T, dim=1)
                soft_t = F.softmax(t_logits / T, dim=1)
                loss_kd = F.kl_div(log_s, soft_t, reduction="batchmean") * (T ** 2)
                loss = (1 - lam) * loss_ce + lam * loss_kd
            else:
                loss = loss_ce
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep+1}/{epochs}  ({time.time()-t0:.1f}s)")
    head.cpu().eval()
    return head.weight.detach().numpy(), head.bias.detach().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dataset", required=True, choices=list(DEFAULTS.keys()))
    p.add_argument("--backbone", required=True, help="folder name under POOLED_FEATURES/{dataset}/")
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--ood-sets", nargs="+",
                   default=["inaturalist", "openimage_o", "textures", "ssb_hard"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ensure_dirs()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    z_tr, y_tr, z_val, y_val, z_oods, tW, tb = load_pooled(
        args.dataset, args.backbone, args.ood_sets)
    print(f"Train {z_tr.shape}  Val {z_val.shape}  OOD {list(z_oods)}")

    cfg = DEFAULTS[args.dataset]
    W, b = train_head(z_tr, y_tr, tW, tb, cfg["num_classes"],
                      args.lam, args.T, args.epochs, args.lr,
                      args.batch_size, args.seed, device)

    head_dir = HEADS_DIR / args.dataset / args.backbone
    head_dir.mkdir(parents=True, exist_ok=True)
    head_path = head_dir / f"lam{args.lam}_T{args.T}.pth"
    torch.save({"weight": torch.from_numpy(W), "bias": torch.from_numpy(b),
                "lambda": args.lam, "T": args.T}, head_path)
    print(f"head saved -> {head_path}")

    avg, per_ood = evaluate_head(W, b, z_val, z_oods)
    print(f"AUROC  MLS={avg['mls']:.4f}  Ene={avg['energy']:.4f}  "
          f"Cos={avg['cosine']:.4f}  CSKD-DC={avg['cskd_dc']:.4f}")

    out_json = RESULTS_DIR / f"head_eval_{args.dataset}.json"
    results = json.loads(out_json.read_text()) if out_json.is_file() else {}
    results.setdefault(args.backbone, {})[f"lam{args.lam}_T{args.T}"] = {
        "lambda": args.lam, "T": args.T, "avg": avg, "per_ood": per_ood,
    }
    out_json.write_text(json.dumps(results, indent=2))
    print(f"results -> {out_json}")

if __name__ == "__main__":
    main()
