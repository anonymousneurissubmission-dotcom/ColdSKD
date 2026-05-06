"""
Extract pooled embeddings + classification head for trained CIFAR-100 ResNet18s.

Walks every checkpoint under TRAINED_MODELS/cifar100/s{seed}/ls_{ls}/last.ckpt,
saves to POOLED_FEATURES/cifar100/{run_name}/:
    cifar100_train_{emb,lbl}.npy
    cifar100_test_{emb,lbl}.npy
    trained_fc.pth
plus per-OOD subfolders.

Run:
    python extraction/extract_cifar100.py --gpu 0
"""
import sys
import argparse
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (DATA_ROOT, OOD_PATHS, TRAINED_MODELS, POOLED_FEATURES,
                    ensure_dirs)
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "extraction"))
from resnet18_32x32 import ResNet18_32x32
from _common import FlatImages, extract_with_hook, save_features, save_fc, already_done

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]
BS = 256
NW = 4

def cifar100_loaders():
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    tf32 = T.Compose([T.Resize(32), T.CenterCrop(32), T.ToTensor(),
                      T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    train = torchvision.datasets.CIFAR100(str(DATA_ROOT), train=True,
                                          download=True, transform=tf)
    test = torchvision.datasets.CIFAR100(str(DATA_ROOT), train=False,
                                         download=True, transform=tf)
    train_loader = DataLoader(train, batch_size=BS, num_workers=NW, pin_memory=True)
    test_loader = DataLoader(test, batch_size=BS, num_workers=NW, pin_memory=True)
    print(f"  CIFAR-100 train {len(train)}  test {len(test)}")

    ood_loaders = {}
    candidates = [
        ("cifar10",    lambda: torchvision.datasets.CIFAR10(
            str(DATA_ROOT), train=False, download=True, transform=tf)),
        ("svhn",       lambda: torchvision.datasets.SVHN(
            str(OOD_PATHS["svhn"]), split="test", download=True, transform=tf)),
        ("textures",   lambda: torchvision.datasets.ImageFolder(
            str(OOD_PATHS["textures"]), transform=tf32)),
        ("places",     lambda: FlatImages(OOD_PATHS["places"], tf32)),
        ("lsun_c",     lambda: FlatImages(OOD_PATHS["lsun_c"], tf32)),
        ("lsun_r",     lambda: FlatImages(OOD_PATHS["lsun_r"], tf32)),
        ("isun",       lambda: FlatImages(OOD_PATHS["isun"], tf32)),
    ]
    for name, fn in candidates:
        try:
            ds = fn()
            ood_loaders[name] = DataLoader(ds, batch_size=BS, num_workers=NW, pin_memory=True)
            print(f"  OOD {name}: {len(ds)}")
        except Exception as e:
            print(f"  OOD {name}: SKIPPED ({e})")
    return train_loader, test_loader, ood_loaders

def discover_runs(root: Path):
    runs = []
    if not root.exists():
        return runs
    for seed_dir in sorted(root.glob("s*")):
        seed = int(seed_dir.name[1:])
        for cond_dir in sorted(seed_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            ckpt = cond_dir / "last.ckpt"
            if ckpt.is_file():
                runs.append((seed, cond_dir.name, ckpt))
    return runs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--trained-dir", type=str,
                   default=str(TRAINED_MODELS / "cifar100"))
    p.add_argument("--out-dir", type=str,
                   default=str(POOLED_FEATURES / "cifar100"))
    args = p.parse_args()

    ensure_dirs()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, ood_loaders = cifar100_loaders()
    out_root = Path(args.out_dir)
    runs = discover_runs(Path(args.trained_dir))
    print(f"Found {len(runs)} checkpoints under {args.trained_dir}")

    for seed, cond, ckpt_path in runs:
        run_name = f"resnet18_s{seed}_{cond}"
        run_dir = out_root / run_name
        print(f"\n[{run_name}]")

        model = ResNet18_32x32(num_classes=100)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        model.to(device).eval()

        if not (run_dir / "trained_fc.pth").is_file():
            save_fc(run_dir, model.fc)

        for tag, loader in [("cifar100_train", train_loader),
                            ("cifar100_test", test_loader)]:
            if already_done(run_dir, tag):
                print(f"  {tag}: cached")
                continue
            print(f"  {tag}: extracting...", end=" ", flush=True)
            emb, lbl = extract_with_hook(model, loader, device, model.avgpool)
            save_features(run_dir, tag, emb, lbl)
            print(f"({emb.shape})")

        for ood_name, ood_loader in ood_loaders.items():
            ood_dir = out_root / "ood" / ood_name / run_name
            if already_done(ood_dir, ood_name):
                continue
            print(f"  OOD {ood_name}...", end=" ", flush=True)
            emb, lbl = extract_with_hook(model, ood_loader, device, model.avgpool)
            save_features(ood_dir, ood_name, emb, lbl)
            print(f"({emb.shape})")

        del model
        torch.cuda.empty_cache()

    print("\nDone.")

if __name__ == "__main__":
    main()
