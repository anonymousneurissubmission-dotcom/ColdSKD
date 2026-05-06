"""
Extract pooled embeddings + classification head for trained ImageNet-200 ResNet50s.

Walks every checkpoint under TRAINED_MODELS/imagenet200/ls_{ls}/last.ckpt,
saves to POOLED_FEATURES/imagenet200/{run_name}/:
    imagenet200_train_{emb,lbl}.npy
    imagenet200_test_{emb,lbl}.npy
    trained_fc.pth
plus per-OOD subfolders.

Run:
    python extraction/extract_imagenet200.py --gpu 0
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (IMAGENET_TRAIN, IMAGENET_VAL, IMAGENET200_IMGLIST,
                    OOD_PATHS, TRAINED_MODELS, POOLED_FEATURES, ensure_dirs)
sys.path.insert(0, str(ROOT / "extraction"))
from _common import FlatImages, extract_with_hook, save_features, save_fc, already_done

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BS = 128
NW = 8

def build_val_index(val_root):
    idx = {}
    for wnid in os.listdir(val_root):
        wdir = Path(val_root) / wnid
        if not wdir.is_dir():
            continue
        for fname in os.listdir(wdir):
            idx[fname] = str(wdir / fname)
    return idx

class ImglistDataset(Dataset):
    def __init__(self, imglist_path, train_root, val_index, transform):
        self.transform = transform
        self.samples = []
        with open(imglist_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, label = line.rsplit(" ", 1)
                if rel.startswith("imagenet_1k/train/"):
                    abs_path = os.path.join(str(train_root), rel[len("imagenet_1k/train/"):])
                elif rel.startswith("imagenet_1k/val/"):
                    abs_path = val_index[rel[len("imagenet_1k/val/"):]]
                else:
                    raise ValueError(f"unexpected prefix: {rel}")
                self.samples.append((abs_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label

def imagenet200_loaders():
    eval_tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_index = build_val_index(IMAGENET_VAL)

    train_ds = ImglistDataset(IMAGENET200_IMGLIST / "train_imagenet200.txt",
                              IMAGENET_TRAIN, val_index, eval_tfm)
    val_ds = ImglistDataset(IMAGENET200_IMGLIST / "val_imagenet200.txt",
                            IMAGENET_TRAIN, val_index, eval_tfm)
    train_loader = DataLoader(train_ds, batch_size=BS, num_workers=NW, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BS, num_workers=NW, pin_memory=True)
    print(f"  ImageNet-200 train {len(train_ds)}  val {len(val_ds)}")

    ood_loaders = {}
    candidates = [
        ("inaturalist", lambda: FlatImages(OOD_PATHS["inaturalist"], eval_tfm)),
        ("openimage_o", lambda: FlatImages(OOD_PATHS["openimage_o"], eval_tfm)),
        ("textures",    lambda: FlatImages(OOD_PATHS["textures"], eval_tfm)),
        ("ssb_hard",    lambda: FlatImages(OOD_PATHS["ssb_hard"], eval_tfm)),
        ("ninco",       lambda: FlatImages(OOD_PATHS["ninco"], eval_tfm)),
    ]
    for name, fn in candidates:
        try:
            ds = fn()
            ood_loaders[name] = DataLoader(ds, batch_size=BS, num_workers=NW, pin_memory=True)
            print(f"  OOD {name}: {len(ds)}")
        except Exception as e:
            print(f"  OOD {name}: SKIPPED ({e})")
    return train_loader, val_loader, ood_loaders

def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def discover_runs(root: Path):
    runs = []
    if not root.exists():
        return runs
    for cond_dir in sorted(root.iterdir()):
        if not cond_dir.is_dir():
            continue
        ckpt = cond_dir / "last.ckpt"
        if ckpt.is_file():
            runs.append((cond_dir.name, ckpt))
    return runs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--trained-dir", type=str,
                   default=str(TRAINED_MODELS / "imagenet200"))
    p.add_argument("--out-dir", type=str,
                   default=str(POOLED_FEATURES / "imagenet200"))
    args = p.parse_args()

    ensure_dirs()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, ood_loaders = imagenet200_loaders()
    out_root = Path(args.out_dir)
    runs = discover_runs(Path(args.trained_dir))
    print(f"Found {len(runs)} checkpoints under {args.trained_dir}")

    for cond, ckpt_path in runs:
        run_name = f"resnet50_{cond}"
        run_dir = out_root / run_name
        print(f"\n[{run_name}]")

        model = build_resnet50(num_classes=200)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        model.to(device).eval()

        if not (run_dir / "trained_fc.pth").is_file():
            save_fc(run_dir, model.fc)

        for tag, loader in [("imagenet200_train", train_loader),
                            ("imagenet200_test", val_loader)]:
            if already_done(run_dir, tag):
                print(f"  {tag}: cached")
                continue
            print(f"  {tag}: extracting...", flush=True)
            emb, lbl = extract_with_hook(model, loader, device, model.avgpool)
            save_features(run_dir, tag, emb, lbl)
            print(f"    -> {emb.shape}")

        for ood_name, ood_loader in ood_loaders.items():
            ood_dir = out_root / "ood" / ood_name / run_name
            if already_done(ood_dir, ood_name):
                continue
            print(f"  OOD {ood_name}...", flush=True)
            emb, lbl = extract_with_hook(model, ood_loader, device, model.avgpool)
            save_features(ood_dir, ood_name, emb, lbl)
            print(f"    -> {emb.shape}")

        del model
        torch.cuda.empty_cache()

    print("\nDone.")

if __name__ == "__main__":
    main()
