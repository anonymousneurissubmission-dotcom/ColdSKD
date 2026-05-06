"""
Train ResNet-50 on ImageNet-200 with configurable label smoothing.

Pretrained ImageNet-1k V1 init, 200 epochs full fine-tune, AMP.
Resumable per-run via last.ckpt + metrics.csv.

Run:
    python trainers/train_imagenet200.py --gpu 0 --ls-values 0.0 0.1 0.2 0.3
"""
import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (IMAGENET_TRAIN, IMAGENET_VAL, IMAGENET200_IMGLIST,
                    TRAINED_MODELS, ensure_dirs)

NUM_CLASSES = 200
LABEL_SMOOTHING_VALUES = [0.0, 0.1, 0.2, 0.3]
EPOCHS = 200
LR = 0.1
WD = 1e-4
MOMENTUM = 0.9
BATCH_SIZE = 128
NUM_WORKERS = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_val_index(val_root: Path) -> dict:
    idx = {}
    for wnid in os.listdir(val_root):
        wdir = val_root / wnid
        if not wdir.is_dir():
            continue
        for fname in os.listdir(wdir):
            idx[fname] = str(wdir / fname)
    print(f"  indexed {len(idx)} val files under {val_root}")
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
                label = int(label)
                if rel.startswith("imagenet_1k/train/"):
                    abs_path = os.path.join(
                        str(train_root), rel[len("imagenet_1k/train/"):])
                elif rel.startswith("imagenet_1k/val/"):
                    fname = rel[len("imagenet_1k/val/"):]
                    abs_path = val_index[fname]
                else:
                    raise ValueError(f"unexpected imglist prefix: {rel}")
                self.samples.append((abs_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def build_loaders(batch_size, num_workers):
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_index = build_val_index(IMAGENET_VAL)

    train_ds = ImglistDataset(
        IMAGENET200_IMGLIST / "train_imagenet200.txt",
        IMAGENET_TRAIN, val_index, train_tfm)
    val_ds = ImglistDataset(
        IMAGENET200_IMGLIST / "val_imagenet200.txt",
        IMAGENET_TRAIN, val_index, eval_tfm)
    print(f"  train {len(train_ds)}  val {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True)
    return train_loader, val_loader

def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    nn.init.kaiming_normal_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    return model

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = inputs.size(0)
        total_loss += loss.item() * bs
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += bs
    return total_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        bs = inputs.size(0)
        total_loss += loss.item() * bs
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += bs
    return total_loss / total, 100.0 * correct / total

def is_complete(run_dir: Path, epochs: int) -> bool:
    if not (run_dir / "last.ckpt").is_file() or not (run_dir / "metrics.csv").is_file():
        return False
    with open(run_dir / "metrics.csv") as f:
        return sum(1 for _ in f) >= epochs + 1

def load_resume(run_dir, model, optimizer, scheduler, scaler, device):
    p = run_dir / "last.ckpt"
    if not p.is_file():
        return 1, 0.0
    ckpt = torch.load(p, map_location=device, weights_only=False)
    if "optimizer" not in ckpt:
        return 1, 0.0
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"  [resume] {run_dir} from epoch {ckpt['epoch']+1} "
          f"(best {ckpt.get('best_test_acc', 0.0):.2f}%)")
    return int(ckpt["epoch"]) + 1, float(ckpt.get("best_test_acc", 0.0))

def train_run(ls, train_loader, val_loader, device, output_dir, epochs):
    run_dir = output_dir / f"ls_{ls}"
    if is_complete(run_dir, epochs):
        ckpt = torch.load(run_dir / "last.ckpt", map_location="cpu",
                          weights_only=False)
        best = ckpt.get("best_test_acc", ckpt.get("test_acc", 0.0))
        print(f"  [skip] ls_{ls} complete (best {best:.2f}%)")
        return best

    print(f"\n{'='*60}\n  ResNet-50 ImageNet-200  |  LS={ls}  |  {epochs} epochs\n{'='*60}")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                    weight_decay=WD, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.GradScaler("cuda")

    start, best = load_resume(run_dir, model, optimizer, scheduler, scaler, device)
    if start == 1:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                    "test_loss", "test_acc", "lr"])

    test_acc = best
    for epoch in range(start, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        test_acc = va

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tl:.4f}", f"{ta:.2f}",
                                    f"{vl:.4f}", f"{va:.2f}", f"{lr:.6f}"])
        if va > best:
            best = va
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "test_acc": va}, run_dir / "best.ckpt")

        print(f"  Ep {epoch:3d}/{epochs}  train {ta:.2f}%  test {va:.2f}%  best {best:.2f}%  lr {lr:.5f}")

        blob = {"state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch, "label_smoothing": ls,
                "test_acc": va, "best_test_acc": best}
        tmp = run_dir / "last.ckpt.tmp"
        torch.save(blob, tmp)
        os.replace(tmp, run_dir / "last.ckpt")

    print(f"  done -- best {best:.2f}%  final {test_acc:.2f}%")
    return best

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ls-values", nargs="+", type=float,
                   default=LABEL_SMOOTHING_VALUES)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--output-dir", type=str,
                   default=str(TRAINED_MODELS / "imagenet200"))
    args = p.parse_args()

    ensure_dirs()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_loader, val_loader = build_loaders(args.batch_size, args.num_workers)
    output_dir = Path(args.output_dir)

    results = {}
    for ls in args.ls_values:
        results[ls] = train_run(ls, train_loader, val_loader, device,
                                output_dir, args.epochs)

    print(f"\n{'='*60}\n  SUMMARY (best test acc %)\n{'='*60}")
    for ls, acc in results.items():
        print(f"  LS={ls:<5}  {acc:.2f}%")

if __name__ == "__main__":
    main()
