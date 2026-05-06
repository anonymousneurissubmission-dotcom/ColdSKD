"""
Train ResNet18 on CIFAR-100 *from scratch* with configurable label smoothing.

200 epochs, SGD + Nesterov, CosineAnnealingLR, on-GPU augmentations.

Run:
    python trainers/train_cifar100.py --gpu 0 --seeds 0 1 2 \
        --ls-values 0.0 0.1 0.2 0.3
"""
import os
import sys
import csv
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import kornia.augmentation as K

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CIFAR100_DIR, TRAINED_MODELS, ensure_dirs
sys.path.insert(0, str(ROOT / "models"))
from resnet18_32x32 import ResNet18_32x32

SEEDS = [0, 1, 2]
LABEL_SMOOTHING_VALUES = [0.0, 0.1, 0.2, 0.3]
NUM_CLASSES = 100
EPOCHS = 200
LR = 0.1
WD = 5e-4
MOMENTUM = 0.9
BATCH_SIZE = 256

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

class VRAMTensorData:
    """Whole dataset on GPU as one tensor; yields batches via indexing."""

    def __init__(self, dataset, device):
        imgs, labels = zip(*[(img, label) for img, label in dataset])
        self.data = torch.stack(imgs).to(device)
        self.targets = torch.tensor(labels, dtype=torch.long, device=device)

    def batches(self, batch_size, shuffle=False):
        n = self.data.size(0)
        if shuffle:
            idx = torch.randperm(n, device=self.data.device)
            data, targets = self.data[idx], self.targets[idx]
        else:
            data, targets = self.data, self.targets
        for i in range(0, n, batch_size):
            yield data[i:i + batch_size], targets[i:i + batch_size]

def get_augmentation(device):
    return K.AugmentationSequential(
        K.RandomCrop((32, 32), padding=4),
        K.RandomHorizontalFlip(p=0.5),
        data_keys=["input"],
    ).to(device)

def load_data(device, data_root):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    raw_train = datasets.CIFAR100(root=str(data_root), train=True,
                                  download=True, transform=to_tensor)
    raw_test = datasets.CIFAR100(root=str(data_root), train=False,
                                 download=True, transform=to_tensor)
    return VRAMTensorData(raw_train, device), VRAMTensorData(raw_test, device)

def train_one_epoch(model, data, batch_size, criterion, optimizer, augment):
    model.train()
    total_loss = torch.zeros(1, device=data.data.device)
    correct = torch.zeros(1, dtype=torch.long, device=data.data.device)
    total = 0
    for inputs, targets in data.batches(batch_size, shuffle=True):
        inputs = augment(inputs)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        bs = inputs.size(0)
        total_loss += loss.detach() * bs
        correct += outputs.detach().argmax(1).eq(targets).sum()
        total += bs
    return total_loss.item() / total, 100.0 * correct.item() / total

@torch.no_grad()
def evaluate(model, data, batch_size, criterion):
    model.eval()
    total_loss = torch.zeros(1, device=data.data.device)
    correct = torch.zeros(1, dtype=torch.long, device=data.data.device)
    total = 0
    for inputs, targets in data.batches(batch_size, shuffle=False):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        bs = inputs.size(0)
        total_loss += loss * bs
        correct += outputs.argmax(1).eq(targets).sum()
        total += bs
    return total_loss.item() / total, 100.0 * correct.item() / total

def train_run(seed, ls, train_data, test_data, batch_size, augment, device,
              epochs, output_dir):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}\n  Seed {seed}  |  LS {ls}  |  {epochs} epochs (from scratch)\n{'='*60}")

    model = ResNet18_32x32(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                    weight_decay=WD, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    run_dir = output_dir / f"s{seed}" / f"ls_{ls}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                "test_loss", "test_acc", "lr"])

    best = 0.0
    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_data, batch_size,
                                 criterion, optimizer, augment)
        vl, va = evaluate(model, test_data, batch_size, criterion)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tl:.4f}", f"{ta:.2f}",
                                    f"{vl:.4f}", f"{va:.2f}", f"{lr:.6f}"])
        if va > best:
            best = va
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "test_acc": va}, run_dir / "best.ckpt")
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Ep {epoch:3d}/{epochs}  train {ta:.2f}%  test {va:.2f}%  best {best:.2f}%  lr {lr:.5f}")

    torch.save({"state_dict": model.state_dict(), "epoch": epochs,
                "label_smoothing": ls, "seed": seed,
                "test_acc": va, "best_test_acc": best},
               run_dir / "last.ckpt")
    return best

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--ls-values", nargs="+", type=float,
                   default=LABEL_SMOOTHING_VALUES)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--data-root", type=str, default=str(CIFAR100_DIR.parent))
    p.add_argument("--output-dir", type=str,
                   default=str(TRAINED_MODELS / "cifar100"))
    args = p.parse_args()

    ensure_dirs()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_data, test_data = load_data(device, args.data_root)
    augment = get_augmentation(device)
    output_dir = Path(args.output_dir)

    summary = {}
    for seed in args.seeds:
        for ls in args.ls_values:
            summary[(seed, ls)] = train_run(
                seed, ls, train_data, test_data, args.batch_size,
                augment, device, args.epochs, output_dir)

    print(f"\n{'='*60}\n  SUMMARY (best test acc %)\n{'='*60}")
    header = f"{'seed':<6}" + "".join(f"LS={ls:<10}" for ls in args.ls_values)
    print(header)
    for seed in args.seeds:
        print(f"{'s'+str(seed):<6}" +
              "".join(f"{summary[(seed, ls)]:<13.2f}" for ls in args.ls_values))

if __name__ == "__main__":
    main()
