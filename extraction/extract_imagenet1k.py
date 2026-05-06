"""
Extract pooled embeddings + classification head for torchvision ImageNet-1K models.

Supports the V1 vs V2 weight pairs used in the paper (V2 was trained with LS).
Reads ImageNet val from IMAGENET_VAL and OOD sets from OOD_PATHS.

Saves to POOLED_FEATURES/imagenet1k/{model}/:
    imagenet1k_val_{emb,lbl}.npy
    trained_fc.pth
plus per-OOD subfolders.

For *training-set* embeddings (very large), pass --include-train.

Run:
    python extraction/extract_imagenet1k.py --gpu 0 \
        --models resnet50_v1 resnet50_v2 resnet101_v1 resnet101_v2 \
                 resnet152_v1 resnet152_v2 efficientnet_b0_v1 efficientnet_b1_v1 \
                 vit_b_16_v1 vit_b_16_v2 swin_t_v1 swin_t_v2
"""
import sys
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (IMAGENET_TRAIN, IMAGENET_VAL, OOD_PATHS,
                    POOLED_FEATURES, ensure_dirs)
sys.path.insert(0, str(ROOT / "extraction"))
from _common import FlatImages, extract_with_hook, save_features, save_fc, already_done

def _r(arch_fn, weights):
    return {"arch_fn": arch_fn, "weights": weights}

MODEL_REGISTRY = {
    "resnet50_v1":        _r(models.resnet50,         models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet50_v2":        _r(models.resnet50,         models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet101_v1":       _r(models.resnet101,        models.ResNet101_Weights.IMAGENET1K_V1),
    "resnet101_v2":       _r(models.resnet101,        models.ResNet101_Weights.IMAGENET1K_V2),
    "resnet152_v1":       _r(models.resnet152,        models.ResNet152_Weights.IMAGENET1K_V1),
    "resnet152_v2":       _r(models.resnet152,        models.ResNet152_Weights.IMAGENET1K_V2),
    "efficientnet_b0_v1": _r(models.efficientnet_b0,  models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b1_v1": _r(models.efficientnet_b1,  models.EfficientNet_B1_Weights.IMAGENET1K_V1),
    "vit_b_16_v1":        _r(models.vit_b_16,         models.ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_16_v2":        _r(models.vit_b_16,         models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    "swin_t_v1":          _r(models.swin_t,           models.Swin_T_Weights.IMAGENET1K_V1),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_target_module_and_fc(model, name: str):
    """Return (hook_module, fc_module). Hook the input to fc by hooking the
    last layer that produces the pooled embedding."""
    if name.startswith(("resnet", "efficientnet")):
        return (model.avgpool,
                model.fc if hasattr(model, "fc") else model.classifier[1])
    if name.startswith("vit_"):
        return (model.heads, model.heads.head)
    if name.startswith("swin_"):
        return (model.avgpool, model.head)
    raise ValueError(f"Unknown model family for {name}")

@torch.no_grad()
def _extract(model, loader, device, hook_module):
    """Variant of extract_with_hook that handles ViT/Swin shapes."""
    import numpy as np
    store = {}

    def fn(_m, inp, _o):
        if isinstance(inp, tuple) and len(inp) == 1 and inp[0].dim() == 2:
            store["emb"] = inp[0]
        else:
            pass

    def fn_out(_m, _i, out):
        store["emb"] = out.flatten(1)

    if isinstance(hook_module, torch.nn.AdaptiveAvgPool2d) or "AvgPool" in type(hook_module).__name__:
        handle = hook_module.register_forward_hook(fn_out)
    else:
        handle = hook_module.register_forward_pre_hook(fn)

    embs, lbls = [], []
    try:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            embs.append(store["emb"].cpu().half().numpy())
            lbls.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    finally:
        handle.remove()
    import numpy as np
    return np.concatenate(embs), np.concatenate(lbls).astype(np.int32)

def build_loaders(transform, batch_size, num_workers, include_train):
    val_ds = ImageFolder(str(IMAGENET_VAL), transform=transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True)
    print(f"  ImageNet-1K val: {len(val_ds)}")

    train_loader = None
    if include_train:
        train_ds = ImageFolder(str(IMAGENET_TRAIN), transform=transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=True)
        print(f"  ImageNet-1K train: {len(train_ds)}")

    ood_loaders = {}
    candidates = [
        ("inaturalist", lambda: FlatImages(OOD_PATHS["inaturalist"], transform)),
        ("openimage_o", lambda: FlatImages(OOD_PATHS["openimage_o"], transform)),
        ("textures",    lambda: FlatImages(OOD_PATHS["textures"], transform)),
        ("ssb_hard",    lambda: FlatImages(OOD_PATHS["ssb_hard"], transform)),
        ("ninco",       lambda: FlatImages(OOD_PATHS["ninco"], transform)),
        ("imagenet_o",  lambda: FlatImages(OOD_PATHS["imagenet_o"], transform)),
    ]
    for name, fn in candidates:
        try:
            ds = fn()
            ood_loaders[name] = DataLoader(ds, batch_size=batch_size,
                                           num_workers=num_workers, pin_memory=True)
            print(f"  OOD {name}: {len(ds)}")
        except Exception as e:
            print(f"  OOD {name}: SKIPPED ({e})")
    return train_loader, val_loader, ood_loaders

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    p.add_argument("--out-dir", type=str,
                   default=str(POOLED_FEATURES / "imagenet1k"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--include-train", action="store_true",
                   help="Also extract ImageNet-1K train embeddings (slow, large).")
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()

    if args.list_models:
        for k in MODEL_REGISTRY:
            print(k)
        return

    ensure_dirs()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out_root = Path(args.out_dir)

    for name in args.models:
        if name not in MODEL_REGISTRY:
            print(f"[skip] unknown model {name}")
            continue
        cfg = MODEL_REGISTRY[name]
        print(f"\n[{name}]")

        weights = cfg["weights"]
        model = cfg["arch_fn"](weights=weights).to(device).eval()
        transform = weights.transforms()

        train_loader, val_loader, ood_loaders = build_loaders(
            transform, args.batch_size, args.num_workers, args.include_train)
        run_dir = out_root / name

        if not (run_dir / "trained_fc.pth").is_file():
            fc = (model.fc if hasattr(model, "fc")
                  else model.classifier[-1] if hasattr(model, "classifier")
                  else model.heads.head if hasattr(model, "heads")
                  else model.head)
            save_fc(run_dir, fc)

        hook_mod, _ = get_target_module_and_fc(model, name)

        if args.include_train and train_loader is not None and not already_done(run_dir, "imagenet1k_train"):
            print("  imagenet1k_train: extracting...", flush=True)
            emb, lbl = _extract(model, train_loader, device, hook_mod)
            save_features(run_dir, "imagenet1k_train", emb, lbl)
            print(f"    -> {emb.shape}")

        if not already_done(run_dir, "imagenet1k_val"):
            print("  imagenet1k_val: extracting...", flush=True)
            emb, lbl = _extract(model, val_loader, device, hook_mod)
            save_features(run_dir, "imagenet1k_val", emb, lbl)
            print(f"    -> {emb.shape}")
        else:
            print("  imagenet1k_val: cached")

        for ood_name, ood_loader in ood_loaders.items():
            ood_dir = out_root / "ood" / ood_name / name
            if already_done(ood_dir, ood_name):
                continue
            print(f"  OOD {ood_name}...", flush=True)
            emb, lbl = _extract(model, ood_loader, device, hook_mod)
            save_features(ood_dir, ood_name, emb, lbl)
            print(f"    -> {emb.shape}")

        del model
        torch.cuda.empty_cache()

    print("\nDone.")

if __name__ == "__main__":
    main()
