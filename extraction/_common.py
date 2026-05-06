"""Shared embedding-extraction helpers."""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FlatImages(Dataset):
    """Recursively reads images from a directory; label = -1."""

    EXTS = (".png", ".jpg", ".jpeg", ".JPEG", ".bmp", ".tif", ".tiff", ".webp")

    def __init__(self, root, transform):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix in self.EXTS)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), -1

@torch.no_grad()
def extract_with_hook(model, loader, device, hook_module, store_key="emb"):
    """Run model over loader, capturing the output of `hook_module` (flattened)."""
    store = {}

    def fn(_m, _i, out):
        store[store_key] = out.flatten(1)

    handle = hook_module.register_forward_hook(fn)
    embs, lbls = [], []
    try:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            embs.append(store[store_key].cpu().half().numpy())
            lbls.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    finally:
        handle.remove()
    return np.concatenate(embs), np.concatenate(lbls).astype(np.int32)

def save_features(out_dir: Path, tag: str, emb, lbl):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{tag}_emb.npy", emb)
    np.save(out_dir / f"{tag}_lbl.npy", lbl)

def save_fc(out_dir: Path, fc_module):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": fc_module.weight.detach().cpu(),
                "bias": fc_module.bias.detach().cpu()},
               out_dir / "trained_fc.pth")

def already_done(out_dir: Path, tag: str) -> bool:
    return (out_dir / f"{tag}_emb.npy").is_file() and (out_dir / f"{tag}_lbl.npy").is_file()
