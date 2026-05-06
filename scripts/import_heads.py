"""
Copy + strip trained head checkpoints from a local research tree into
trained_heads/.

The original checkpoints embed unrelated bookkeeping that inflates each
file by ~100x. We re-save only {weight, bias, lambda, T} so the released
set is practical (each linear head is ~D*K*4 bytes).

Source paths are taken from the COLDSKD_SRC_HEADS_*  environment
variables (each pointing at the original research tree's per-dataset
head directory). They are not bundled with the release because they
reference the author's local layout; reviewers do not need this script
since the stripped heads ship as a separate Zenodo deposit (see the top-
level README and trained_heads/README.md).

Usage (author-only):
    export COLDSKD_SRC_HEADS_CIFAR=/abs/path/to/cifar100/heads
    export COLDSKD_SRC_HEADS_IM200=/abs/path/to/imagenet200/heads
    export COLDSKD_SRC_HEADS_IM1K_TSWEEP=/abs/path/to/imagenet1k/tsweep
    export COLDSKD_SRC_HEADS_IM1K_TTEST=/abs/path/to/imagenet1k/ttest
    python scripts/import_heads.py --datasets cifar100
    python scripts/import_heads.py --datasets cifar100 imagenet200 imagenet1k
"""
import os
import sys
import argparse
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import OUT_ROOT  # noqa: F401  (only to confirm config imports)

DEST = ROOT / "trained_heads"

SOURCES = {
    "cifar100": Path(os.environ.get(
        "COLDSKD_SRC_HEADS_CIFAR", "/path/to/source/cifar100/heads")),
    "imagenet200": Path(os.environ.get(
        "COLDSKD_SRC_HEADS_IM200", "/path/to/source/imagenet200/heads")),
    "imagenet1k_tsweep": Path(os.environ.get(
        "COLDSKD_SRC_HEADS_IM1K_TSWEEP",
        "/path/to/source/imagenet1k/tsweep")),
    "imagenet1k_ttest": Path(os.environ.get(
        "COLDSKD_SRC_HEADS_IM1K_TTEST",
        "/path/to/source/imagenet1k/ttest")),
}

CIFAR_FIG2_BACKBONES = [
    f"resnet18_s{s}_ls_{a}_grid"
    for s in (0, 1, 2) for a in ("0.1", "0.2", "0.3")
]

def _im200_canonical():
    out = []
    for s in (0, 1, 2):
        out.append(f"baseline_s{s}")
        for ls in ("0.1", "0.2", "0.3"):
            out.append(f"ls_{ls}_s{s}")
        for mu in ("0.5", "1.0"):
            out.append(f"mixup_{mu}_s{s}")
        out.append(f"cutmix_1.0_s{s}")
        out.append(f"ls_0.1_cutmix_1.0_s{s}")
    out += [f"ls_{ls}_mixup_1.0_s0" for ls in ("0.1", "0.2", "0.3")]
    out.append("v2_recipe_s0")
    return out

IM200_CANONICAL = _im200_canonical()

IM1K_TTEST_KEEP_PTH = ("linear_kd_seed42.pth",)
IM1K_TTEST_KEEP_JSON_SUFFIX = "_ttest_results.json"

def strip_lamT_head(src: Path, dst: Path):
    """Save only the linear head weights + the (lambda, T) tag."""
    d = torch.load(src, map_location="cpu", weights_only=False)
    out = {"weight": d["weight"].detach().clone(),
           "bias": d["bias"].detach().clone()}
    for k in ("lam", "lambda", "T"):
        if k in d:
            out[k] = d[k]
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)

def import_cifar100():
    src_root = SOURCES["cifar100"]
    if not src_root.is_dir():
        print(f"[skip cifar100] {src_root} missing")
        return
    out_root = DEST / "cifar100"
    n, total_in, total_out = 0, 0, 0
    for bb in CIFAR_FIG2_BACKBONES:
        src_bb = src_root / bb
        if not src_bb.is_dir():
            print(f"  [skip] {bb} not found")
            continue
        dst_bb = out_root / bb
        for f in sorted(src_bb.glob("lam*_T*.pth")):
            dst = dst_bb / f.name
            if dst.is_file():
                continue
            strip_lamT_head(f, dst)
            n += 1
            total_in += f.stat().st_size
            total_out += dst.stat().st_size
        print(f"  [{bb}] {len(list(dst_bb.glob('*.pth')))} heads")
    print(f"cifar100: stripped {n} new files  "
          f"({total_in/1e6:.1f} MB -> {total_out/1e6:.1f} MB)")

def import_imagenet200(backbones=None):
    src_root = SOURCES["imagenet200"]
    if not src_root.is_dir():
        print(f"[skip imagenet200] {src_root} missing")
        return
    out_root = DEST / "imagenet200"
    if backbones is None:
        backbones = IM200_CANONICAL
    n, total_in, total_out = 0, 0, 0
    for bb_name in backbones:
        src_bb = src_root / bb_name
        if not src_bb.is_dir():
            print(f"  [skip] {bb_name} not found")
            continue
        for f in sorted(src_bb.glob("lam*_T*.pth")):
            dst = out_root / bb_name / f.name
            if dst.is_file():
                continue
            try:
                strip_lamT_head(f, dst)
                n += 1
                total_in += f.stat().st_size
                total_out += dst.stat().st_size
            except Exception as e:
                print(f"    [error {f.name}] {e}")
        kept = len(list((out_root / bb_name).glob('*.pth')))
        print(f"  [{bb_name}] {kept} heads")
    print(f"imagenet200: stripped {n} new files  "
          f"({total_in/1e6:.1f} MB -> {total_out/1e6:.1f} MB)")

def import_imagenet1k_tsweep(models=None):
    """Copy ImageNet-1K T-sweep heads (uses the {model}/T{p}{q}/final.pth tree)."""
    src_root = SOURCES["imagenet1k_tsweep"]
    if not src_root.is_dir():
        print(f"[skip imagenet1k_tsweep] {src_root} missing")
        return
    out_root = DEST / "imagenet1k_tsweep"
    n = 0
    for src_model in sorted(src_root.iterdir()):
        if not src_model.is_dir():
            continue
        if models and src_model.name not in models:
            continue
        for src_T in sorted(src_model.iterdir()):
            if not src_T.is_dir():
                continue
            final = src_T / "final.pth"
            if not final.is_file():
                continue
            dst = out_root / src_model.name / src_T.name / "final.pth"
            if dst.is_file():
                continue
            d = torch.load(final, map_location="cpu", weights_only=False)
            sd = {"weight": d["weight"].detach().clone(),
                  "bias": d["bias"].detach().clone()}
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sd, dst)
            n += 1
        print(f"  [{src_model.name}] {n} files written so far")
    print(f"imagenet1k_tsweep: stripped {n} new files")

def import_imagenet1k_ttest(models=None):
    """Copy ONLY the files figs 4/5 actually load:
       {model}/{model}_ttest_results.json   (small)
       {model}/linear_kd_seed42.pth         (~8 MB; head weights only)
    plus any top-level summary JSONs.
    """
    src_root = SOURCES["imagenet1k_ttest"]
    if not src_root.is_dir():
        print(f"[skip imagenet1k_ttest] {src_root} missing")
        return
    out_root = DEST / "imagenet1k_ttest"
    out_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for entry in sorted(src_root.iterdir()):
        if entry.is_file() and entry.suffix == ".json":
            dst = out_root / entry.name
            if not dst.is_file():
                shutil.copy2(entry, dst)
                n += 1
            continue
        if not entry.is_dir():
            continue
        if models and entry.name not in models:
            continue
        model_out = out_root / entry.name
        model_out.mkdir(parents=True, exist_ok=True)
        for j in entry.glob(f"*{IM1K_TTEST_KEEP_JSON_SUFFIX}"):
            dst = model_out / j.name
            if not dst.is_file():
                shutil.copy2(j, dst)
                n += 1
        for fname in IM1K_TTEST_KEEP_PTH:
            src = entry / fname
            dst = model_out / fname
            if src.is_file() and not dst.is_file():
                d = torch.load(src, map_location="cpu", weights_only=False)
                if "weight" in d and "bias" in d:
                    sd = {"weight": d["weight"].detach().clone(),
                          "bias": d["bias"].detach().clone()}
                else:
                    sd = d
                torch.save(sd, dst)
                n += 1
    print(f"imagenet1k_ttest: copied {n} files")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["cifar100"],
                   choices=["cifar100", "imagenet200",
                            "imagenet1k_tsweep", "imagenet1k_ttest"])
    p.add_argument("--imagenet200-backbones", nargs="*", default=None)
    p.add_argument("--imagenet1k-models", nargs="*", default=None)
    args = p.parse_args()

    print(f"Destination: {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)

    if "cifar100" in args.datasets:
        import_cifar100()
    if "imagenet200" in args.datasets:
        import_imagenet200(args.imagenet200_backbones)
    if "imagenet1k_tsweep" in args.datasets:
        import_imagenet1k_tsweep(args.imagenet1k_models)
    if "imagenet1k_ttest" in args.datasets:
        import_imagenet1k_ttest(args.imagenet1k_models)

if __name__ == "__main__":
    main()
