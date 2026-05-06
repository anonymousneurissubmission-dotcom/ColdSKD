"""
Central paths for ColdSKD.

All locations resolve from environment variables with sensible defaults.
Override any path by exporting the matching env var, e.g.:

    export COLDSKD_DATA_ROOT=/scratch/data
    export COLDSKD_OUT_ROOT=/scratch/coldskd_out
"""
import os
from pathlib import Path

def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))

DATA_ROOT = _env_path("COLDSKD_DATA_ROOT",
                       str(Path.home() / "data"))
OUT_ROOT = _env_path(
    "COLDSKD_OUT_ROOT",
    str(Path(__file__).resolve().parent / "outputs"),
)

CIFAR100_DIR = DATA_ROOT / "cifar-100-python"
TINY_IMAGENET_DIR = DATA_ROOT / "Tiny_imagenet"
IMAGENET_ROOT = DATA_ROOT / "imagenet1k" / "Imagenet"
IMAGENET_TRAIN = IMAGENET_ROOT / "train"
IMAGENET_VAL = IMAGENET_ROOT / "val" / "ILSVRC2012_img_val"
IMAGENET200_IMGLIST = DATA_ROOT / "benchmark_imglist" / "imagenet200"

OOD_PATHS = {
    "cifar10":     DATA_ROOT / "cifar-10-batches-py",
    "svhn":        DATA_ROOT / "SVHN",
    "textures":    DATA_ROOT / "dtd" / "dtd" / "images",
    "lsun_c":      DATA_ROOT / "LSUN" / "test",
    "lsun_r":      DATA_ROOT / "LSUN_resize" / "LSUN_resize",
    "isun":        DATA_ROOT / "iSUN" / "iSUN_patches",
    "places":      DATA_ROOT / "Places" / "images",
    "inaturalist": DATA_ROOT / "iNaturalist" / "images",
    "openimage_o": DATA_ROOT / "openimage_o" / "images",
    "ssb_hard":    DATA_ROOT / "ssb_hard",
    "ninco":       DATA_ROOT / "NINCO" / "NINCO" / "NINCO_OOD_classes",
    "imagenet_o":  DATA_ROOT / "imagenet_o",
}

TRAINED_MODELS = OUT_ROOT / "trained_models"
HEADS_DIR = OUT_ROOT / "heads"
RESULTS_DIR = OUT_ROOT / "results"
FIGURES_DIR = OUT_ROOT / "figures"

POOLED_FEATURES = _env_path("COLDSKD_POOLED_DIR",
                             str(OUT_ROOT / "pooled_features"))

def ensure_dirs():
    for d in (TRAINED_MODELS, HEADS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
