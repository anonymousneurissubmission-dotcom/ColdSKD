# Cold Self-Distillation Restores OOD Detection in Label-Smoothed Models

This repository is the official implementation of *Cold Self-Distillation
Restores OOD Detection in Label-Smoothed Models* (NeurIPS 2026 submission).

We retrain only the classification head of a label-smoothed pretrained
model with a cold knowledge-distillation objective (T < 1) to restore
the OOD-detection signal that label smoothing erodes, while preserving
ID accuracy.

## Requirements

```setup
pip install -r requirements.txt
```

(`pip install -e .` also works — uses the bundled `setup.py`.)

All paths resolve through `config.py`. Override the defaults by exporting
environment variables before any script:

```bash
export COLDSKD_DATA_ROOT=/path/to/datasets        # CIFAR-100, ImageNet, OOD sets
export COLDSKD_OUT_ROOT=/path/to/coldskd_out      # checkpoints, features, results
export COLDSKD_POOLED_DIR=/path/to/pooled_features  # optional, points elsewhere
```

## Training

The pipeline has two stages: (1) train backbones, (2) retrain heads on the
frozen backbone's pooled features.

**Backbones** (CIFAR-100 ResNet-18 and ImageNet-200 ResNet-50, both with
configurable label-smoothing strength α):

```train
python trainers/train_cifar100.py    --gpu 0 --seeds 0 1 2 --ls-values 0.0 0.1 0.2 0.3
python trainers/train_imagenet200.py --gpu 0           --ls-values 0.0 0.1 0.2 0.3
```

ImageNet-1K backbones are taken directly from torchvision/timm (no training).

**Pooled-feature extraction** (one-time, freezes a backbone's penultimate
embeddings + classification head):

```bash
python extraction/extract_cifar100.py    --gpu 0
python extraction/extract_imagenet200.py --gpu 0
python extraction/extract_imagenet1k.py  --gpu 0
```

**Head retraining (CSKD)** — single (λ, T):

```train
python heads/train_head_lambda_T.py --gpu 0 \
    --dataset imagenet200 --backbone resnet50_ls_0.1 \
    --lam 0.7 --T 0.5
```

**Head retraining (full λ × T sweep, parallel)**:

```train
python heads/lambda_T_sweep.py --gpu 0 \
    --dataset imagenet200 --backbone resnet50_ls_0.1
```

**Multi-seed AugDelete + CSKD T=0.5 (matches IM-1K t-test schema)**:

```train
python scripts/multi_seed_heads.py --gpu 0 --dataset cifar100
python scripts/multi_seed_heads.py --gpu 0 --dataset imagenet200
```

## Evaluation

```eval
python scoring/score_models.py --dataset imagenet200 --ash-percentile 90
```

Detectors written per `(backbone, head, ood_set)`: `mls`, `ash_s_mls`,
`ash_b_mls`, `ash_p_mls`, `cskd_dc` (= z-score(cosine) + z-score(energy),
the paper's CSKD-DC score). Filter scope with `--include resnet50` /
`--exclude vit`.

To re-render the four paper figures:

```eval
python figures/fig2_cifar100_alpha_sweep.py
python figures/fig3_kd_variance_3datasets.py
python figures/fig4_temperature_sweep.py
python figures/fig5_acc_vs_auroc_v2.py --methods Pretrained AugDelete "CSKD T=0.5" \
    --out-name fig1_pretrained_augdelete_cskd
```

## Pre-trained Models

Trained classification heads (the `(λ, T)` heads + multi-seed t-test
heads + IM-1K temperature-sweep heads, ~2.6 GB) are released as a
separate anonymous bundle:

- **Heads bundle (anonymous DOI):** https://doi.org/10.5281/zenodo.20042997

After download, extract into `trained_heads/` so the layout becomes:

```
trained_heads/
├── cifar100/             855 heads, 9 backbones        (Fig 2)
├── cifar100_ttest/       12 backbones × 3 seeds × 2 heads
├── imagenet200/          2974 heads, 28 backbones      (Tables)
├── imagenet200_ttest/    21 backbones × 3 seeds × 2 heads
├── imagenet1k_tsweep/    180 heads, 15 models          (Figs 4 & 5)
└── imagenet1k_ttest/     15 heads + 16 JSONs           (Fig 5)
```

The figure scripts auto-detect heads at this location. Override with
`COLDSKD_HEADS_CIFAR`, `COLDSKD_TSWEEP_DIR`, `COLDSKD_TTEST_DIR`.

ImageNet-1K backbones come straight from torchvision/timm — see the model
registry in `extraction/extract_imagenet1k.py` for the exact weight enums.

## Results

CSKD restores OOD detection on label-smoothed ImageNet-1K models while
preserving ID accuracy. **MLS AUROC** averaged over 6 OpenOOD OOD sets
(SSB-hard, NINCO, iNaturalist, OpenImage-O, DTD, Textures); **CSKD-DC**
is `z(cosine) + z(energy)` on the CSKD T=0.5 head.

### ImageNet-1K — Label-Smoothed (V2 / modern recipe)

| Model           | Acc (%) | Pretrained | AugDelete | **CSKD T=0.5** | **CSKD-DC** |
| --------------- | ------: | ---------: | --------: | -------------: | ----------: |
| RN50-V2         |   80.9  |     0.733  |    0.835  |       0.839    | **0.856**   |
| RN101-V2        |   81.9  |     0.758  |    0.829  |       0.841    | **0.866**   |
| RN152-V2        |   82.3  |     0.758  |    0.825  |       0.838    | **0.870**   |
| Swin-T-V1       |   81.5  |     0.809  |    0.846  |       0.864    | **0.875**   |
| Swin-T-V2       |   82.1  |     0.803  |    0.837  |       0.854    | **0.876**   |
| ViT-B16-V1      |   81.1  |     0.770  |    0.847  |       0.864    | **0.869**   |
| ViT-B16-SWAG    |   85.3  |     0.881  |    0.897  |     **0.911**  |   0.900     |
| RegNet-V2       |   82.9  |     0.756  |    0.823  |       0.831    | **0.864**   |
| EffNet-B1-V2    |   78.9  |     0.804  |    0.864  |       0.868    | **0.878**   |

CSKD-DC sets a new best on **11 of 14** LS-trained ImageNet-1K models.
On undegraded V1 (no LS) recipes CSKD is competitive with AugDelete and
neither degrades the pretrained baseline.

### Smaller-K results (CIFAR-100 K=100, ImageNet-200 K=200)

| Dataset      | Pretrained | AugDelete | **CSKD T=0.5** | **CSKD-DC** |
| ------------ | ---------: | --------: | -------------: | ----------: |
| CIFAR-100, α=0.1 |   0.762  |    0.772  |     **0.773**  |    0.773    |
| ImageNet-200, α=0.1 | 0.820  |    0.845  |       0.849   | **0.856**   |

CSKD's gain scales with K — see Section 6 of the paper for the full
discussion.

## Layout

```
ColdSKD/
├── config.py               Central paths (override via COLDSKD_* env vars)
├── trainers/               Full from-scratch trainers with LS
├── extraction/             Pooled-embedding extractors (CIFAR / IM200 / IM1K)
├── heads/                  Linear-head retraining: single (λ, T) + parallel sweep
├── scoring/                Filter + score (MLS, ASH, CSKD-DC)
├── figures/                Paper figure scripts (figs 2 - 5)
├── models/                 Backbone definitions (ResNet18-32x32 for CIFAR)
├── scripts/                Helper utilities (multi-seed t-test, MLS+cos eval)
├── trained_heads/          Released head checkpoints (download separately)
├── setup.py
└── requirements.txt
```

## Code

Anonymous code repository: https://doi.org/10.5281/zenodo.20042997

## Contributing

Released under the MIT License. See `LICENSE` for details.

## Citation

```bibtex
@inproceedings{coldskd2026,
  title  = {Cold Self-Distillation Restores OOD Detection in Label-Smoothed Models},
  author = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year   = {2026},
  note   = {Under double-blind review}
}
```
