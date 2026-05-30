---
title: Reconstruction
sidebar_label: Reconstruction
---

# Reconstruction strategies

All strategies implement the same `ReconStrategy` trait, take a
calibrated k-space tensor `[ch, kz, ky, kx]`, and return image-
domain volumes. 3D acquisitions with ky-only undersampling
decouple along kz via a 1-D IFFT and then run independent
per-slice unfolds, so parallel imaging is supported for both 2D
multi-slice and 3D Cartesian data.

## `ifft-rss`

Centred IFFT plus root-sum-of-squares coil combination. Fastest
strategy; produces magnitude images only. Recommended baseline.

## `grappa`

k-space GRAPPA (Griswold 2002). ACS lines are extracted
automatically; a convolution kernel is fit via ridge regression and
applied to fill the missing ky lines.

| Flag                        | Default | Description                       |
| --------------------------- | ------- | --------------------------------- |
| `--grappa-kernel-ky <k>`    | 4       | Kernel size along ky              |
| `--grappa-kernel-kx <k>`    | 5       | Kernel size along kx              |
| `--grappa-ridge <lam>`      | 1e-3    | Tikhonov ridge for kernel fit     |

## `sense`

Image-domain SENSE unfold (Pruessmann 1999) with ridge
stabilisation. Sensitivity maps are estimated via either Walsh
(eigenvector method, Walsh 2000) or ESPIRiT
(auto-calibrating, Uecker 2014).

| Flag                              | Default   | Description                               |
| --------------------------------- | --------- | ----------------------------------------- |
| `--sense-maps walsh\|espirit`     | `walsh`   | Coil sensitivity estimation               |
| `--sense-walsh-window <w>`        | 7         | Walsh smoothing window                    |
| `--sense-walsh-iters <n>`         | 3         | Walsh power-method iterations             |
| `--espirit-kernel <k>`            | 6         | ESPIRiT kernel size                       |
| `--espirit-threshold <f>`         | 0.02      | ESPIRiT singular value threshold          |
| `--espirit-iters <n>`             | 50        | ESPIRiT power iterations                  |
| `--sense-ridge <lam>`             | 1e-4      | Tikhonov ridge for unfold                 |
| `--sense-gfactor`                 | off       | Compute g-factor map                      |
| `--write-gfactor`                 | off       | Also emit g-factor PNGs                   |

## `cs`

L1-wavelet compressed sensing via FISTA (Lustig 2007; Beck &
Teboulle 2009). Slower than GRAPPA/SENSE but better at high
acceleration with random ky undersampling.

| Flag                       | Default | Description                          |
| -------------------------- | ------- | ------------------------------------ |
| `--cs-iters <n>`           | 80      | FISTA iterations                     |
| `--cs-lambda <lam>`        | 1e-3    | L1 regularisation strength           |
| `--cs-wavelet-levels <n>`  | 4       | Wavelet decomposition levels         |
