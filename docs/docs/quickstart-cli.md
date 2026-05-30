---
title: CLI quickstart
sidebar_label: CLI quickstart
---

# CLI quickstart

## Inspect a file

```sh
openkspace info  scan.h5    # print header metadata
openkspace probe scan.h5    # scan acquisition index ranges
```

## Reconstruct (defaults)

```sh
openkspace recon scan.h5
```

By default this writes PNGs of every slice into `recon_out/` using
the IFFT + root-sum-of-squares strategy and all calibration passes
auto-detected from the data.

## Choose a reconstruction strategy

```sh
openkspace recon scan.h5 --strategy ifft-rss
openkspace recon scan.h5 --strategy grappa
openkspace recon scan.h5 --strategy sense  --sense-maps espirit
openkspace recon scan.h5 --strategy cs
```

## Common options

```sh
openkspace recon scan.h5 \
    --out my_out/ \
    --slice 12 \
    --strategy sense --sense-maps walsh \
    --sense-gfactor --write-gfactor \
    --format both
```

| Flag                            | Effect                              |
| ------------------------------- | ----------------------------------- |
| `--out <dir>`                   | Output directory (default `recon_out/`) |
| `--slice <N>`                   | Reconstruct one slice only          |
| `--fft auto\|2d\|3d`            | FFT mode (default `auto`)           |
| `--format png\|nifti\|both`     | Output format (default `png`)       |
| `--no-prewhiten`                | Skip noise pre-whitening            |
| `--no-phasecorr`                | Skip navigator phase correction     |
| `--no-oversampling-removal`     | Keep 2x kx samples                  |
| `--no-partial-fourier`          | Skip homodyne                       |
| `--no-crop`                     | Keep full oversampled FOV           |
| `--pct-low <f> --pct-high <f>`  | Contrast window percentiles         |

See the [CLI reference](./cli.md) for the full strategy-specific
flag set (GRAPPA kernel sizes, SENSE map sources, CS regulariser,
etc.).
