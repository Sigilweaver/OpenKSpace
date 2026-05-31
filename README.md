# OpenKSpace

[![CI](https://github.com/Sigilweaver/OpenKSpace/actions/workflows/ci.yml/badge.svg)](https://github.com/Sigilweaver/OpenKSpace/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20470593.svg)](https://doi.org/10.5281/zenodo.20470593)
[![crates.io](https://img.shields.io/crates/v/openkspace-cli.svg)](https://crates.io/crates/openkspace-cli)
[![docs.rs](https://img.shields.io/docsrs/openkspace-io)](https://docs.rs/openkspace-io)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust MSRV](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](https://www.rust-lang.org)
[![Docs](https://img.shields.io/badge/docs-sigilweaver.app-blue.svg)](https://sigilweaver.app/openkspace/docs/)

A Rust library and CLI for Cartesian MRI k-space reconstruction from
[ISMRMRD](https://ismrmrd.github.io/) `.h5` files.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Features

### I/O
- ISMRMRD HDF5 reader with full `AcquisitionHeader` parsing
- Automatic extraction of noise and calibration scans
- Auto-detection of 2D vs 3D encoding from the data

### Calibration passes
- Noise pre-whitening via per-coil noise covariance + Cholesky
  (Kellman & McVeigh 2005)
- Navigator-echo phase correction (removes N/2 ghosting)
- Readout oversampling removal (image-domain crop after IFFT)
- Partial-Fourier along ky via homodyne reconstruction
  (Noll 1991; McGibney 1993)

### Reconstruction strategies (pluggable via the `ReconStrategy` trait)
| Strategy | Description |
|---|---|
| `ifft-rss` | Centred IFFT + root-sum-of-squares coil combination |
| `grappa` | k-space GRAPPA with ACS-fit convolution kernel (Griswold 2002) |
| `sense` | Image-domain SENSE unfold, ridge-stabilised (Pruessmann 1999), with optional **g-factor** map output |
| `cs` | L1-wavelet compressed sensing via FISTA (Lustig 2007; Beck & Teboulle 2009) |

SENSE coil-sensitivity maps can be estimated with either
- **Walsh** (eigenvector method, Walsh 2000), or
- **ESPIRiT** (auto-calibrating, Uecker 2014).

All strategies work on both 2D multi-slice and 3D Cartesian data. For 3D
acquisitions with ky-only undersampling, parallel-imaging
reconstructions decouple along kz via a 1-D IFFT, reducing the problem
to independent per-slice unfolds.

### Output
- PNG image output with percentile contrast windowing (default)
- NIfTI-1 single-file volume (`.nii`) for downstream analysis tools; select with `--format nifti` or `--format both`
- Optional g-factor PNG output for SENSE (linear window `[1, p99]`)

## Usage

```
openkspace info  <file.h5>                   # print header metadata
openkspace probe <file.h5>                   # scan acquisition index ranges
openkspace recon <file.h5>                   # reconstruct all slices -> recon_out/
    [--out <dir>]                            # output directory (default: recon_out/)
    [--slice <N>]                            # reconstruct one slice only
    [--fft auto|2d|3d]                       # FFT mode (default: auto)
    [--strategy ifft-rss|grappa|sense|cs]    # reconstruction strategy
    [--no-prewhiten]                         # skip noise pre-whitening
    [--no-phasecorr]                         # skip navigator phase correction
    [--no-oversampling-removal]              # keep 2x kx samples
    [--no-partial-fourier]                   # skip homodyne
    [--no-crop]                              # keep full oversampled FOV
    [--pct-low <f>] [--pct-high <f>]         # contrast window percentiles
    [--format png|nifti|both]                # output format (default: png)

  GRAPPA:
    [--grappa-kernel-ky <k>] [--grappa-kernel-kx <k>]
    [--grappa-ridge <lam>]

  SENSE:
    [--sense-maps walsh|espirit]
    [--sense-walsh-window <w>] [--sense-walsh-iters <n>]
    [--espirit-kernel <k>] [--espirit-threshold <f>] [--espirit-iters <n>]
    [--sense-ridge <lam>]
    [--sense-gfactor]                        # compute g-factor
    [--write-gfactor]                        # also emit g-factor PNGs

  CS:
    [--cs-iters <n>] [--cs-lambda <f>]
```

### Examples

```sh
# Plain IFFT + RSS
openkspace recon data.h5 --slice 15

# GRAPPA
openkspace recon data.h5 --strategy grappa

# SENSE with ESPIRiT maps and g-factor output
openkspace recon data.h5 --strategy sense \
    --sense-maps espirit --write-gfactor

# Compressed sensing
openkspace recon data.h5 --strategy cs --cs-iters 120 --cs-lambda 0.01
```

## Workspace layout

| Crate | Description |
|---|---|
| `openkspace-io` | ISMRMRD reader, acquisition structs, calibration scan extraction |
| `openkspace-recon` | Reconstruction math: FFT, coil combination, prewhitening, phase correction, GRAPPA, SENSE, ESPIRiT, Walsh, CS |
| `openkspace-cli` | `openkspace` command-line binary |

## Building

Requires Rust 1.75+ and HDF5 system libraries.

```sh
cargo build --release
cargo test
```

The `openkspace` binary is produced at `target/release/openkspace`.

## Validation

A Python harness in [scripts/](scripts/) compares the Rust reconstruction
against a reference on a per-slice basis using SSIM. It supports both
ISMRMRD and FastMRI files (format is auto-detected). See
[scripts/README.md](scripts/README.md) for details.

```sh
./scripts/validate.sh path/to/file.h5 --slice 15
```

On the NYU/Meta FastMRI multicoil knee corpus (652 files, slices 0-35
each), the IFFT+RSS+center-crop pipeline achieves **SSIM = 1.0000**
against the `reconstruction_rss` ground truth bundled in each file,
confirming bit-exact agreement between the Rust implementation and the
reference reconstruction.

## Citation

If you use OpenKSpace in research, please cite the underlying algorithm
paper for whichever reconstruction method you invoked (see
[CREDITS.md](CREDITS.md)) and, optionally, this repository.

## License

Apache-2.0. See [LICENSE](LICENSE).
