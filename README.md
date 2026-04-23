# OpenKSpace

A Rust library and CLI for Cartesian MRI k-space reconstruction from [ISMRMRD](https://ismrmrd.github.io/) `.h5` files.

## Features

- ISMRMRD HDF5 reader with full `AcquisitionHeader` parsing
- Noise pre-whitening (Cholesky covariance, Kellman & McVeigh 2005)
- Navigator echo phase correction
- Centred IFFT + root-sum-of-squares (RSS) coil combination
- 2D multi-slice and 3D Cartesian reconstruction paths
- Auto-detection of 2D vs 3D encoding from the data
- PNG image output with percentile contrast windowing

## Usage

```
openkspace info  <file.h5>                   # print header metadata
openkspace probe <file.h5>                   # scan acquisition index ranges
openkspace recon <file.h5>                   # reconstruct all slices -> recon_out/
    [--out <dir>]                            # output directory (default: recon_out/)
    [--slice <N>]                            # reconstruct one slice only
    [--fft auto|2d|3d]                       # FFT mode (default: auto)
    [--no-prewhiten]                         # skip noise pre-whitening
    [--no-phasecorr]                         # skip navigator phase correction
    [--no-crop]                              # keep full oversampled FOV
    [--pct-low <f>] [--pct-high <f>]        # contrast window percentiles
```

## Workspace layout

| Crate | Description |
|---|---|
| `openkspace-io` | ISMRMRD reader, acquisition structs, calibration scan extraction |
| `openkspace-recon` | Reconstruction math: FFT, coil combination, prewhitening, phase correction |
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
against a numpy reference on a per-slice basis using SSIM. See
[scripts/README.md](scripts/README.md) for details.

```sh
./scripts/validate.sh path/to/file.h5 --slice 15
```

## Citation

If you use Sigil in research, please cite the underlying algorithm paper for
whichever reconstruction method you invoked (see above) and, optionally,
this repository. The reference list is in [CREDITS.md](CREDITS.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
