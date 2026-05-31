# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- _No unreleased changes yet._

## [0.3.0] - 2026-05-31

### Added

- `CITATION.cff`: author identity (Nathan Riley + ORCID) and a
  scaffolded `identifiers:` block ready for the Zenodo concept DOI.

### Changed

- **Panic surface eliminated (WP17).** Production code in
  `openkspace-cli` no longer calls `unwrap()`; the static
  `ProgressStyle::template()` is now `expect("static progress
  template")`, and the BTreeSet next-value lookup uses
  `copied().unwrap_or(0)`. Library crates carry
  `#![cfg_attr(not(test), warn(clippy::unwrap_used,
  clippy::expect_used))]` so future regressions are linted.
- HDF5 dependency bumped from `hdf5-metno 0.9` to `0.12`
  (`hdf5-metno-sys 0.11.3` supports HDF5 2.x).
- CI: `HDF5_DIR` exported on macOS so `hdf5-metno-sys` can find
  the brew install.

## [0.2.0] - 2026-05-22

Publication-ready glow up. Brings OpenKSpace into line with the rest
of the Sigilweaver suite conventions.

### Changed

- Workspace MSRV raised from `1.75` to `1.87` to match the rest of
  the suite.
- Crate metadata moved fully under `[workspace.package]`:
  `authors`, `repository`, `homepage`, `documentation`, `readme`,
  `keywords`, `categories` are now declared once and inherited.
- Forbid `unsafe_code` workspace-wide via `[workspace.lints.rust]`.

### Added

- README badges: CI, docs.
- `CONTRIBUTING.md`.
- GitHub Actions CI workflow (`cargo fmt` / `clippy` / `test` on Linux
  and macOS, with HDF5 system deps).
- GitHub Actions release workflow with crates.io trusted publishing
  for `openkspace-io`, `openkspace-recon`, `openkspace-cli`.
- `homepage = "https://sigilweaver.app/openkspace/"` and
  `documentation = "https://sigilweaver.app/openkspace/docs/"` for
  crates.io / docs.rs discovery.

## [0.1.0] - 2025-05-04

Initial public release.

### Added

**openkspace-io**
- ISMRMRD HDF5 reader: `IsmrmrdFile`, `IsmrmrdHeader`, `AcquisitionHeader`, `EncodingCounters`.
- FastMRI HDF5 reader: `FastmriFile`, `FastmriMeta`, `is_brain()` heuristic.
- All public enums and major structs are `#[non_exhaustive]` for forward-compatible API.

**openkspace-recon**
- `IfftRss`: 2-D and 3-D IFFT + RSS coil combination.
- `GrappaRss`: auto-calibrated GRAPPA kernel synthesis and gap filling.
- `SenseRss`: image-domain SENSE unfold (Pruessmann 1999) with ridge stabilisation;
  optional g-factor map output.
- `CsRss`: FISTA compressed sensing with wavelet sparsity prior.
- Supporting primitives: pre-whitening, navigator phase correction, homodyne
  partial-Fourier, readout oversampling removal, centre crop.
- `ReconStrategy` trait for pluggable back-ends.
- New error types: `SenseError`, `CsError`, `GrappaError` (all `#[non_exhaustive]`).
- Library functions return `Result` types instead of panicking on invalid input.

**openkspace-cli**
- `openkspace recon` command with `--strategy`, `--fft`, `--format` (PNG / NIfTI / both),
  `--slice`, `--out`, `--write-gfactor`, SENSE, GRAPPA, and CS tuning flags.
- `openkspace info` command (plain and `--json`).
- `openkspace probe` command for scanning acquisition index ranges.
- PNG output with configurable percentile contrast windowing.
- NIfTI-1 single-file volume output for downstream analysis.
- Progress bars via `indicatif`.

**Validation**
- Corpus of ISMRMRD and FastMRI samples under `corpus/`.
- `scripts/validate.py` for SSIM regression testing (FastMRI: SSIM = 1.0000).

### Changed

- `GrappaKernel::synthesize` no longer accepts a `pattern` argument; the kernel
  already encodes its own acceleration factor.
- `GrappaKernel::weights` demoted from `pub` to `pub(crate)` (opaque kernel internals).
- `AcquisitionHeader` and `EncodingCounters` implement `Default` (safe zeroing;
  replaces `unsafe { mem::zeroed() }` at all call sites).

### Fixed

- NIfTI dimension fields now use `i16::try_from` with error propagation instead of a
  silently wrapping `as i16` cast (was unsound for matrices larger than 32767).
- Unused `rayon` and `memmap2` workspace dependencies removed.

### Security

- No `unsafe` code introduced; existing `mem::zeroed` usage on `AcquisitionHeader`
  replaced with safe `Default::default()`.
