# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-11-26

First crates.io release since 0.2.0. Version 0.3.0 was tagged and has a
GitHub release, but its crates.io publish failed (the release commit
bumped the workspace version while the internal crate pins still said
0.2.0), so 0.3.0 was never published. Everything listed under 0.3.0
below ships here too, together with the changes since.

### Breaking

- **`ndarray` 0.15 to 0.17.** `ndarray` types appear in the public API of
  `openkspace-io` (`read_kspace`, `read_reconstruction_rss`,
  `Acquisition::as_array_view`, ...) and `openkspace-recon` (FFT, shift,
  coil-combine, GRAPPA, wavelet and partial-Fourier functions, and
  `ImageVolume`). Downstream code must use `ndarray` 0.17.
- **`hdf5-metno` 0.9 to 0.15** (via 0.12 in 0.3.0). `IoError::Hdf5` wraps
  `hdf5_metno::Error`, so code that matches on or names that type needs
  `hdf5-metno` 0.15.
- **Minimum supported Rust version raised from 1.87 to 1.88**, which the
  locked `image` 0.25.10 (used by `openkspace-cli`) already requires.

### Changed

- `quick-xml` bumped from `0.36` to `0.42`, clearing RUSTSEC-2026-0194
  and RUSTSEC-2026-0195. The ISMRMRD header parser (`openkspace-io`)
  moves from `BytesText::unescape()` to `BytesText::xml10_content()`
  followed by `quick_xml::escape::unescape()`. The stale `--ignore`
  flags for those two advisories are removed from the audit workflow.
- `thiserror` 1 to 2, `indicatif` 0.17 to 0.18, plus routine patch
  updates of `clap`, `serde`, `serde_json` and `anyhow`. None of these
  change the public API.
- `Acquisition::from_raw_f32` no longer `debug_assert`s that the input
  length matches the header; see Security below.
- Em dashes, en dashes and box-drawing characters in source comments
  and `CREDITS.md` replaced with ASCII.

### Added

- `LICENSE` (Apache-2.0) is included in each published crate package
  as a copy in each crate directory; the 0.2.0 packages did not carry it.
- `CITATION.cff` Zenodo concept DOI, and a CI check that its version
  and release date match the crate release.
- `cargo-audit` workflow, Dependabot configuration, and a CI
  concurrency group.
- Regression test for the `Acquisition::from_raw_f32` allocation bound.

### Security

- `Acquisition::from_raw_f32` no longer sizes its allocation from the
  file-controlled `number_of_samples` / `active_channels` header fields;
  capacity now tracks the actual decoded data, so a crafted ISMRMRD
  header can no longer force an oversized allocation independent of
  what was actually read. `IsmrmrdFile::for_each`'s header/data length
  cross-check now uses checked multiplication so a crafted header
  cannot wrap the expected-length computation. Fixes #1. (@Nabejo)

### Fixed

- Internal crate version pins track the workspace version, so
  `cargo publish` resolves the matching `openkspace-io` /
  `openkspace-recon` release (the cause of the failed 0.3.0 publish).
- CI: macOS builds provision HDF5 1.14.6 and export its install path so
  `hdf5-metno-sys` can find it.

## [0.3.0] - 2026-05-31

Tagged and released on GitHub only; never published to crates.io. These
changes first reach crates.io in 0.4.0.

### Added

- `CITATION.cff`: author identity (Nathan Riley + ORCID) and a
  scaffolded `identifiers:` block ready for the Zenodo concept DOI.

### Changed

- **Panic surface eliminated (WP17).** Production code in
  `openkspace-cli` no longer calls `unwrap()`; the static
  `ProgressStyle::template()` is now `expect("static progress
  template")`, and the BTreeSet next-value lookup uses
  `copied().unwrap_or(0)`.
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
