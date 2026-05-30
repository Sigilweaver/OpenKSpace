---
title: Introduction
sidebar_label: Intro
slug: /
---

# OpenKSpace

**Pure-Rust library and CLI for Cartesian MRI k-space reconstruction
from [ISMRMRD](https://ismrmrd.github.io/) `.h5` files.**

## What it does

OpenKSpace reads ISMRMRD acquisition files, runs a full calibration
pipeline, and produces image-domain reconstructions using any of
four pluggable strategies. It is designed to be a clean-room
reference implementation, easy to read, and fast enough for real
clinical-scale acquisitions.

## Capabilities

- Full `AcquisitionHeader` parsing with automatic extraction of
  noise and calibration scans.
- Automatic detection of 2D multi-slice vs 3D Cartesian encoding.
- Calibration passes: noise pre-whitening (Kellman & McVeigh
  2005), navigator-echo phase correction, readout oversampling
  removal, partial-Fourier homodyne reconstruction.
- Reconstruction strategies via the `ReconStrategy` trait:
  `ifft-rss`, `grappa`, `sense` (with optional g-factor), `cs`
  (FISTA L1-wavelet).
- SENSE coil-sensitivity maps via Walsh or ESPIRiT.
- Output: PNG (percentile windowing), NIfTI-1, or both.

## Not (yet) supported

- Non-Cartesian trajectories.
- Wave-CAIPI / multi-band SMS.
- Spiral / radial / EPI distortion correction.

## Status

OpenKSpace is **research software**. It is not validated for
diagnostic use.

## Get started

- [Install](./install.md)
- [CLI quickstart](./quickstart-cli.md)
- [Rust quickstart](./quickstart-rust.md)
- [Citations and references](./citations.md)
