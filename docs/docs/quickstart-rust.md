---
title: Rust quickstart
sidebar_label: Rust quickstart
---

# Rust quickstart

OpenKSpace exposes its pipeline as a library, split into two crates:

- [`openkspace-io`](https://crates.io/crates/openkspace-io) - the
  ISMRMRD HDF5 reader and header types.
- [`openkspace-recon`](https://crates.io/crates/openkspace-recon) -
  calibration passes, reconstruction strategies, and the
  `ReconStrategy` trait.

## Minimal example

```rust
use openkspace_io::IsmrmrdFile;
use openkspace_recon::{
    strategies::IfftRssStrategy,
    ReconStrategy, ReconConfig,
};

fn main() -> anyhow::Result<()> {
    let file = IsmrmrdFile::open("scan.h5")?;

    let cfg = ReconConfig::default();
    let strategy = IfftRssStrategy::default();

    let images = strategy.recon(&file, &cfg)?;
    println!("Reconstructed {} slice(s)", images.len());
    Ok(())
}
```

`ReconConfig` is constructed via builder methods to toggle the
individual calibration passes (`prewhiten`, `phase_correction`,
`oversampling_removal`, `partial_fourier`, `crop`).

## Selecting another strategy

```rust
use openkspace_recon::strategies::{GrappaStrategy, SenseStrategy, CsStrategy};

let grappa = GrappaStrategy::new(/* kernel_ky */ 5, /* kernel_kx */ 4)
    .with_ridge(1e-4);

let sense = SenseStrategy::default()
    .with_maps(openkspace_recon::sense::MapSource::Espirit)
    .with_gfactor(true);

let cs = CsStrategy::default()
    .with_wavelet_levels(4)
    .with_lambda(1e-3);
```

## API docs

Full rustdoc on docs.rs:

- [openkspace-io](https://docs.rs/openkspace-io)
- [openkspace-recon](https://docs.rs/openkspace-recon)
