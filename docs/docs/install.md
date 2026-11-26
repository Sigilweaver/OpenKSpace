---
title: Install
sidebar_label: Install
---

# Install

## Command-line tool

```sh
cargo install openkspace-cli
openkspace --help
```

This installs an `openkspace` binary on your `PATH`.

## Rust library

Add the appropriate crate to your `Cargo.toml`:

```toml
[dependencies]
openkspace-io    = "0.4"   # ISMRMRD HDF5 reader
openkspace-recon = "0.4"   # calibration + recon strategies
```

MSRV: Rust 1.87.

## System dependencies

OpenKSpace reads HDF5 files via the `hdf5-metno` crate. On Linux:

```sh
sudo apt-get install libhdf5-dev
```

On macOS:

```sh
brew install hdf5
```

On Windows, the `hdf5-metno` build script will download a
pre-built HDF5 binary automatically.

## From source

```sh
git clone https://github.com/Sigilweaver/OpenKSpace
cd OpenKSpace
cargo build --release
./target/release/openkspace --help
```
