//! FastMRI HDF5 reader module.
//!
//! Provides [`FastmriFile`] for opening and reading fastMRI `.h5` files
//! produced by NYU Langone Health / Meta AI Research.

pub mod reader;

pub use reader::{FastmriFile, FastmriMeta};
