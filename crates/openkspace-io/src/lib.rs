//! OpenKSpace-IO: Readers for raw MRI formats.
//!
//! Supported today:
//!   * ISMRMRD (.h5)  -- vendor-agnostic HDF5 container used by mridata.org
//!
//! Planned:
//!   * Siemens TWIX  (.dat)  -- MDH / Sync-link parser
//!   * GE P-file     (.7)    -- Pfile header + data
//!   * Philips raw   (.raw)  -- paired with .lab / .sin

pub mod error;
pub mod ismrmrd;

pub use error::{IoError, IoResult};
