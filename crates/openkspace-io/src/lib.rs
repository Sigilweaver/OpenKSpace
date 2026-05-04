//! OpenKSpace-IO: Readers for raw MRI formats.
//!
//! Supported today:
//!   * ISMRMRD (.h5)  -- vendor-agnostic HDF5 container used by mridata.org
//!   * FastMRI (.h5)  -- NYU/Facebook multicoil knee/brain dataset format
//!
//! Planned:
//!   * Siemens TWIX  (.dat)  -- MDH / Sync-link parser
//!   * GE P-file     (.7)    -- Pfile header + data
//!   * Philips raw   (.raw)  -- paired with .lab / .sin
//!
//! # Quick start
//!
//! ```rust,no_run
//! use openkspace_io::{is_fastmri, FastmriFile};
//! use openkspace_io::ismrmrd::IsmrmrdFile;
//!
//! let path = std::path::Path::new("scan.h5");
//! if is_fastmri(path) {
//!     let f = FastmriFile::open(path).unwrap();
//!     let kspace = f.read_kspace().unwrap(); // [coils, slices, ky, kx]
//! } else {
//!     let f = IsmrmrdFile::open(path).unwrap();
//!     // kspace built slice-by-slice via f.read_acquisitions()
//! }
//! ```

pub mod error;
pub mod fastmri;
pub mod ismrmrd;

pub use error::{IoError, IoResult};
pub use fastmri::{FastmriFile, FastmriMeta};

/// Lightweight format probe: returns `true` if the file has a `/kspace`
/// dataset (FastMRI layout), `false` otherwise.
///
/// Opens the HDF5 file without parsing any headers or logging.
pub fn is_fastmri<P: AsRef<std::path::Path>>(path: P) -> bool {
    hdf5_metno::File::open(path)
        .map(|f| f.dataset("kspace").is_ok())
        .unwrap_or(false)
}
