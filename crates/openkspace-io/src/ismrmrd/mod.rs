//! ISMRMRD (International Society for Magnetic Resonance in Medicine Raw Data)
//! HDF5 reader.
//!
//! Layout of a dataset (confirmed against mridata.org files):
//!
//! ```text
//! /dataset/xml   : variable-length UTF-8 string -- ISMRMRD XML header
//! /dataset/data  : compound[N] where each record is:
//!     head : AcquisitionHeader  (fixed 340 bytes)
//!     traj : vlen<f32>          (trajectory samples -- empty for cartesian)
//!     data : vlen<f32>          (interleaved real/imag:
//!                                len = number_of_samples * active_channels * 2)
//! ```
//!
//! This module decodes the header, streams all acquisitions, and assembles
//! a cartesian k-space tensor of shape `[channels, kz, ky, kx]`.

pub mod acquisition;
pub mod header;
pub mod reader;

pub use acquisition::{Acquisition, AcquisitionHeader, EncodingCounters};
pub use header::{EncodingInfo, IsmrmrdHeader, MatrixSize};
pub use reader::{CalibrationScans, IsmrmrdFile};
