//! OpenKSpace-Recon: K-space -> image reconstruction math.
//!
//! Core pipeline:
//!   1. (optional) noise pre-whitening      -- `prewhiten`
//!   2. (optional) navigator phase correction -- `phasecorr`
//!   3. fftshift -> IFFT -> fftshift           -- `fft` + `shift`
//!   4. RSS coil combine                     -- `coil`
//!   5. (optional) centre crop to recon matrix -- `crop`
//!
//! Reconstructions are composed behind the [`ReconStrategy`] trait so that
//! future parallel-imaging / compressed-sensing back-ends can slot in
//! without touching this crate's public shape.

pub mod coil;
pub mod crop;
pub mod fft;
pub mod oversampling;
pub mod phasecorr;
pub mod prewhiten;
pub mod shift;
pub mod strategy;

pub use coil::rss_combine;
pub use crop::center_crop_3d;
pub use fft::{ifft2_inplace, ifft3_inplace};
pub use oversampling::OversamplingRemover;
pub use phasecorr::PhaseCorrector;
pub use prewhiten::NoisePrewhitener;
pub use shift::{fftshift_axis, ifftshift_axis};
pub use strategy::{FftMode, IfftRss, ImageVolume, ReconStrategy};
