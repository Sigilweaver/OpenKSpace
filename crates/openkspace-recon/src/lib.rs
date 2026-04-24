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
pub mod grappa;
pub mod oversampling;
pub mod partial_fourier;
pub mod phasecorr;
pub mod prewhiten;
pub mod sense;
pub mod sensitivity;
pub mod shift;
pub mod strategy;

pub use coil::rss_combine;
pub use crop::center_crop_3d;
pub use fft::{ifft2_inplace, ifft3_inplace};
pub use grappa::{GrappaKernel, SamplingPattern};
pub use oversampling::OversamplingRemover;
pub use partial_fourier::{homodyne_reconstruct, PartialFourierPlan};
pub use phasecorr::PhaseCorrector;
pub use prewhiten::NoisePrewhitener;
pub use sense::sense_unfold_1d;
pub use sensitivity::walsh_sensitivity_maps;
pub use shift::{fftshift_axis, ifftshift_axis};
pub use strategy::{FftMode, GrappaRss, IfftRss, ImageVolume, ReconStrategy, SenseRss};
