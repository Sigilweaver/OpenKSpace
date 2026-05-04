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
pub mod cs;
pub mod espirit;
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
pub mod wavelet;

pub use coil::{rss_combine, rss_combine_4d};
pub use crop::center_crop_3d;
pub use cs::{fista_cs_single_coil, CsError};
pub use espirit::espirit_sensitivity_maps;
pub use fft::{ifft1_inplace, ifft2_inplace, ifft3_inplace};
pub use grappa::{GrappaError, GrappaKernel, SamplingPattern};
pub use oversampling::OversamplingRemover;
pub use partial_fourier::{homodyne_reconstruct, PartialFourierPlan};
pub use phasecorr::PhaseCorrector;
pub use prewhiten::NoisePrewhitener;
pub use sense::{sense_gfactor_1d, sense_unfold_1d, SenseError};
pub use sensitivity::walsh_sensitivity_maps;
pub use shift::{fftshift_axis, ifftshift_axis};
pub use strategy::{
    CsRss, FftMode, GrappaRss, IfftRss, ImageVolume, ReconStrategy, SenseMapSource, SenseRss,
};
