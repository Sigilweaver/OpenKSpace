//! Reconstruction-strategy abstraction.
//!
//! A [`ReconStrategy`] turns raw k-space data into a real magnitude image
//! volume. Multiple strategies can coexist: a textbook Fourier recon
//! ([`IfftRss`]), parallel-imaging (GRAPPA/SENSE -- planned), or compressed
//! sensing (planned).
//!
//! This trait is intentionally narrow. It hides the decision of whether a
//! dataset is 2D multi-slice or 3D, whether pre-whitening is enabled, and
//! so forth -- each strategy handles those internally.

use crate::coil::rss_combine_4d;
use crate::crop::center_crop_3d;
use crate::fft::{ifft2_inplace, ifft3_inplace};
use crate::oversampling::OversamplingRemover;
use crate::phasecorr::PhaseCorrector;
use crate::prewhiten::NoisePrewhitener;
use ndarray::Array3;
use openkspace_io::error::IoResult;
use openkspace_io::ismrmrd::IsmrmrdFile;
use tracing::info;

/// A magnitude image volume with simple provenance.
#[derive(Debug)]
pub struct ImageVolume {
    /// Real magnitude image, axis order `[slice, y, x]`.
    pub data: Array3<f32>,
    /// Name of the strategy that produced this volume (for logging).
    pub strategy: &'static str,
}

/// Reconstruction strategy: k-space -> magnitude image.
pub trait ReconStrategy {
    /// A human-readable identifier (used in logs & output filenames).
    fn name(&self) -> &'static str;

    /// Produce a magnitude volume from the given file.
    ///
    /// The strategy owns any pre-processing passes it wants to perform
    /// (reading noise / navigator scans, computing calibrations, etc.).
    fn reconstruct(&self, file: &IsmrmrdFile) -> IoResult<ImageVolume>;
}

/// Which FFT axes to transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftMode {
    /// Pick 2D vs 3D from the acquisitions (2D if `kspace_encode_step_2==0`).
    Auto,
    /// Force per-slice 2D IFFT over `(ky, kx)` only.
    TwoD,
    /// Force 3D IFFT over `(kz, ky, kx)`.
    ThreeD,
}

/// Classical textbook reconstruction: centred IFFT + RSS coil combine,
/// optionally preceded by readout oversampling removal, noise
/// pre-whitening, and navigator phase correction.
#[derive(Debug, Clone, Copy)]
pub struct IfftRss {
    pub remove_oversampling: bool,
    pub prewhiten: bool,
    pub phase_correct: bool,
    pub fft_mode: FftMode,
    pub crop_to_recon_matrix: bool,
}

impl Default for IfftRss {
    fn default() -> Self {
        Self {
            remove_oversampling: true,
            prewhiten: true,
            phase_correct: true,
            fft_mode: FftMode::Auto,
            crop_to_recon_matrix: true,
        }
    }
}

impl ReconStrategy for IfftRss {
    fn name(&self) -> &'static str {
        "ifft-rss"
    }

    fn reconstruct(&self, file: &IsmrmrdFile) -> IoResult<ImageVolume> {
        // --- 1. Calibration pass --------------------------------------------
        let (whitener, phase_corr) = if self.prewhiten || self.phase_correct {
            let cal = file.read_calibration()?;
            let w = if self.prewhiten {
                NoisePrewhitener::from_noise_acqs(&cal.noise)
            } else {
                None
            };
            let pc = if self.phase_correct {
                PhaseCorrector::from_phasecorr_acqs(&cal.phasecorr)
            } else {
                PhaseCorrector::default()
            };
            (w, pc)
        } else {
            (None, PhaseCorrector::default())
        };

        if whitener.is_some() {
            info!("Noise pre-whitening: enabled");
        }
        if !phase_corr.is_empty() {
            info!("Navigator phase correction: enabled");
        }

        // --- 1b. Readout oversampling removal -------------------------------
        let os_remover = if self.remove_oversampling {
            let enc_x = file.header.encoding.encoded_matrix.x as usize;
            let rec_x = file.header.encoding.recon_matrix.x as usize;
            let r = OversamplingRemover::new(enc_x, rec_x);
            if let Some(r) = &r {
                r.log_summary();
            }
            r
        } else {
            None
        };

        // --- 2. Decide 2D vs 3D ---------------------------------------------
        let three_d = match self.fft_mode {
            FftMode::Auto => file.is_3d_encoding()?,
            FftMode::TwoD => false,
            FftMode::ThreeD => true,
        };
        info!(
            "FFT mode: {}",
            if three_d {
                "3D (kz, ky, kx)"
            } else {
                "2D (ky, kx)"
            }
        );

        // --- 3. Image pass: read, apply corrections, place into k-space -----
        let mut kspace = file.read_kspace_with(|acq| {
            if let Some(w) = whitener.as_ref() {
                w.apply(acq);
            }
            phase_corr.apply(acq);
            if let Some(os) = os_remover.as_ref() {
                os.apply(acq);
            }
        })?;

        // --- 4. Centred inverse FFT -----------------------------------------
        //
        // Tensor axes are `[channel, kz, ky, kx]` = `[0, 1, 2, 3]`.
        if three_d {
            info!("Running 3D IFFT on axes (kz=1, ky=2, kx=3)");
            ifft3_inplace(&mut kspace, (1, 2, 3));
        } else {
            info!("Running 2D IFFT on axes (ky=2, kx=3) for all channels/slices");
            ifft2_inplace(&mut kspace, (2, 3));
        }

        // --- 5. RSS coil combine --------------------------------------------
        info!("RSS coil combine");
        let mut magnitude = rss_combine_4d(&kspace);
        drop(kspace);

        // --- 6. Optional crop to the recon matrix ---------------------------
        if self.crop_to_recon_matrix {
            let rm = &file.header.encoding.recon_matrix;
            let (nz, ny, nx) = magnitude.dim();
            let tz = if (rm.z as usize) >= 2 {
                (rm.z as usize).min(nz)
            } else {
                nz
            };
            let ty = if rm.y as usize >= 1 {
                (rm.y as usize).min(ny)
            } else {
                ny
            };
            let tx = if rm.x as usize >= 1 {
                (rm.x as usize).min(nx)
            } else {
                nx
            };
            if (tz, ty, tx) != (nz, ny, nx) && tx <= nx && ty <= ny && tz <= nz {
                info!(
                    "Cropping recon from {}x{}x{} -> {}x{}x{} (recon matrix)",
                    nz, ny, nx, tz, ty, tx
                );
                magnitude = center_crop_3d(&magnitude, (tz, ty, tx));
            }
        }

        Ok(ImageVolume {
            data: magnitude,
            strategy: self.name(),
        })
    }
}

// ----------------------------------------------------------------------------
// GRAPPA
// ----------------------------------------------------------------------------

use crate::grappa::{detect_pattern, extract_acs_slice, GrappaKernel};
use ndarray::s;
use tracing::warn;

/// Parallel-imaging reconstruction via GRAPPA kernel synthesis, followed
/// by IFFT + RSS coil combine.
///
/// Detects the undersampling pattern from the acquisition mask, calibrates
/// a kernel per slice from the auto-calibration region, synthesizes the
/// missing lines, and then runs the standard Fourier path.
///
/// Falls back to [`IfftRss`] behavior when the data is fully sampled or
/// the sampling pattern is unsupported.
#[derive(Debug, Clone, Copy)]
pub struct GrappaRss {
    pub remove_oversampling: bool,
    pub prewhiten: bool,
    pub phase_correct: bool,
    pub kernel_ky: usize,
    pub kernel_kx: usize,
    pub ridge: f32,
    pub fft_mode: FftMode,
    pub crop_to_recon_matrix: bool,
}

impl Default for GrappaRss {
    fn default() -> Self {
        Self {
            remove_oversampling: true,
            prewhiten: true,
            phase_correct: true,
            kernel_ky: 4,
            kernel_kx: 5,
            ridge: 1e-3,
            fft_mode: FftMode::Auto,
            crop_to_recon_matrix: true,
        }
    }
}

impl ReconStrategy for GrappaRss {
    fn name(&self) -> &'static str {
        "grappa-rss"
    }

    fn reconstruct(&self, file: &IsmrmrdFile) -> IoResult<ImageVolume> {
        // --- 1. Calibration pass (same as IfftRss) --------------------------
        let (whitener, phase_corr) = if self.prewhiten || self.phase_correct {
            let cal = file.read_calibration()?;
            let w = if self.prewhiten {
                NoisePrewhitener::from_noise_acqs(&cal.noise)
            } else {
                None
            };
            let pc = if self.phase_correct {
                PhaseCorrector::from_phasecorr_acqs(&cal.phasecorr)
            } else {
                PhaseCorrector::default()
            };
            (w, pc)
        } else {
            (None, PhaseCorrector::default())
        };

        let os_remover = if self.remove_oversampling {
            let enc_x = file.header.encoding.encoded_matrix.x as usize;
            let rec_x = file.header.encoding.recon_matrix.x as usize;
            let r = OversamplingRemover::new(enc_x, rec_x);
            if let Some(r) = &r {
                r.log_summary();
            }
            r
        } else {
            None
        };

        // --- 2. Read k-space with sampling mask -----------------------------
        let (mut kspace, mask) = file.read_kspace_with_mask(|acq| {
            if let Some(w) = whitener.as_ref() {
                w.apply(acq);
            }
            phase_corr.apply(acq);
            if let Some(os) = os_remover.as_ref() {
                os.apply(acq);
            }
        })?;

        // --- 3. Detect pattern ----------------------------------------------
        match detect_pattern(&mask) {
            None => {
                info!(
                    "GRAPPA: data appears fully sampled or pattern unsupported; \
                     skipping kernel synthesis"
                );
            }
            Some(pattern) => {
                info!(
                    "GRAPPA: R={}, ACS ky=[{}, {}) (length {})",
                    pattern.r,
                    pattern.acs_start,
                    pattern.acs_end,
                    pattern.acs_len()
                );
                let nz = kspace.shape()[1];
                // Calibrate per slice and synthesize.
                for kz in 0..nz {
                    let acs = extract_acs_slice(&kspace, kz, &pattern);
                    match GrappaKernel::calibrate(
                        acs.view(),
                        pattern.r,
                        self.kernel_ky,
                        self.kernel_kx,
                        self.ridge,
                    ) {
                        Ok(kernel) => {
                            // Synthesize only on this slice: build a view and
                            // call synthesize on the full tensor -- it walks
                            // all slices but only touches those whose ACS
                            // matches. Simpler: build a per-slice kspace view
                            // by slicing along axis 1 and call synthesize.
                            let mut slice_view = kspace.slice_mut(s![.., kz..=kz, .., ..]);
                            let mut slice_owned = slice_view.to_owned();
                            kernel.synthesize(&mut slice_owned, &pattern);
                            slice_view.assign(&slice_owned);
                        }
                        Err(e) => {
                            warn!(
                                "GRAPPA calibration failed on slice {}: {} \
                                 -- leaving this slice undersampled",
                                kz, e
                            );
                        }
                    }
                }
            }
        }

        // --- 4. IFFT (2D per-slice or full 3D) ------------------------------
        let three_d = match self.fft_mode {
            FftMode::Auto => file.is_3d_encoding()?,
            FftMode::TwoD => false,
            FftMode::ThreeD => true,
        };
        if three_d {
            info!("Running 3D IFFT on axes (kz=1, ky=2, kx=3)");
            ifft3_inplace(&mut kspace, (1, 2, 3));
        } else {
            info!("Running 2D IFFT on axes (ky=2, kx=3) for all channels/slices");
            ifft2_inplace(&mut kspace, (2, 3));
        }

        // --- 5. RSS coil combine --------------------------------------------
        info!("RSS coil combine");
        let mut magnitude = rss_combine_4d(&kspace);
        drop(kspace);

        // --- 6. Optional crop to the recon matrix ---------------------------
        if self.crop_to_recon_matrix {
            let rm = &file.header.encoding.recon_matrix;
            let (nz, ny, nx) = magnitude.dim();
            let tz = if (rm.z as usize) >= 2 {
                (rm.z as usize).min(nz)
            } else {
                nz
            };
            let ty = if rm.y as usize >= 1 {
                (rm.y as usize).min(ny)
            } else {
                ny
            };
            let tx = if rm.x as usize >= 1 {
                (rm.x as usize).min(nx)
            } else {
                nx
            };
            if (tz, ty, tx) != (nz, ny, nx) && tx <= nx && ty <= ny && tz <= nz {
                info!(
                    "Cropping recon from {}x{}x{} -> {}x{}x{} (recon matrix)",
                    nz, ny, nx, tz, ty, tx
                );
                magnitude = center_crop_3d(&magnitude, (tz, ty, tx));
            }
        }

        Ok(ImageVolume {
            data: magnitude,
            strategy: self.name(),
        })
    }
}
