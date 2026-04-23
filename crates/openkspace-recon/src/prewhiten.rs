//! Noise pre-whitening from ISMRMRD noise-adjust scans.
//!
//! ## Background
//!
//! Every receive channel has its own thermal noise; different coils also share
//! noise through preamp coupling and body loading. Left uncorrected this
//! produces spatially non-uniform noise in an RSS image and breaks the
//! assumptions behind parallel-imaging reconstructions (SENSE, GRAPPA,
//! ESPIRiT).
//!
//! Pre-whitening applies a linear decorrelating transform `W` to every
//! readout such that the transformed channels have unit covariance:
//!
//! ```text
//! Psi  = (1 / (N-1)) * Sum n_k n_k^H          (sample covariance of N noise samples)
//! L  = cholesky(Psi)                     (lower triangular)
//! W  = L^-1                             (lower triangular inverse)
//! s' = W * s                           (applied to each acquired readout)
//! ```
//!
//! ## Reference
//!
//! *Kellman P, McVeigh ER.* "Image reconstruction in SNR units: a general
//! method for SNR measurement." **MRM** 54(6):1439-1447, 2005. -- original
//! description of the Cholesky-based noise pre-whitening transform used
//! throughout modern MRI reconstruction.

use ndarray::Array2;
use num_complex::Complex32;
use openkspace_io::ismrmrd::Acquisition;
use tracing::{info, warn};

/// Pre-whitens complex multi-channel readouts so that inter-channel noise
/// covariance becomes the identity.
#[derive(Debug, Clone)]
pub struct NoisePrewhitener {
    /// `W = L^-1`, the lower-triangular whitening matrix `[nc, nc]`.
    whitening: Array2<Complex32>,
    /// Number of channels (= matrix side length).
    nc: usize,
}

impl NoisePrewhitener {
    /// Build a whitener from the sample covariance of one or more noise scans.
    ///
    /// Returns `None` if no noise samples are available -- callers should
    /// simply skip pre-whitening in that case.
    pub fn from_noise_acqs(noise: &[Acquisition]) -> Option<Self> {
        if noise.is_empty() {
            return None;
        }
        let nc = noise[0].num_channels();
        if nc == 0 {
            return None;
        }

        // Total sample count across all noise scans
        let total_samples: usize = noise.iter().map(|a| a.num_samples()).sum();
        if total_samples < nc {
            warn!(
                "Noise calibration: only {} samples for {} channels -- \
                 whitening matrix is under-determined, skipping.",
                total_samples, nc
            );
            return None;
        }

        // Accumulate Psi = (1/(N-1)) * Sum_n s(n) s(n)^H over all samples.
        //
        // s(n) is the channel vector at sample n; Psi is an [nc, nc] complex
        // matrix. This is done as a straight triple loop -- noise scans are
        // small (typically a few hundred samples), so BLAS integration
        // would be overkill.
        let mut psi = Array2::<Complex32>::zeros((nc, nc));
        for acq in noise {
            if acq.num_channels() != nc {
                warn!(
                    "Noise scan has {} channels but expected {} -- skipped",
                    acq.num_channels(),
                    nc
                );
                continue;
            }
            let view = acq.as_array_view(); // [nc, ns]
            let ns = view.ncols();
            for n in 0..ns {
                let col = view.column(n);
                // Outer product s * s^H accumulated into psi.
                for i in 0..nc {
                    let si = col[i];
                    for j in 0..nc {
                        psi[[i, j]] += si * col[j].conj();
                    }
                }
            }
        }

        let denom = (total_samples as f32 - 1.0).max(1.0);
        psi.mapv_inplace(|v| v / Complex32::new(denom, 0.0));

        // Cholesky: Psi = L * L^H, then W = L^-1 (forward-substitution).
        let l = cholesky_lower(&psi)?;
        let whitening = invert_lower_triangular(&l)?;

        info!(
            "Noise pre-whitening calibrated from {} channels x {} samples",
            nc, total_samples
        );

        Some(Self { whitening, nc })
    }

    /// Apply the whitening matrix to every sample of an acquisition in place.
    ///
    /// The acquisition's `data` is interpreted as `[nc, ns]`; each column
    /// vector `s` is replaced by `W * s`.
    pub fn apply(&self, acq: &mut Acquisition) {
        if acq.num_channels() != self.nc {
            // Silently skip mismatched acquisitions (e.g. calibration scans
            // from a different channel count).
            return;
        }
        let nc = self.nc;
        let ns = acq.num_samples();
        let mut view = acq.as_array_view_mut(); // [nc, ns]
        let mut buf = vec![Complex32::new(0.0, 0.0); nc];
        for n in 0..ns {
            // Copy column to buf
            for i in 0..nc {
                buf[i] = view[(i, n)];
            }
            // Lower-triangular multiply W * buf -> column
            for i in 0..nc {
                let mut acc = Complex32::new(0.0, 0.0);
                for j in 0..=i {
                    acc += self.whitening[[i, j]] * buf[j];
                }
                view[(i, n)] = acc;
            }
        }
    }
}

/// Cholesky factorization of a Hermitian positive-definite complex matrix.
/// Returns the lower triangular factor `L` with `A = L * L^H`, or `None` if
/// the matrix is not positive-definite.
pub(crate) fn cholesky_lower(a: &Array2<Complex32>) -> Option<Array2<Complex32>> {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    let mut l = Array2::<Complex32>::zeros((n, n));

    for i in 0..n {
        // Diagonal element
        let mut diag = a[[i, i]];
        for k in 0..i {
            diag -= l[[i, k]] * l[[i, k]].conj();
        }
        // Numerical diagonal must be real and positive
        let d_re = diag.re;
        if !(d_re.is_finite()) || d_re <= 0.0 {
            warn!("Cholesky: non-positive-definite at row {i} (diag={diag:?})");
            return None;
        }
        let l_ii = Complex32::new(d_re.sqrt(), 0.0);
        l[[i, i]] = l_ii;
        let inv_l_ii = Complex32::new(1.0 / l_ii.re, 0.0);

        // Below-diagonal column
        for j in (i + 1)..n {
            let mut s = a[[j, i]];
            for k in 0..i {
                s -= l[[j, k]] * l[[i, k]].conj();
            }
            l[[j, i]] = s * inv_l_ii;
        }
    }
    Some(l)
}

/// Invert a lower-triangular matrix via forward substitution on each
/// unit vector column.
pub(crate) fn invert_lower_triangular(l: &Array2<Complex32>) -> Option<Array2<Complex32>> {
    let n = l.nrows();
    let mut inv = Array2::<Complex32>::zeros((n, n));

    for col in 0..n {
        // Solve L * x = e_col
        let mut x = vec![Complex32::new(0.0, 0.0); n];
        x[col] = Complex32::new(1.0, 0.0);
        for i in 0..n {
            let mut s = x[i];
            for j in 0..i {
                s -= l[[i, j]] * inv[[j, col]];
            }
            let diag = l[[i, i]];
            if diag.re.abs() < f32::EPSILON && diag.im.abs() < f32::EPSILON {
                return None;
            }
            inv[[i, col]] = s / diag;
        }
    }
    // zero out strict upper triangle (already zero, but be explicit)
    for i in 0..n {
        for j in (i + 1)..n {
            inv[[i, j]] = Complex32::new(0.0, 0.0);
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Given synthetic correlated noise, the whitened covariance should
    /// converge to the identity.
    #[test]
    fn whitens_correlated_noise_to_identity() {
        // Build a known L, generate noise = L * n_0 where n_0 is i.i.d. ~ CN(0,1),
        // feed it through the whitener, and check the output covariance ~= I.
        //
        // We construct deterministic "noise" by using a fixed LCG so the test
        // is reproducible without a random dependency.
        let nc = 4;
        let ns = 4096;

        // True L (lower triangular, positive diagonal).
        let mut l_true = Array2::<Complex32>::zeros((nc, nc));
        for i in 0..nc {
            l_true[[i, i]] = Complex32::new(1.0 + 0.5 * i as f32, 0.0);
            for j in 0..i {
                l_true[[i, j]] = Complex32::new(0.1 * (i + j) as f32, -0.05 * i as f32);
            }
        }

        // Deterministic CN-ish samples.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as u32) as f32 / u32::MAX as f32;
            u - 0.5
        };

        // Build the acquisition's data vector [nc*ns] = L * n_0
        let mut data = vec![Complex32::new(0.0, 0.0); nc * ns];
        for n in 0..ns {
            let n0: Vec<Complex32> = (0..nc).map(|_| Complex32::new(rng(), rng())).collect();
            for i in 0..nc {
                let mut s = Complex32::new(0.0, 0.0);
                for j in 0..=i {
                    s += l_true[[i, j]] * n0[j];
                }
                data[i * ns + n] = s;
            }
        }

        // Wrap in an Acquisition
        use openkspace_io::ismrmrd::AcquisitionHeader;
        let mut header: AcquisitionHeader = unsafe { std::mem::zeroed() };
        header.number_of_samples = ns as u16;
        header.active_channels = nc as u16;
        let flat: Vec<f32> = data.iter().flat_map(|c| [c.re, c.im]).collect();
        let acq = Acquisition::from_raw_f32(header, &flat);

        let whitener = NoisePrewhitener::from_noise_acqs(&[acq]).expect("cov should be PD");

        // Pass the same samples through and verify covariance ~= I.
        let mut acq2 = Acquisition::from_raw_f32(header, &flat);
        whitener.apply(&mut acq2);

        let view = acq2.as_array_view();
        let mut cov = Array2::<Complex32>::zeros((nc, nc));
        for n in 0..ns {
            let col = view.column(n);
            for i in 0..nc {
                for j in 0..nc {
                    cov[[i, j]] += col[i] * col[j].conj();
                }
            }
        }
        cov.mapv_inplace(|v| v / Complex32::new((ns - 1) as f32, 0.0));

        for i in 0..nc {
            for j in 0..nc {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (cov[[i, j]].re - target).abs() < 0.08,
                    "cov[{i},{j}].re = {} (expected ~= {})",
                    cov[[i, j]].re,
                    target
                );
                assert!(
                    cov[[i, j]].im.abs() < 0.08,
                    "cov[{i},{j}].im = {} (expected ~= 0)",
                    cov[[i, j]].im
                );
            }
        }
    }
}
