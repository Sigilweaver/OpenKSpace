//! Compressed-sensing reconstruction with an L1 wavelet prior.
//!
//! Implements a single-coil FISTA solver (Beck & Teboulle 2009) for the
//! unconstrained objective
//!
//! ```text
//!   min_x  0.5 * || M F x - y ||_2^2  +  λ * || W x ||_1
//! ```
//!
//! where `F` is a centred 2-D FFT, `M` the sampling mask, `W` the 1-level
//! Haar wavelet transform (see [`crate::wavelet`]), and `y` the measured
//! (zero-filled) k-space data for a single coil.
//!
//! Because `F^H F = I` (with the unitary IFFT used here) and `W^H W = I`,
//! the gradient Lipschitz constant of the data-fidelity term is `1`, and
//! FISTA's step size can be taken as `1`. The proximal operator of
//! `λ ||W · ||_1` is implemented via forward-wavelet + detail-coefficient
//! soft-threshold + inverse-wavelet (see
//! [`crate::wavelet::soft_threshold_details`]).
//!
//! This implementation is intentionally small (no code is copied from any
//! external CS library); it is suitable for demonstrating CS on
//! moderately undersampled Cartesian 2-D acquisitions. For multi-coil
//! data the current strategy applies CS to every coil independently and
//! RSS-combines -- a simple, robust baseline; SENSE-CS joint recon is a
//! natural follow-up.
//!
//! References (credited in `CREDITS.md`, no code copied):
//! * Lustig, Donoho, Pauly, "Sparse MRI", *MRM* 58(6), 2007.
//! * Beck & Teboulle, "A fast iterative shrinkage-thresholding
//!   algorithm", *SIAM J. Imaging Sci.* 2(1), 2009.

use ndarray::Array2;
use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::shift::{fftshift_axis, ifftshift_axis};
use crate::wavelet::{haar_forward, haar_inverse, soft_threshold_details};

/// Errors returned by the CS solver.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum CsError {
    #[error("CS: mask shape {mask:?} does not match kspace shape {kspace:?}")]
    ShapeMismatch { kspace: (usize, usize), mask: (usize, usize) },
    #[error("CS: Ny ({ny}) and Nx ({nx}) must both be even for Haar wavelet")]
    OddDimension { ny: usize, nx: usize },
}

/// Reconstruct one coil's image from zero-filled k-space + sampling mask
/// using `iters` FISTA iterations at regularisation weight `lambda`.
///
/// * `kspace_zf`: `[Ny, Nx]` measured k-space with zeros at unsampled
///   positions (centred convention).
/// * `mask`: `[Ny, Nx]` boolean sampling mask.
pub fn fista_cs_single_coil(
    kspace_zf: &Array2<Complex32>,
    mask: &Array2<bool>,
    iters: usize,
    lambda: f32,
) -> Result<Array2<Complex32>, CsError> {
    let (ny, nx) = kspace_zf.dim();
    if mask.dim() != (ny, nx) {
        return Err(CsError::ShapeMismatch { kspace: (ny, nx), mask: mask.dim() });
    }
    if ny % 2 != 0 || nx % 2 != 0 {
        return Err(CsError::OddDimension { ny, nx });
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);

    // A^H y: adjoint = centred IFFT of (mask-gated) data. Our forward
    // operator A x = mask . F x, so A^H = F^H . mask.
    let mut atb = kspace_zf.clone();
    for i in 0..ny {
        for j in 0..nx {
            if !mask[[i, j]] {
                atb[[i, j]] = Complex32::new(0.0, 0.0);
            }
        }
    }
    centred_ifft2(&mut atb, &*ifft_x, &*ifft_y);

    // FISTA variables.
    let mut x = atb.clone(); // warm-start with zero-filled recon
    let mut z = x.clone();
    let mut t = 1.0f32;

    for _ in 0..iters {
        // Gradient step: g = A^H (A z - y)
        let mut az = z.clone();
        centred_fft2(&mut az, &*fft_x, &*fft_y);
        for i in 0..ny {
            for j in 0..nx {
                if mask[[i, j]] {
                    az[[i, j]] -= kspace_zf[[i, j]];
                } else {
                    az[[i, j]] = Complex32::new(0.0, 0.0);
                }
            }
        }
        centred_ifft2(&mut az, &*ifft_x, &*ifft_y);
        // x_new = prox_{lambda * ||W.||_1}(z - g)
        let mut v = Array2::<Complex32>::zeros((ny, nx));
        for i in 0..ny {
            for j in 0..nx {
                v[[i, j]] = z[[i, j]] - az[[i, j]];
            }
        }
        let mut coef = haar_forward(v.view());
        soft_threshold_details(&mut coef, lambda);
        let x_new = haar_inverse(coef.view());

        // Momentum update.
        let t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        let alpha = (t - 1.0) / t_new;
        let mut z_new = Array2::<Complex32>::zeros((ny, nx));
        for i in 0..ny {
            for j in 0..nx {
                z_new[[i, j]] =
                    x_new[[i, j]] + Complex32::new(alpha, 0.0) * (x_new[[i, j]] - x[[i, j]]);
            }
        }
        x = x_new;
        z = z_new;
        t = t_new;
    }
    Ok(x)
}

fn centred_fft2(
    a: &mut Array2<Complex32>,
    fft_x: &dyn rustfft::Fft<f32>,
    fft_y: &dyn rustfft::Fft<f32>,
) {
    let (ny, nx) = a.dim();
    ifftshift_axis(a, 0);
    ifftshift_axis(a, 1);
    let mut row = vec![Complex32::new(0.0, 0.0); nx];
    for i in 0..ny {
        for j in 0..nx {
            row[j] = a[[i, j]];
        }
        fft_x.process(&mut row);
        for j in 0..nx {
            a[[i, j]] = row[j];
        }
    }
    let mut col = vec![Complex32::new(0.0, 0.0); ny];
    for j in 0..nx {
        for i in 0..ny {
            col[i] = a[[i, j]];
        }
        fft_y.process(&mut col);
        for i in 0..ny {
            a[[i, j]] = col[i];
        }
    }
    fftshift_axis(a, 0);
    fftshift_axis(a, 1);
    // Unitary normalisation.
    let s = 1.0 / ((ny as f32 * nx as f32).sqrt());
    for i in 0..ny {
        for j in 0..nx {
            a[[i, j]] = a[[i, j]] * Complex32::new(s, 0.0);
        }
    }
}

fn centred_ifft2(
    a: &mut Array2<Complex32>,
    ifft_x: &dyn rustfft::Fft<f32>,
    ifft_y: &dyn rustfft::Fft<f32>,
) {
    let (ny, nx) = a.dim();
    ifftshift_axis(a, 0);
    ifftshift_axis(a, 1);
    let mut row = vec![Complex32::new(0.0, 0.0); nx];
    for i in 0..ny {
        for j in 0..nx {
            row[j] = a[[i, j]];
        }
        ifft_x.process(&mut row);
        for j in 0..nx {
            a[[i, j]] = row[j];
        }
    }
    let mut col = vec![Complex32::new(0.0, 0.0); ny];
    for j in 0..nx {
        for i in 0..ny {
            col[i] = a[[i, j]];
        }
        ifft_y.process(&mut col);
        for i in 0..ny {
            a[[i, j]] = col[i];
        }
    }
    fftshift_axis(a, 0);
    fftshift_axis(a, 1);
    // Unitary normalisation: IFFT = conj(FFT)/N, so for unitary we
    // multiply by sqrt(N)/N = 1/sqrt(N).
    let s = 1.0 / ((ny as f32 * nx as f32).sqrt());
    for i in 0..ny {
        for j in 0..nx {
            a[[i, j]] = a[[i, j]] * Complex32::new(s, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_pair_is_unitary() {
        let mut planner = FftPlanner::<f32>::new();
        let ny = 8;
        let nx = 8;
        let fft_x = planner.plan_fft_forward(nx);
        let fft_y = planner.plan_fft_forward(ny);
        let ifft_x = planner.plan_fft_inverse(nx);
        let ifft_y = planner.plan_fft_inverse(ny);
        let mut x = Array2::<Complex32>::zeros((ny, nx));
        for i in 0..ny {
            for j in 0..nx {
                x[[i, j]] = Complex32::new((i + j) as f32, (i as f32 - j as f32) * 0.3);
            }
        }
        let e_in: f32 = x.iter().map(|c| c.norm_sqr()).sum();
        let mut y = x.clone();
        centred_fft2(&mut y, &*fft_x, &*fft_y);
        let e_mid: f32 = y.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (e_in - e_mid).abs() < 1e-3,
            "unitary FFT lost energy: {} -> {}",
            e_in,
            e_mid
        );
        centred_ifft2(&mut y, &*ifft_x, &*ifft_y);
        for i in 0..ny {
            for j in 0..nx {
                let e = (y[[i, j]] - x[[i, j]]).norm();
                assert!(e < 1e-4, "roundtrip err {} at ({},{})", e, i, j);
            }
        }
    }

    #[test]
    fn cs_recovers_sparse_phantom() {
        // Sparse phantom: a handful of isolated delta-like blocks.
        let ny = 16;
        let nx = 16;
        let mut truth = Array2::<Complex32>::zeros((ny, nx));
        truth[[5, 4]] = Complex32::new(1.0, 0.0);
        truth[[10, 11]] = Complex32::new(0.8, 0.0);
        truth[[3, 12]] = Complex32::new(0.6, 0.0);
        truth[[12, 3]] = Complex32::new(0.5, 0.0);

        // Full k-space.
        let mut planner = FftPlanner::<f32>::new();
        let fft_x = planner.plan_fft_forward(nx);
        let fft_y = planner.plan_fft_forward(ny);
        let ifft_x = planner.plan_fft_inverse(nx);
        let ifft_y = planner.plan_fft_inverse(ny);
        let mut k = truth.clone();
        centred_fft2(&mut k, &*fft_x, &*fft_y);

        // R=2 uniform ky mask with 4 central ACS lines.
        let mut mask = Array2::<bool>::from_elem((ny, nx), false);
        for i in 0..ny {
            if i % 2 == 0 || (ny / 2 - 2..ny / 2 + 2).contains(&i) {
                for j in 0..nx {
                    mask[[i, j]] = true;
                }
            }
        }
        // Zero-fill.
        let mut kzf = k.clone();
        for i in 0..ny {
            for j in 0..nx {
                if !mask[[i, j]] {
                    kzf[[i, j]] = Complex32::new(0.0, 0.0);
                }
            }
        }

        // Zero-filled recon baseline.
        let mut zfimg = kzf.clone();
        centred_ifft2(&mut zfimg, &*ifft_x, &*ifft_y);
        let zf_err: f32 = zfimg
            .iter()
            .zip(truth.iter())
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f32>()
            .sqrt();

        let recon = fista_cs_single_coil(&kzf, &mask, 200, 0.02).expect("CS failed");
        let cs_err: f32 = recon
            .iter()
            .zip(truth.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f32>()
            .sqrt();

        assert!(
            cs_err < 0.8 * zf_err,
            "CS did not improve over zero-fill: cs={:.4} zf={:.4}",
            cs_err,
            zf_err
        );
    }
}
