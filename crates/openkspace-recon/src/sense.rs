//! Image-domain SENSE (SENSitivity Encoding) unfolding for regular
//! 1-D Cartesian undersampling along ky.
//!
//! Reference: Pruessmann, Weiger, Scheidegger, Boesiger, "SENSE:
//! sensitivity encoding for fast MRI", *MRM* 42(5), 1999.
//!
//! Given
//!   * full-FOV complex sensitivity maps `S_c(y, x)`, `c = 0..Nc`,
//!     obtained from [`crate::sensitivity::walsh_sensitivity_maps`];
//!   * aliased coil images `I_c(y, x)` produced by an inverse FFT of
//!     the zero-filled undersampled k-space (so `I_c` has the same
//!     `Ny × Nx` shape but is periodic along y with period `Ny/R`);
//!
//! we solve a separate `R`-variable least-squares system at every
//! voxel `(y0, x)` with `y0 in 0..Ny/R`:
//!
//! ```text
//!     C(y0, x) rho(y0, x) = a(y0, x),
//!     C[c, k]  = S_c(y0 + k*Ny/R, x),   k = 0..R
//!     a[c]     = I_c(y0, x) * R,        (1/R factor absorbed)
//! ```
//!
//! The regularized normal-equation solution
//! `rho = (C^H C + λ I)^{-1} C^H a` is scattered back into the full-FOV
//! output at positions `{y0 + k*Ny/R}`.
//!
//! Nothing here is copied from any external SENSE implementation; the
//! algebra follows directly from the Pruessmann paper.

use ndarray::{Array2, Array3};
use num_complex::Complex32;

use crate::prewhiten::{cholesky_lower, invert_lower_triangular};

/// Unfold an aliased coil-image stack along axis 1 using SENSE.
///
/// * `aliased`: shape `[Nc, Ny, Nx]` - complex coil images after IFFT
///   of the zero-filled undersampled k-space. Expected to be periodic
///   along y with period `Ny/R` (i.e. the aliased replicas are
///   identical). Only rows `0..Ny/R` are read.
/// * `maps`: shape `[Nc, Ny, Nx]` - full-FOV complex sensitivity maps.
/// * `r`: integer acceleration factor (must divide `Ny`).
/// * `ridge`: Tikhonov regularisation added to `C^H C` (in the
///   g-factor-noisy voxels the system is ill-conditioned; a small
///   positive ridge stabilises the inversion).
///
/// Returns the unfolded complex image `[Ny, Nx]`. Take `.norm()` of
/// every element to get a magnitude image.
pub fn sense_unfold_1d(
    aliased: &Array3<Complex32>,
    maps: &Array3<Complex32>,
    r: usize,
    ridge: f32,
) -> Array2<Complex32> {
    let (nc, ny, nx) = aliased.dim();
    assert!(r >= 1, "SENSE: acceleration factor must be >= 1");
    assert_eq!(maps.dim(), (nc, ny, nx), "SENSE: map/aliased shape mismatch");
    assert!(ny % r == 0, "SENSE: Ny ({}) must be divisible by R ({})", ny, r);
    let ny_red = ny / r;

    let mut out = Array2::<Complex32>::zeros((ny, nx));
    if nc == 0 || nx == 0 || ny_red == 0 {
        return out;
    }

    // Pre-allocated scratch for the NcxR SENSE system.
    let mut c_mat = Array2::<Complex32>::zeros((nc, r));
    let mut chc = Array2::<Complex32>::zeros((r, r));
    let mut rhs = vec![Complex32::new(0.0, 0.0); r];

    for x in 0..nx {
        for y0 in 0..ny_red {
            // Build C (Nc x R) from sensitivity maps at the R aliased rows.
            for c in 0..nc {
                for k in 0..r {
                    c_mat[[c, k]] = maps[[c, y0 + k * ny_red, x]];
                }
            }

            // C^H C (R x R Hermitian) with Tikhonov ridge.
            for a in 0..r {
                for b in 0..r {
                    let mut acc = Complex32::new(0.0, 0.0);
                    for c in 0..nc {
                        acc += c_mat[[c, a]].conj() * c_mat[[c, b]];
                    }
                    if a == b {
                        acc += Complex32::new(ridge, 0.0);
                    }
                    chc[[a, b]] = acc;
                }
            }

            // rhs = C^H a, where a = aliased[:, y0, x] * R.
            for a in 0..r {
                let mut acc = Complex32::new(0.0, 0.0);
                for c in 0..nc {
                    acc += c_mat[[c, a]].conj() * aliased[[c, y0, x]];
                }
                rhs[a] = acc * Complex32::new(r as f32, 0.0);
            }

            // Solve (chc) rho = rhs via Cholesky. If the ridge is small
            // and the maps happen to be zero in the background, chc may
            // fail to be positive-definite -- in that case leave those
            // pixels zero.
            let Some(lower) = cholesky_lower(&chc) else {
                continue;
            };
            let Some(inv) = invert_lower_triangular(&lower) else {
                continue;
            };
            // rho = inv^H * (inv * rhs)
            let mut tmp = vec![Complex32::new(0.0, 0.0); r];
            for i in 0..r {
                let mut acc = Complex32::new(0.0, 0.0);
                for j in 0..=i {
                    acc += inv[[i, j]] * rhs[j];
                }
                tmp[i] = acc;
            }
            let mut rho = vec![Complex32::new(0.0, 0.0); r];
            for i in 0..r {
                let mut acc = Complex32::new(0.0, 0.0);
                for j in i..r {
                    acc += inv[[j, i]].conj() * tmp[j];
                }
                rho[i] = acc;
            }

            // Scatter into the full-FOV output.
            for k in 0..r {
                out[[y0 + k * ny_red, x]] = rho[k];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rustfft::FftPlanner;
    use std::f32::consts::PI;

    /// SENSE on a synthetic 4-coil R=2 dataset with known sensitivities
    /// should recover the truth with low NRMSE.
    #[test]
    fn sense_unfolds_r2_phantom_below_threshold() {
        let nc = 4usize;
        let ny = 32usize;
        let nx = 24usize;
        let r = 2usize;

        // 1. Build a real phantom (rectangle + small dot).
        let mut truth = Array2::<f32>::zeros((ny, nx));
        for y in 6..26 {
            for x in 4..20 {
                truth[[y, x]] = 1.0;
            }
        }
        for y in 14..18 {
            for x in 10..14 {
                truth[[y, x]] = 0.3;
            }
        }

        // 2. Four gaussian coil sensitivities located at 4 corners.
        let mut maps = Array3::<Complex32>::zeros((nc, ny, nx));
        let centres = [
            (0.25, 0.25),
            (0.25, 0.75),
            (0.75, 0.25),
            (0.75, 0.75),
        ];
        for (c, (fy, fx)) in centres.iter().enumerate() {
            let cy = fy * ny as f32;
            let cx = fx * nx as f32;
            for y in 0..ny {
                for x in 0..nx {
                    let dy = y as f32 - cy;
                    let dx = x as f32 - cx;
                    let mag = (-(dy * dy + dx * dx) / 200.0).exp();
                    let ph = 0.2 * (c as f32);
                    maps[[c, y, x]] = Complex32::new(mag * ph.cos(), mag * ph.sin());
                }
            }
        }

        // 3. Coil images c_c = S_c * truth.
        let mut coil_img = Array3::<Complex32>::zeros((nc, ny, nx));
        for c in 0..nc {
            for y in 0..ny {
                for x in 0..nx {
                    coil_img[[c, y, x]] = maps[[c, y, x]] * truth[[y, x]];
                }
            }
        }

        // 4. FFT each coil image, drop every other ky line, inverse FFT.
        //    Use plain (non-centered) FFT/IFFT here -- the aliasing
        //    relation holds identically in either convention since
        //    a cyclic shift cannot break periodicity.
        let mut planner = FftPlanner::<f32>::new();
        let fft_y = planner.plan_fft_forward(ny);
        let ifft_y = planner.plan_fft_inverse(ny);

        let mut aliased = Array3::<Complex32>::zeros((nc, ny, nx));
        let mut buf = vec![Complex32::new(0.0, 0.0); ny];
        for c in 0..nc {
            for x in 0..nx {
                for y in 0..ny {
                    buf[y] = coil_img[[c, y, x]];
                }
                fft_y.process(&mut buf);
                // Keep only every R-th ky, zero the rest.
                for (i, v) in buf.iter_mut().enumerate() {
                    if i % r != 0 {
                        *v = Complex32::new(0.0, 0.0);
                    }
                }
                ifft_y.process(&mut buf);
                let scale = 1.0 / ny as f32;
                for y in 0..ny {
                    aliased[[c, y, x]] = buf[y] * scale;
                }
            }
        }

        // Sanity: the aliased image should be periodic along y.
        let err: f32 = (0..nc)
            .flat_map(|c| (0..nx).map(move |x| (c, x)))
            .map(|(c, x)| {
                (aliased[[c, 0, x]] - aliased[[c, ny / r, x]]).norm_sqr()
            })
            .sum::<f32>()
            .sqrt();
        assert!(err < 1e-3, "aliased image should be ny/r-periodic, err={}", err);

        // 5. SENSE unfold.
        let out = sense_unfold_1d(&aliased, &maps, r, 1e-5);

        // 6. Compute NRMSE vs truth (magnitudes).
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for y in 0..ny {
            for x in 0..nx {
                let t = truth[[y, x]];
                let m = out[[y, x]].norm();
                num += (t - m) * (t - m);
                den += t * t;
            }
        }
        let nrmse = (num / den.max(1e-20)).sqrt();
        assert!(nrmse < 0.1, "SENSE NRMSE {} too large", nrmse);
    }

    /// Degenerate case: R=1 should pass the (1x1) system through
    /// essentially unchanged, so SENSE with R=1 reduces to an
    /// RSS-equivalent map-weighted coil combine.
    #[test]
    fn sense_r1_passthrough() {
        let nc = 2;
        let ny = 8;
        let nx = 6;
        // Constant-phase unit maps, gaussian "aliased" values.
        let mut maps = Array3::<Complex32>::zeros((nc, ny, nx));
        let mut aliased = Array3::<Complex32>::zeros((nc, ny, nx));
        for c in 0..nc {
            for y in 0..ny {
                for x in 0..nx {
                    maps[[c, y, x]] = Complex32::new(1.0, 0.0);
                    let v = ((y as f32) * 0.1 + (x as f32) * 0.05 + c as f32).sin();
                    aliased[[c, y, x]] = Complex32::new(v, 0.0);
                }
            }
        }
        let out = sense_unfold_1d(&aliased, &maps, 1, 0.0);
        // With S_c = 1 for all coils, R=1, ridge=0:
        //   rho = (sum_c 1) a_c / (sum_c 1) = mean(a_c) ... not exactly;
        //   (C^H C)^-1 C^H a = (nc)^-1 * sum_c a_c
        // So out(y,x) = mean_c(aliased[c,y,x]).
        let _ = Array1::<f32>::zeros(1);
        for y in 0..ny {
            for x in 0..nx {
                let mut mean = Complex32::new(0.0, 0.0);
                for c in 0..nc {
                    mean += aliased[[c, y, x]];
                }
                mean = mean * Complex32::new(1.0 / nc as f32, 0.0);
                let err = (out[[y, x]] - mean).norm();
                assert!(err < 1e-4, "R=1 passthrough err={}", err);
            }
        }
        // Kill unused-import warnings.
        let _ = PI;
    }
}
