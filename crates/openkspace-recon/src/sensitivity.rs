//! Coil sensitivity estimation via the Walsh adaptive method.
//!
//! Given low-resolution coil images `I_c(y, x)` (typically obtained by
//! zero-filling the ACS region and inverse-FFTing), we want per-coil
//! complex sensitivity maps `S_c(y, x)` that are smooth in space and
//! whose norm follows the local signal intensity.
//!
//! Walsh, Gmitro, Marcellin ("Adaptive reconstruction of phased array
//! MR imagery", *MRM* 43(5), 2000) derive the optimal coil-combination
//! weights at each voxel as the dominant eigenvector of the `Nc × Nc`
//! sample covariance taken over a small spatial neighbourhood. The same
//! eigenvector, scaled by the square root of the dominant eigenvalue,
//! is a standard estimate of the complex sensitivity vector because it
//! captures both the direction of the coil response and its magnitude.
//!
//! Implementation notes (no code copied from any other project):
//!
//! * Covariance is accumulated on a rectangular window of half-size
//!   `window` in both dimensions, clipped at image boundaries.
//! * The dominant eigenvector is obtained by power iteration (fixed
//!   number of steps), initialized from the coil vector at the centre
//!   voxel so we never start from a zero vector on signal pixels.
//! * The global phase ambiguity is fixed per voxel by rotating the
//!   eigenvector so that the largest-magnitude coil is real-positive.
//!   This removes the random phase jumps between neighbouring voxels
//!   that power iteration would otherwise produce.

use ndarray::{Array2, Array3};
use num_complex::Complex32;

/// Estimate complex coil sensitivity maps from low-resolution coil images.
///
/// Input shape: `[nc, ny, nx]`. Output has the same shape. `window` is the
/// half-size of the accumulation neighbourhood (e.g. `window = 3` uses a
/// 7x7 window). `power_iters` is the number of power-iteration steps.
#[allow(clippy::needless_range_loop)]
pub fn walsh_sensitivity_maps(
    coil_imgs: &Array3<Complex32>,
    window: usize,
    power_iters: usize,
) -> Array3<Complex32> {
    let (nc, ny, nx) = coil_imgs.dim();
    let mut out = Array3::<Complex32>::zeros((nc, ny, nx));
    if nc == 0 || ny == 0 || nx == 0 {
        return out;
    }

    for y in 0..ny {
        let y0 = y.saturating_sub(window);
        let y1 = (y + window + 1).min(ny);
        for x in 0..nx {
            let x0 = x.saturating_sub(window);
            let x1 = (x + window + 1).min(nx);

            // Accumulate Hermitian covariance R = sum_nbhd s s^H
            let mut r = Array2::<Complex32>::zeros((nc, nc));
            for yy in y0..y1 {
                for xx in x0..x1 {
                    for i in 0..nc {
                        let si = coil_imgs[[i, yy, xx]];
                        for j in 0..nc {
                            let sj = coil_imgs[[j, yy, xx]];
                            r[[i, j]] += si * sj.conj();
                        }
                    }
                }
            }

            // Power iteration for dominant eigenvector of r.
            // Initialize with the centre voxel's coil vector (fall back
            // to e_0 if it is identically zero).
            let mut v = vec![Complex32::new(0.0, 0.0); nc];
            let mut init_norm = 0.0f32;
            for c in 0..nc {
                let s = coil_imgs[[c, y, x]];
                v[c] = s;
                init_norm += s.norm_sqr();
            }
            if init_norm == 0.0 {
                v[0] = Complex32::new(1.0, 0.0);
            } else {
                let inv = 1.0 / init_norm.sqrt();
                for c in 0..nc {
                    v[c] *= inv;
                }
            }

            let mut lambda = 0.0f32;
            for _ in 0..power_iters {
                // w = R v
                let mut w = vec![Complex32::new(0.0, 0.0); nc];
                for i in 0..nc {
                    let mut acc = Complex32::new(0.0, 0.0);
                    for j in 0..nc {
                        acc += r[[i, j]] * v[j];
                    }
                    w[i] = acc;
                }
                let mut nrm = 0.0f32;
                for c in 0..nc {
                    nrm += w[c].norm_sqr();
                }
                nrm = nrm.sqrt();
                if nrm < 1e-20 {
                    break;
                }
                let inv = 1.0 / nrm;
                for c in 0..nc {
                    v[c] = w[c] * inv;
                }
                lambda = nrm; // Rayleigh quotient in the unit-norm regime
            }

            // Fix global phase: rotate so the coil with the largest
            // magnitude is real and positive.
            let mut imax = 0usize;
            let mut mmax = 0.0f32;
            for c in 0..nc {
                let m = v[c].norm();
                if m > mmax {
                    mmax = m;
                    imax = c;
                }
            }
            if mmax > 0.0 {
                let ref_val = v[imax];
                let phase = Complex32::new(ref_val.re / mmax, -ref_val.im / mmax);
                for c in 0..nc {
                    v[c] *= phase;
                }
            }

            // Scale so the sensitivity magnitude matches the local
            // signal scale (sqrt of dominant eigenvalue).
            let scale = lambda.sqrt();
            for c in 0..nc {
                out[[c, y, x]] = v[c] * scale;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// On a 2-coil phantom with known smooth sensitivities,
    /// `walsh_sensitivity_maps` should recover maps whose magnitude
    /// ratio matches the true ratio up to a common per-voxel scale.
    #[test]
    fn walsh_recovers_sensitivity_ratio() {
        let ny = 32;
        let nx = 32;

        // True sensitivities: two gaussians on opposite sides.
        let mut s1 = Array2::<f32>::zeros((ny, nx));
        let mut s2 = Array2::<f32>::zeros((ny, nx));
        for y in 0..ny {
            for x in 0..nx {
                let yy = y as f32 - ny as f32 / 2.0;
                let x1 = x as f32 - nx as f32 / 3.0;
                let x2 = x as f32 - 2.0 * nx as f32 / 3.0;
                s1[[y, x]] = (-(yy * yy + x1 * x1) / 200.0).exp();
                s2[[y, x]] = (-(yy * yy + x2 * x2) / 200.0).exp();
            }
        }

        // Real phantom (no coil phase).
        let mut phantom = Array2::<f32>::zeros((ny, nx));
        for y in 8..24 {
            for x in 8..24 {
                phantom[[y, x]] = 1.0;
            }
        }

        // Coil images = S_c * phantom * exp(i*theta_c) for some coil phase.
        let mut coil = Array3::<Complex32>::zeros((2, ny, nx));
        let phase = [0.0f32, 0.7];
        for c in 0..2 {
            for y in 0..ny {
                for x in 0..nx {
                    let s = if c == 0 { s1[[y, x]] } else { s2[[y, x]] };
                    let m = s * phantom[[y, x]];
                    coil[[c, y, x]] = Complex32::new(m * phase[c].cos(), m * phase[c].sin());
                }
            }
        }

        let maps = walsh_sensitivity_maps(&coil, 3, 8);

        // On pixels inside the phantom, the ratio |S_1| / |S_0| from the
        // maps should closely match the true s2 / s1 ratio.
        let mut n = 0;
        let mut err = 0.0f32;
        for y in 10..22 {
            for x in 10..22 {
                let m0 = maps[[0, y, x]].norm();
                let m1 = maps[[1, y, x]].norm();
                if m0 < 1e-4 {
                    continue;
                }
                let got = m1 / m0;
                let want = s2[[y, x]] / s1[[y, x]];
                err += (got - want).abs();
                n += 1;
            }
        }
        let mean_err = err / n.max(1) as f32;
        assert!(
            mean_err < 0.1,
            "mean ratio error {} too large (expected < 0.1)",
            mean_err
        );
    }

    #[test]
    fn walsh_phase_is_locally_smooth() {
        // A single-coil constant-phase image should yield a constant
        // phase map (power-iteration sign ambiguity gets fixed by the
        // phase convention).
        let ny = 8;
        let nx = 8;
        let mut coil = Array3::<Complex32>::zeros((1, ny, nx));
        let phi = 0.35 * PI;
        for y in 0..ny {
            for x in 0..nx {
                coil[[0, y, x]] = Complex32::new(phi.cos(), phi.sin());
            }
        }
        let maps = walsh_sensitivity_maps(&coil, 1, 6);
        // After phase alignment (largest-magnitude coil -> real-positive),
        // the single-coil map should be real-positive everywhere.
        for y in 0..ny {
            for x in 0..nx {
                let v = maps[[0, y, x]];
                assert!(v.re > 0.5, "expected positive-real, got {:?}", v);
                assert!(v.im.abs() < 1e-4, "imag should be ~0, got {:?}", v);
            }
        }
    }
}
