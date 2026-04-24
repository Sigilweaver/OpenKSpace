//! ESPIRiT coil sensitivity estimation.
//!
//! ESPIRiT (Uecker, Lai, Murphy, Virtue, Elad, Pauly, Vasanawala, Lustig,
//! "ESPIRiT -- An Eigenvalue Approach to Autocalibrating Parallel MRI",
//! *MRM* 71(3), 2014) estimates per-voxel complex coil sensitivity maps
//! from an auto-calibration signal (ACS) region entirely in k-space.
//!
//! The recipe implemented here (no code copied from any other project):
//!
//! 1. Build the calibration matrix `A` by sliding a `kw × kw × Nc`
//!    window across the fully-sampled ACS block. Each window position
//!    contributes one row of length `Nc · kw · kw`.
//! 2. Compute the right singular vectors of `A` as eigenvectors of the
//!    Hermitian matrix `A^H A` (small, `N × N` with `N = Nc·kw·kw`)
//!    using power iteration with Hotelling deflation.
//! 3. Retain singular vectors whose singular value is above
//!    `threshold · σ_max`. Reshape each into a `(Nc, kw, kw)` k-space
//!    kernel and zero-pad it in k-space to the full image size.
//! 4. Inverse-FFT each kernel (centred) to obtain
//!    `M[k, c, y, x]`, then form, at every voxel, the Hermitian
//!    `Nc × Nc` Gram matrix
//!    `G[c1, c2, y, x] = Σ_k M[k, c1, y, x] · conj(M[k, c2, y, x])`.
//! 5. The dominant eigenvector of `G[:, :, y, x]` is the sensitivity
//!    estimate at that voxel, scaled by the square root of its
//!    dominant eigenvalue; the global per-voxel phase is fixed by
//!    rotating the largest-magnitude coil to real-positive.
//!
//! The resulting maps satisfy the usual ESPIRiT eigenvalue screening:
//! voxels whose dominant eigenvalue falls well below 1 are (nearly)
//! outside the support of any coil and end up with magnitude near zero.

use ndarray::{Array2, Array3, Array4};
use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::shift::{fftshift_axis, ifftshift_axis};

/// Compute ESPIRiT coil sensitivity maps from a contiguous ACS block.
///
/// Inputs:
/// * `acs`: shape `[Nc, Kacs_y, Kacs_x]` -- the fully-sampled ACS
///   region of k-space.
/// * `image_shape`: `(Ny, Nx)` -- target output image size.
/// * `kernel_size`: kernel window size in k-space (odd, e.g. 5).
/// * `threshold`: fraction of the maximum calibration singular value
///   below which kernels are discarded (typical: `0.02`).
/// * `power_iters`: power-iteration steps used both for the
///   calibration SVD and the per-voxel eigenproblem.
///
/// Output: `[Nc, Ny, Nx]` complex sensitivity map.
pub fn espirit_sensitivity_maps(
    acs: &Array3<Complex32>,
    image_shape: (usize, usize),
    kernel_size: usize,
    threshold: f32,
    power_iters: usize,
) -> Array3<Complex32> {
    let (nc, kacs_y, kacs_x) = acs.dim();
    let (ny, nx) = image_shape;
    assert!(
        kernel_size <= kacs_y && kernel_size <= kacs_x,
        "ESPIRiT: kernel ({}) larger than ACS ({}x{})",
        kernel_size,
        kacs_y,
        kacs_x
    );
    assert!(
        kernel_size <= ny && kernel_size <= nx,
        "ESPIRiT: kernel ({}) larger than image ({}x{})",
        kernel_size,
        ny,
        nx
    );
    assert!(nc > 0 && ny > 0 && nx > 0);

    let kw = kernel_size;
    let n_per = nc * kw * kw;

    // 1. Calibration matrix: one row per sliding window position.
    let wy = kacs_y - kw + 1;
    let wx = kacs_x - kw + 1;
    let n_rows = wy * wx;
    let mut a = Array2::<Complex32>::zeros((n_rows, n_per));
    for iy in 0..wy {
        for ix in 0..wx {
            let row = iy * wx + ix;
            for c in 0..nc {
                for dy in 0..kw {
                    for dx in 0..kw {
                        let col = c * kw * kw + dy * kw + dx;
                        a[[row, col]] = acs[[c, iy + dy, ix + dx]];
                    }
                }
            }
        }
    }

    // 2. A^H A (N x N Hermitian).
    let mut aha = Array2::<Complex32>::zeros((n_per, n_per));
    for i in 0..n_per {
        for j in i..n_per {
            let mut acc = Complex32::new(0.0, 0.0);
            for r in 0..n_rows {
                acc += a[[r, i]].conj() * a[[r, j]];
            }
            aha[[i, j]] = acc;
            if i != j {
                aha[[j, i]] = acc.conj();
            }
        }
    }

    // 3. Top eigenpairs of A^H A via power iteration with deflation.
    //    Keep pairs whose sqrt(eigval) >= threshold * sigma_max.
    let max_kept = n_per.min(256); // hard cap to keep cost bounded
    let eigs = top_hermitian_eigpairs(&aha, max_kept, power_iters);
    let sigma_max = eigs.first().map(|(l, _)| l.max(0.0).sqrt()).unwrap_or(0.0);
    let cutoff = (threshold * sigma_max).max(0.0);
    let mut kernels: Vec<Vec<Complex32>> = Vec::new();
    for (lam, v) in &eigs {
        let sigma = lam.max(0.0).sqrt();
        if sigma < cutoff {
            break;
        }
        kernels.push(v.clone());
    }
    if kernels.is_empty() {
        return Array3::<Complex32>::zeros((nc, ny, nx));
    }

    // 4. For each kernel, zero-pad in centred k-space to (Ny, Nx) and
    //    inverse-FFT per coil. Scale so the resulting images have the
    //    same magnitude as the coil sensitivity rather than the image
    //    signal (the `sqrt(Ny*Nx)/kw` convention absorbs the normal-
    //    ization difference between the calibration block and the full
    //    image).
    let mut planner = FftPlanner::<f32>::new();
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);
    let scale = (ny as f32 * nx as f32).sqrt() / (kw as f32 * kw as f32);

    let mut kernel_imgs = Array4::<Complex32>::zeros((kernels.len(), nc, ny, nx));
    for (k, ker) in kernels.iter().enumerate() {
        for c in 0..nc {
            let mut pad = Array2::<Complex32>::zeros((ny, nx));
            let y0 = (ny - kw) / 2;
            let x0 = (nx - kw) / 2;
            for dy in 0..kw {
                for dx in 0..kw {
                    let col = c * kw * kw + dy * kw + dx;
                    // Conjugate the kernel coefficients: the ESPIRiT
                    // convention places the adjoint convolution in
                    // image space, so IFFT of the conjugate kernel
                    // yields the correct per-coil operator.
                    pad[[y0 + dy, x0 + dx]] = ker[col].conj();
                }
            }
            // Centred IFFT2.
            ifftshift_axis(&mut pad, 0);
            ifftshift_axis(&mut pad, 1);
            // IFFT along rows.
            let mut buf = vec![Complex32::new(0.0, 0.0); nx];
            for yy in 0..ny {
                for xx in 0..nx {
                    buf[xx] = pad[[yy, xx]];
                }
                ifft_x.process(&mut buf);
                for xx in 0..nx {
                    pad[[yy, xx]] = buf[xx];
                }
            }
            // IFFT along cols.
            let mut bufy = vec![Complex32::new(0.0, 0.0); ny];
            for xx in 0..nx {
                for yy in 0..ny {
                    bufy[yy] = pad[[yy, xx]];
                }
                ifft_y.process(&mut bufy);
                for yy in 0..ny {
                    pad[[yy, xx]] = bufy[yy];
                }
            }
            fftshift_axis(&mut pad, 0);
            fftshift_axis(&mut pad, 1);
            let inv_n = 1.0 / (ny as f32 * nx as f32);
            for yy in 0..ny {
                for xx in 0..nx {
                    kernel_imgs[[k, c, yy, xx]] =
                        pad[[yy, xx]] * Complex32::new(inv_n * scale, 0.0);
                }
            }
        }
    }

    // 5. Per-voxel eigenproblem on G = sum_k m_k m_k^H (Nc x Nc).
    let k_keep = kernels.len();
    let mut maps = Array3::<Complex32>::zeros((nc, ny, nx));
    for y in 0..ny {
        for x in 0..nx {
            // Build G (nc x nc) Hermitian.
            let mut g = Array2::<Complex32>::zeros((nc, nc));
            for c1 in 0..nc {
                for c2 in 0..nc {
                    let mut acc = Complex32::new(0.0, 0.0);
                    for k in 0..k_keep {
                        acc += kernel_imgs[[k, c1, y, x]] * kernel_imgs[[k, c2, y, x]].conj();
                    }
                    g[[c1, c2]] = acc;
                }
            }

            // Dominant eigvec via power iteration; init with the
            // first kernel-image at this voxel.
            let mut v = vec![Complex32::new(0.0, 0.0); nc];
            let mut nrm0 = 0.0f32;
            for c in 0..nc {
                v[c] = kernel_imgs[[0, c, y, x]];
                nrm0 += v[c].norm_sqr();
            }
            if nrm0 < 1e-30 {
                v[0] = Complex32::new(1.0, 0.0);
                nrm0 = 1.0;
            }
            let inv0 = 1.0 / nrm0.sqrt();
            for c in 0..nc {
                v[c] = v[c] * Complex32::new(inv0, 0.0);
            }

            let mut lambda = 0.0f32;
            for _ in 0..power_iters {
                let mut w = vec![Complex32::new(0.0, 0.0); nc];
                for c1 in 0..nc {
                    let mut acc = Complex32::new(0.0, 0.0);
                    for c2 in 0..nc {
                        acc += g[[c1, c2]] * v[c2];
                    }
                    w[c1] = acc;
                }
                let mut nrm = 0.0f32;
                for c in 0..nc {
                    nrm += w[c].norm_sqr();
                }
                nrm = nrm.sqrt();
                if nrm < 1e-30 {
                    break;
                }
                let inv = 1.0 / nrm;
                for c in 0..nc {
                    v[c] = w[c] * Complex32::new(inv, 0.0);
                }
                lambda = nrm;
            }

            // Phase convention: rotate largest-magnitude coil to
            // real-positive.
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
                let r = v[imax];
                let phase = Complex32::new(r.re / mmax, -r.im / mmax);
                for c in 0..nc {
                    v[c] = v[c] * phase;
                }
            }

            let s = lambda.sqrt();
            for c in 0..nc {
                maps[[c, y, x]] = v[c] * Complex32::new(s, 0.0);
            }
        }
    }

    maps
}

/// Compute the `num` largest eigenpairs of a Hermitian complex matrix
/// via power iteration with Hotelling deflation. Returns pairs sorted
/// by eigenvalue, largest first.
///
/// This routine is deliberately simple and is only used on the small
/// `N × N` calibration matrix `A^H A` (with `N = Nc · kw · kw`).
fn top_hermitian_eigpairs(
    a: &Array2<Complex32>,
    num: usize,
    iters: usize,
) -> Vec<(f32, Vec<Complex32>)> {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    let mut m = a.clone();
    let mut out = Vec::with_capacity(num);

    for k in 0..num.min(n) {
        // Deterministic seed vector with a rotation so successive
        // iterations don't all land on the same Krylov subspace.
        let mut v = vec![Complex32::new(0.0, 0.0); n];
        for i in 0..n {
            let theta = 0.13 + 0.37 * (i as f32) + 0.91 * (k as f32);
            v[i] = Complex32::new(theta.cos(), theta.sin());
        }
        // Normalise.
        let nrm0 = v.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt();
        if nrm0 < 1e-30 {
            break;
        }
        for x in &mut v {
            *x = *x * Complex32::new(1.0 / nrm0, 0.0);
        }

        let mut lambda = 0.0f32;
        let mut prev_lambda = 0.0f32;
        for it in 0..iters {
            let mut w = vec![Complex32::new(0.0, 0.0); n];
            for i in 0..n {
                let mut acc = Complex32::new(0.0, 0.0);
                for j in 0..n {
                    acc += m[[i, j]] * v[j];
                }
                w[i] = acc;
            }
            // Rayleigh quotient.
            let mut rq = 0.0f32;
            for i in 0..n {
                rq += (v[i].conj() * w[i]).re;
            }
            let nrm = w.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt();
            if nrm < 1e-30 {
                lambda = 0.0;
                break;
            }
            for i in 0..n {
                v[i] = w[i] * Complex32::new(1.0 / nrm, 0.0);
            }
            lambda = rq;
            if it > 2 && (lambda - prev_lambda).abs() < 1e-6 * lambda.abs().max(1e-6) {
                break;
            }
            prev_lambda = lambda;
        }
        if lambda <= 0.0 {
            break;
        }
        // Deflate: M -= lambda * v v^H
        for i in 0..n {
            for j in 0..n {
                m[[i, j]] -= Complex32::new(lambda, 0.0) * v[i] * v[j].conj();
            }
        }
        out.push((lambda, v));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::FftPlanner;

    #[test]
    fn top_eigpairs_recovers_diagonal() {
        // Diagonal matrix with known eigenvalues.
        let mut m = Array2::<Complex32>::zeros((4, 4));
        m[[0, 0]] = Complex32::new(3.0, 0.0);
        m[[1, 1]] = Complex32::new(1.0, 0.0);
        m[[2, 2]] = Complex32::new(5.0, 0.0);
        m[[3, 3]] = Complex32::new(2.0, 0.0);
        let pairs = top_hermitian_eigpairs(&m, 3, 50);
        assert_eq!(pairs.len(), 3);
        assert!((pairs[0].0 - 5.0).abs() < 1e-3, "got {}", pairs[0].0);
        assert!((pairs[1].0 - 3.0).abs() < 1e-3, "got {}", pairs[1].0);
        assert!((pairs[2].0 - 2.0).abs() < 1e-3, "got {}", pairs[2].0);
    }

    /// Build a synthetic 2-coil image + sensitivities, FFT to get
    /// full k-space, extract a centred ACS block, feed ESPIRiT, and
    /// check that the recovered sensitivity-magnitude ratio matches
    /// the truth inside the phantom support.
    #[test]
    fn espirit_recovers_sensitivity_ratio() {
        let nc = 2usize;
        let ny = 32usize;
        let nx = 32usize;

        // True sensitivities: two gaussians with some overlap.
        let mut s1 = Array2::<f32>::zeros((ny, nx));
        let mut s2 = Array2::<f32>::zeros((ny, nx));
        for y in 0..ny {
            for x in 0..nx {
                let yy = y as f32 - ny as f32 / 2.0;
                let x1 = x as f32 - nx as f32 / 3.0;
                let x2 = x as f32 - 2.0 * nx as f32 / 3.0;
                s1[[y, x]] = (-(yy * yy + x1 * x1) / 250.0).exp();
                s2[[y, x]] = (-(yy * yy + x2 * x2) / 250.0).exp();
            }
        }

        // Uniform phantom.
        let mut phantom = Array2::<f32>::zeros((ny, nx));
        for y in 6..26 {
            for x in 6..26 {
                phantom[[y, x]] = 1.0;
            }
        }

        // Coil images.
        let mut coil = Array3::<Complex32>::zeros((nc, ny, nx));
        for c in 0..nc {
            for y in 0..ny {
                for x in 0..nx {
                    let s = if c == 0 { s1[[y, x]] } else { s2[[y, x]] };
                    coil[[c, y, x]] = Complex32::new(s * phantom[[y, x]], 0.0);
                }
            }
        }

        // Centred forward FFT per coil to get k-space.
        let mut planner = FftPlanner::<f32>::new();
        let fft_x = planner.plan_fft_forward(nx);
        let fft_y = planner.plan_fft_forward(ny);
        let mut k = Array3::<Complex32>::zeros((nc, ny, nx));
        for c in 0..nc {
            let mut plane = Array2::<Complex32>::zeros((ny, nx));
            for y in 0..ny {
                for x in 0..nx {
                    plane[[y, x]] = coil[[c, y, x]];
                }
            }
            ifftshift_axis(&mut plane, 0);
            ifftshift_axis(&mut plane, 1);
            let mut buf = vec![Complex32::new(0.0, 0.0); nx];
            for y in 0..ny {
                for x in 0..nx {
                    buf[x] = plane[[y, x]];
                }
                fft_x.process(&mut buf);
                for x in 0..nx {
                    plane[[y, x]] = buf[x];
                }
            }
            let mut bufy = vec![Complex32::new(0.0, 0.0); ny];
            for x in 0..nx {
                for y in 0..ny {
                    bufy[y] = plane[[y, x]];
                }
                fft_y.process(&mut bufy);
                for y in 0..ny {
                    plane[[y, x]] = bufy[y];
                }
            }
            fftshift_axis(&mut plane, 0);
            fftshift_axis(&mut plane, 1);
            for y in 0..ny {
                for x in 0..nx {
                    k[[c, y, x]] = plane[[y, x]];
                }
            }
        }

        // Extract centred 16x16 ACS block.
        let kacs = 16;
        let y0 = (ny - kacs) / 2;
        let x0 = (nx - kacs) / 2;
        let mut acs = Array3::<Complex32>::zeros((nc, kacs, kacs));
        for c in 0..nc {
            for iy in 0..kacs {
                for ix in 0..kacs {
                    acs[[c, iy, ix]] = k[[c, y0 + iy, x0 + ix]];
                }
            }
        }

        let maps = espirit_sensitivity_maps(&acs, (ny, nx), 5, 0.02, 40);

        // Compare magnitude ratios inside the phantom support.
        let mut n = 0;
        let mut err = 0.0f32;
        for y in 8..24 {
            for x in 8..24 {
                let m0 = maps[[0, y, x]].norm();
                let m1 = maps[[1, y, x]].norm();
                if m0 < 1e-4 {
                    continue;
                }
                let got = m1 / m0;
                let want = s2[[y, x]] / s1[[y, x]];
                err += (got - want).abs() / want.max(1e-6);
                n += 1;
            }
        }
        let mean_rel_err = err / n.max(1) as f32;
        assert!(
            mean_rel_err < 0.05,
            "ESPIRiT mean rel-ratio error {} too large",
            mean_rel_err
        );
    }
}
