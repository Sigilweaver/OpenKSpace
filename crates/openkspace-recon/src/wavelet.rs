//! Single-level 2-D Haar wavelet transform (complex-valued).
//!
//! Used by the compressed-sensing strategy as a sparsifying transform.
//! Only one decomposition level is implemented on purpose -- it captures
//! most of the sparsity benefit on typical MR images without pulling in
//! a heavyweight wavelet library. Both the forward and inverse transforms
//! are orthonormal (Parseval-preserving) when the spatial dimensions are
//! even.
//!
//! A 2×2 block `[[a, b], [c, d]]` is mapped to four sub-bands (LL, LH,
//! HL, HH) each of half the original side length, stacked into the same
//! output array of shape `[Ny, Nx]` in the standard layout:
//!
//! ```text
//!   +-------+-------+
//!   |  LL   |  HL   |
//!   +-------+-------+
//!   |  LH   |  HH   |
//!   +-------+-------+
//! ```
//!
//! (No code copied from any external wavelet library.)

use ndarray::{Array2, ArrayView2};
use num_complex::Complex32;

const HAAR_NORM: f32 = 0.5;

/// Forward 1-level Haar transform.
///
/// Panics if any spatial dimension is odd.
pub fn haar_forward(x: ArrayView2<Complex32>) -> Array2<Complex32> {
    let (ny, nx) = x.dim();
    assert!(ny % 2 == 0 && nx % 2 == 0, "Haar: dims must be even");
    let hy = ny / 2;
    let hx = nx / 2;
    let mut out = Array2::<Complex32>::zeros((ny, nx));
    let c = Complex32::new(HAAR_NORM, 0.0);
    for i in 0..hy {
        for j in 0..hx {
            let a = x[[2 * i, 2 * j]];
            let b = x[[2 * i, 2 * j + 1]];
            let d = x[[2 * i + 1, 2 * j]];
            let e = x[[2 * i + 1, 2 * j + 1]];
            out[[i, j]] = (a + b + d + e) * c; // LL
            out[[i, hx + j]] = (a - b + d - e) * c; // HL
            out[[hy + i, j]] = (a + b - d - e) * c; // LH
            out[[hy + i, hx + j]] = (a - b - d + e) * c; // HH
        }
    }
    out
}

/// Inverse 1-level Haar transform (adjoint == inverse for orthonormal Haar).
pub fn haar_inverse(y: ArrayView2<Complex32>) -> Array2<Complex32> {
    let (ny, nx) = y.dim();
    assert!(ny % 2 == 0 && nx % 2 == 0, "Haar: dims must be even");
    let hy = ny / 2;
    let hx = nx / 2;
    let mut out = Array2::<Complex32>::zeros((ny, nx));
    let c = Complex32::new(HAAR_NORM, 0.0);
    for i in 0..hy {
        for j in 0..hx {
            let ll = y[[i, j]];
            let hl = y[[i, hx + j]];
            let lh = y[[hy + i, j]];
            let hh = y[[hy + i, hx + j]];
            out[[2 * i, 2 * j]] = (ll + hl + lh + hh) * c;
            out[[2 * i, 2 * j + 1]] = (ll - hl + lh - hh) * c;
            out[[2 * i + 1, 2 * j]] = (ll + hl - lh - hh) * c;
            out[[2 * i + 1, 2 * j + 1]] = (ll - hl - lh + hh) * c;
        }
    }
    out
}

/// Complex soft-thresholding applied to detail (non-LL) sub-bands only.
/// The LL sub-band (top-left quadrant) is left untouched, which avoids
/// removing the DC / low-frequency content that every MR image relies on.
pub fn soft_threshold_details(y: &mut Array2<Complex32>, lambda: f32) {
    let (ny, nx) = y.dim();
    let hy = ny / 2;
    let hx = nx / 2;
    for i in 0..ny {
        for j in 0..nx {
            if i < hy && j < hx {
                continue; // keep LL
            }
            let z = y[[i, j]];
            let m = z.norm();
            if m <= lambda {
                y[[i, j]] = Complex32::new(0.0, 0.0);
            } else {
                let s = (m - lambda) / m;
                y[[i, j]] = z * Complex32::new(s, 0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haar_roundtrip_recovers_input() {
        let mut x = Array2::<Complex32>::zeros((8, 8));
        for i in 0..8 {
            for j in 0..8 {
                x[[i, j]] = Complex32::new(i as f32, j as f32 * 0.3);
            }
        }
        let y = haar_forward(x.view());
        let z = haar_inverse(y.view());
        for i in 0..8 {
            for j in 0..8 {
                let e = (z[[i, j]] - x[[i, j]]).norm();
                assert!(e < 1e-5, "roundtrip err {} at ({},{})", e, i, j);
            }
        }
    }

    #[test]
    fn haar_preserves_energy() {
        let mut x = Array2::<Complex32>::zeros((4, 4));
        x[[0, 0]] = Complex32::new(1.0, 0.0);
        x[[1, 2]] = Complex32::new(0.0, 1.0);
        x[[3, 3]] = Complex32::new(0.5, -0.5);
        let e_in: f32 = x.iter().map(|c| c.norm_sqr()).sum();
        let y = haar_forward(x.view());
        let e_out: f32 = y.iter().map(|c| c.norm_sqr()).sum();
        assert!((e_in - e_out).abs() < 1e-5);
    }
}
