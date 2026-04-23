//! Coil combination strategies.
//!
//! For fully-sampled parallel-imaging data, root-sum-of-squares (RSS) is
//! the standard reference. It produces a real-valued magnitude image that
//! is sensitivity-weighted but free of phase information:
//!
//! $$ I(r) = \sqrt{\sum_c |I_c(r)|^2} $$

use ndarray::{Array, Array3, Axis, Dimension};
use num_complex::Complex32;

/// Root-sum-of-squares coil combination along `axis = 0`.
///
/// Input  shape: `[channels, ...]`  (complex)
/// Output shape: `[...]`            (real f32)
pub fn rss_combine<D: Dimension>(
    coil_images: &Array<Complex32, D>,
) -> Array<f32, D::Smaller>
where
    D: ndarray::RemoveAxis,
{
    let sum_sq = coil_images.map(|c| c.norm_sqr()).sum_axis(Axis(0));
    sum_sq.mapv_into(f32::sqrt)
}

/// Convenience specialization for `[C, Z, Y, X]` -> `[Z, Y, X]`.
pub fn rss_combine_4d(coil_images: &ndarray::Array4<Complex32>) -> Array3<f32> {
    let sum_sq = coil_images.map(|c| c.norm_sqr()).sum_axis(Axis(0));
    sum_sq.mapv_into(f32::sqrt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn rss_3_4_5_pythagorean() {
        // Three channels, one pixel each: 3, 4, 5 -> sqrt(9+16+25) = sqrt(50)
        let mut a: ndarray::Array3<Complex32> = ndarray::Array3::zeros((3, 1, 1));
        a[[0, 0, 0]] = Complex32::new(3.0, 0.0);
        a[[1, 0, 0]] = Complex32::new(4.0, 0.0);
        a[[2, 0, 0]] = Complex32::new(5.0, 0.0);

        let r: Array2<f32> = rss_combine(&a);
        assert!((r[[0, 0]] - 50.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn rss_respects_complex_magnitude() {
        let mut a: ndarray::Array3<Complex32> = ndarray::Array3::zeros((2, 1, 1));
        a[[0, 0, 0]] = Complex32::new(3.0, 4.0); // |.| = 5
        a[[1, 0, 0]] = Complex32::new(0.0, 12.0); // |.| = 12
        let r: Array2<f32> = rss_combine(&a);
        assert!((r[[0, 0]] - 13.0).abs() < 1e-6);
    }
}
