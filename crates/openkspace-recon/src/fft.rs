//! Multi-dimensional IFFTs on ndarray tensors, using rustfft.
//!
//! We apply a 1-D inverse FFT along each target axis in sequence. rustfft
//! returns an *un-normalized* inverse; the conventional `1/N` scale per axis
//! is applied so the final image has the same amplitude convention as NumPy's
//! `np.fft.ifftn`.

use crate::shift::{fftshift_axis, ifftshift_axis};
use ndarray::{Array, ArrayViewMut, Axis, Dimension};
use num_complex::Complex32;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

/// Centred 2D IFFT: performs  ifftshift -> IFFT(axes 2,3) -> fftshift.
///
/// Operates in place on a tensor of shape `[.., .., H, W]`. For a typical
/// k-space layout `[channels, slices, ky, kx]`, this reconstructs each
/// (channel, slice) slab.
pub fn ifft2_inplace<D: Dimension>(a: &mut Array<Complex32, D>, axes: (usize, usize)) {
    let (a1, a2) = axes;
    ifftshift_axis(a, a1);
    ifftshift_axis(a, a2);

    ifft_axis(a.view_mut(), a1);
    ifft_axis(a.view_mut(), a2);

    fftshift_axis(a, a1);
    fftshift_axis(a, a2);
}

/// Centred 3D IFFT along the three given axes.
pub fn ifft3_inplace<D: Dimension>(a: &mut Array<Complex32, D>, axes: (usize, usize, usize)) {
    let (a1, a2, a3) = axes;
    ifftshift_axis(a, a1);
    ifftshift_axis(a, a2);
    ifftshift_axis(a, a3);

    ifft_axis(a.view_mut(), a1);
    ifft_axis(a.view_mut(), a2);
    ifft_axis(a.view_mut(), a3);

    fftshift_axis(a, a1);
    fftshift_axis(a, a2);
    fftshift_axis(a, a3);
}

/// 1-D inverse FFT along one axis, normalized by `1/n`.
fn ifft_axis<D: Dimension>(mut a: ArrayViewMut<Complex32, D>, axis: usize) {
    let n = a.len_of(Axis(axis));
    if n < 2 {
        return;
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft: Arc<dyn rustfft::Fft<f32>> = planner.plan_fft_inverse(n);

    // rustfft operates on `Complex<f32>`, which is an alias of `num_complex::Complex<f32>`.
    // Our data is `Complex32` which is the same type -- safe to reuse buffers.
    let scratch_len = fft.get_inplace_scratch_len();
    let mut scratch: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); scratch_len];
    let mut lane_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n];
    let norm = 1.0f32 / (n as f32);

    a.lanes_mut(Axis(axis)).into_iter().for_each(|mut lane| {
        for i in 0..n {
            lane_buf[i] = lane[i];
        }
        fft.process_with_scratch(&mut lane_buf, &mut scratch);
        for i in 0..n {
            lane[i] = lane_buf[i] * norm;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn ifft2_impulse_gives_constant() {
        // A centred delta in k-space -> constant-magnitude image after
        // centred IFFT. (Ifftshift moves the centre to index 0 where the
        // un-centred IFFT expects DC.)
        let n = 8;
        let mut k: Array2<Complex32> = Array2::zeros((n, n));
        k[[n / 2, n / 2]] = Complex32::new(1.0, 0.0);

        ifft2_inplace(&mut k, (0, 1));

        let expected = 1.0 / (n as f32 * n as f32);
        for v in k.iter() {
            assert!(
                (v.norm() - expected).abs() < 1e-6,
                "expected |{}| ~= {}, got {}",
                v,
                expected,
                v.norm()
            );
        }
    }

    #[test]
    fn ifft2_roundtrip() {
        // IFFT of a known spatial pattern -- we only check that amplitudes
        // are preserved within rustfft's f32 tolerance.
        let n = 16;
        let mut k: Array2<Complex32> = Array2::zeros((n, n));
        k[[0, 0]] = Complex32::new(1.0, 0.0);
        k[[1, 2]] = Complex32::new(0.5, -0.25);

        let before_sum: f32 = k.iter().map(|c| c.norm_sqr()).sum();

        // Parseval (with centred IFFT & 1/N norm):  Sum|x|^2 = (1/N) Sum|X|^2
        ifft2_inplace(&mut k, (0, 1));
        let after_sum: f32 = k.iter().map(|c| c.norm_sqr()).sum();

        let n2 = (n * n) as f32;
        let expected = before_sum / n2;
        assert!(
            (after_sum - expected).abs() < 1e-5,
            "Parseval mismatch: before={before_sum}, after={after_sum}, expected={expected}"
        );
    }

    #[test]
    fn ifft3_impulse_gives_constant() {
        // A centred delta in 3D k-space -> constant-magnitude image after a
        // centred 3D IFFT (identical to the 2D case, but across three axes).
        use ndarray::Array3;
        let (nz, ny, nx) = (4, 8, 8);
        let mut k: Array3<Complex32> = Array3::zeros((nz, ny, nx));
        k[[nz / 2, ny / 2, nx / 2]] = Complex32::new(1.0, 0.0);

        ifft3_inplace(&mut k, (0, 1, 2));

        let expected = 1.0 / (nz as f32 * ny as f32 * nx as f32);
        for v in k.iter() {
            assert!(
                (v.norm() - expected).abs() < 1e-6,
                "expected |{}| ~= {}, got {}",
                v,
                expected,
                v.norm()
            );
        }
    }
}
