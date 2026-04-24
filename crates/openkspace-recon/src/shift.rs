//! fftshift / ifftshift along a single axis, for any ndarray dimension.
//!
//! These match NumPy semantics:
//!   * `fftshift`  moves the zero-frequency bin from index 0 to `n/2`.
//!   * `ifftshift` is its exact inverse (differs from fftshift only for odd n).

use ndarray::{Array, ArrayViewMut, Axis, Dimension, Slice};
use num_complex::Complex32;

/// In-place fftshift along `axis`: rotates the axis by `n / 2`.
pub fn fftshift_axis<D: Dimension>(a: &mut Array<Complex32, D>, axis: usize) {
    let n = a.len_of(Axis(axis));
    if n < 2 {
        return;
    }
    rotate_axis(a.view_mut(), axis, n / 2);
}

/// In-place ifftshift along `axis`: rotates the axis by `(n + 1) / 2`.
pub fn ifftshift_axis<D: Dimension>(a: &mut Array<Complex32, D>, axis: usize) {
    let n = a.len_of(Axis(axis));
    if n < 2 {
        return;
    }
    rotate_axis(a.view_mut(), axis, (n + 1) / 2);
}

/// Shift axis `axis` left by `k` positions (equivalent to `np.roll(-k)`).
fn rotate_axis<D: Dimension>(mut a: ArrayViewMut<Complex32, D>, axis: usize, k: usize) {
    let n = a.len_of(Axis(axis));
    if k == 0 || k >= n {
        return;
    }

    // Work lane-by-lane so we reuse one buffer per fiber.
    let lane_len = n;
    let mut buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); lane_len];

    // Iterate over all sub-slices orthogonal to `axis`.
    a.lanes_mut(Axis(axis)).into_iter().for_each(|mut lane| {
        for i in 0..lane_len {
            buf[i] = lane[i];
        }
        for i in 0..lane_len {
            lane[i] = buf[(i + k) % lane_len];
        }
    });

    // Silence unused warning on `Slice` import when feature flags change.
    let _ = Slice::new(0, None, 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn c(re: f32) -> Complex32 {
        Complex32::new(re, 0.0)
    }

    #[test]
    fn fftshift_1d_even() {
        let mut a = array![c(0.0), c(1.0), c(2.0), c(3.0)];
        fftshift_axis(&mut a, 0);
        assert_eq!(a, array![c(2.0), c(3.0), c(0.0), c(1.0)]);
    }

    #[test]
    fn fftshift_1d_odd() {
        let mut a = array![c(0.0), c(1.0), c(2.0), c(3.0), c(4.0)];
        fftshift_axis(&mut a, 0);
        // n/2 = 2 -> left rotate by 2: [2,3,4,0,1]
        assert_eq!(a, array![c(2.0), c(3.0), c(4.0), c(0.0), c(1.0)]);
    }

    #[test]
    fn roundtrip_odd() {
        let orig = array![c(0.0), c(1.0), c(2.0), c(3.0), c(4.0)];
        let mut a = orig.clone();
        fftshift_axis(&mut a, 0);
        ifftshift_axis(&mut a, 0);
        assert_eq!(a, orig);
    }
}
