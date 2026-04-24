//! Crop a reconstructed image volume to the recon FOV, removing the
//! vendor-added readout oversampling (commonly 2x along kx).

use ndarray::{s, Array3};

/// Return a view-cropped owned copy of `vol` (shape `[nz, ny, nx]`) centred
/// on its middle. Each output axis has size `target[i]` (clamped to input).
///
/// This is the standard step after `IFFT + RSS` to remove the wide FOV
/// produced by readout oversampling: the reconstructed image has the
/// encoded matrix size, but only the central `recon_matrix` region is
/// physical -- the rest is aliased padding.
pub fn center_crop_3d(vol: &Array3<f32>, target: (usize, usize, usize)) -> Array3<f32> {
    let (nz, ny, nx) = vol.dim();
    let (tz, ty, tx) = target;

    let tz = tz.min(nz).max(1);
    let ty = ty.min(ny).max(1);
    let tx = tx.min(nx).max(1);

    let z0 = (nz - tz) / 2;
    let y0 = (ny - ty) / 2;
    let x0 = (nx - tx) / 2;

    vol.slice(s![z0..z0 + tz, y0..y0 + ty, x0..x0 + tx])
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn crop_even_centered() {
        let mut v = Array3::<f32>::zeros((1, 4, 8));
        // Central 2x4 region -- mark it with 1.0.
        for y in 1..3 {
            for x in 2..6 {
                v[[0, y, x]] = 1.0;
            }
        }
        let c = center_crop_3d(&v, (1, 2, 4));
        assert_eq!(c.dim(), (1, 2, 4));
        assert!(c.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn crop_no_change_when_target_larger() {
        let v = Array3::<f32>::ones((2, 3, 5));
        let c = center_crop_3d(&v, (10, 10, 10));
        assert_eq!(c.dim(), (2, 3, 5));
    }
}
