//! GRAPPA parallel-imaging reconstruction (2-D Cartesian).
//!
//! GRAPPA (Griswold et al., 2002) synthesizes missing k-space lines from
//! an undersampled acquisition by learning a per-coil convolution kernel
//! against a fully sampled auto-calibration signal (ACS) region.
//!
//! This implementation:
//! * supports regular 1-D undersampling along ky (acceleration factor R)
//!   with a centrally located, contiguous ACS block;
//! * calibrates one kernel per missing offset `d \in 1..R` using normal
//!   equations with Tikhonov regularization;
//! * synthesizes missing ky lines per slice, using all source coils as
//!   inputs for each target coil;
//! * operates only on the `[nc, kz, ky, kx]` tensor layout produced by
//!   [`crate::ReconStrategy`]-compatible readers.
//!
//! Limitations: no ESPIRiT; kx is assumed fully sampled; non-integer
//! accelerations and irregular / CAIPIRINHA patterns are rejected.

use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView3, Axis};
use num_complex::Complex32;
use tracing::{debug, info, warn};

use crate::prewhiten::{cholesky_lower, invert_lower_triangular};

/// 1-D cartesian sampling pattern detected from a `[kz, ky, kx]` mask.
#[derive(Debug, Clone)]
pub struct SamplingPattern {
    /// Integer acceleration factor along ky.
    pub r: usize,
    /// First inclusive ky of the contiguous fully sampled ACS block.
    pub acs_start: usize,
    /// Exclusive end of the ACS block.
    pub acs_end: usize,
    /// ky range that actually carries any data (outside is all-zero).
    pub ky_lo: usize,
    pub ky_hi: usize,
}

impl SamplingPattern {
    pub fn acs_len(&self) -> usize {
        self.acs_end.saturating_sub(self.acs_start)
    }

    /// Analyze a per-slice ky mask (any-kx projection) to detect the
    /// undersampling pattern. Returns `None` if the pattern is
    /// fully sampled or irregular.
    pub fn detect(ky_any: &[bool]) -> Option<Self> {
        let _ = ky_any.len();
        let sampled: Vec<usize> = ky_any
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();
        if sampled.len() < 4 {
            return None;
        }

        let ky_lo = *sampled.first()?;
        let ky_hi = *sampled.last()?;

        // Find the longest run of consecutive sampled ky indices -- that's ACS.
        let mut best_start = 0usize;
        let mut best_len = 0usize;
        let mut cur_start = sampled[0];
        let mut cur_len = 1usize;
        for w in sampled.windows(2) {
            if w[1] == w[0] + 1 {
                cur_len += 1;
            } else {
                if cur_len > best_len {
                    best_len = cur_len;
                    best_start = cur_start;
                }
                cur_start = w[1];
                cur_len = 1;
            }
        }
        if cur_len > best_len {
            best_len = cur_len;
            best_start = cur_start;
        }
        let acs_start = best_start;
        let acs_end = best_start + best_len;

        // If the whole sampled range is the ACS, the data is fully sampled.
        if acs_start == ky_lo && acs_end == ky_hi + 1 {
            return None;
        }
        // Require ACS to be at least a minimal usable size.
        if best_len < 8 {
            debug!("GRAPPA: ACS too small ({} ky), refusing", best_len);
            return None;
        }

        // Determine R from spacing of sampled lines OUTSIDE the ACS.
        let mut outside: Vec<usize> = sampled
            .iter()
            .copied()
            .filter(|&i| i < acs_start || i >= acs_end)
            .collect();
        outside.sort_unstable();
        if outside.len() < 2 {
            return None;
        }
        let mut diffs: Vec<usize> = outside.windows(2).map(|w| w[1] - w[0]).collect();
        diffs.sort_unstable();
        let r = diffs[diffs.len() / 2];
        if r < 2 || r > 8 {
            debug!("GRAPPA: unsupported acceleration R={}", r);
            return None;
        }
        // Verify pattern is regular outside the ACS (median agrees with most).
        let agree = diffs.iter().filter(|&&d| d == r).count();
        if agree * 2 < diffs.len() {
            debug!("GRAPPA: irregular pattern, R spacing not consistent");
            return None;
        }

        Some(Self {
            r,
            acs_start,
            acs_end,
            ky_lo,
            ky_hi,
        })
    }
}

/// Calibrated GRAPPA kernel for one acceleration pattern.
///
/// For each target coil `c_t` and each missing offset `d \in 1..R`, stores
/// a complex weight vector over all source samples in the neighbourhood.
pub struct GrappaKernel {
    pub r: usize,
    pub kernel_ky: usize, // number of sampled source rows
    pub kernel_kx: usize, // number of kx taps
    pub nc: usize,
    // weights[d-1] has shape [nc_target, nc_src * kernel_ky * kernel_kx]
    pub weights: Vec<Array2<Complex32>>,
}

impl GrappaKernel {
    /// Calibrate from an ACS tensor `acs[nc, ky_acs, kx]`.
    ///
    /// `r` is the acceleration, `kernel_ky` is how many sampled rows feed
    /// each target (at spacing `r`), `kernel_kx` is the kx-window width,
    /// `ridge` is the Tikhonov regularization applied to `A^H A`.
    pub fn calibrate(
        acs: ArrayView3<Complex32>,
        r: usize,
        kernel_ky: usize,
        kernel_kx: usize,
        ridge: f32,
    ) -> Result<Self, GrappaError> {
        let (nc, ny_acs, nx_acs) = (acs.shape()[0], acs.shape()[1], acs.shape()[2]);
        if r < 2 {
            return Err(GrappaError::BadConfig("acceleration must be >= 2"));
        }
        if kernel_ky < 2 || kernel_ky % 2 != 0 {
            return Err(GrappaError::BadConfig("kernel_ky must be even >= 2"));
        }
        if kernel_kx == 0 || kernel_kx % 2 == 0 {
            return Err(GrappaError::BadConfig("kernel_kx must be odd >= 1"));
        }

        // Source rows are spaced R apart. The kernel spans `(kernel_ky-1)*R + 1`
        // ky rows in total. The missing targets sit between the central source
        // pair, at offsets d=1..R relative to source index (kernel_ky/2 - 1).
        let ky_span = (kernel_ky - 1) * r + 1;
        let kx_half = kernel_kx / 2;

        if ny_acs < ky_span + r {
            return Err(GrappaError::AcsTooSmall {
                need: ky_span + r,
                got: ny_acs,
            });
        }
        if nx_acs < kernel_kx {
            return Err(GrappaError::AcsTooSmall {
                need: kernel_kx,
                got: nx_acs,
            });
        }

        let n_src = nc * kernel_ky * kernel_kx;
        // position count: (ny_acs - ky_span) target rows (any of d=1..R fits),
        //                 (nx_acs - kernel_kx + 1) kx positions
        // ky index of the first source row for position p is p.
        // target row for offset d is p + (kernel_ky/2 - 1) * r + d  (must be < ny_acs)
        let kky_center_src_row = kernel_ky / 2 - 1; // last source before the gap
        let max_target_off = kky_center_src_row * r + (r - 1);
        let n_ky_pos = ((ny_acs as isize) - (ky_span as isize).max(0))
            .max(0)
            .min((ny_acs as isize) - (max_target_off as isize) - 1)
            .max(0) as usize;
        let n_kx_pos = nx_acs.saturating_sub(kernel_kx - 1);
        let n_pos = n_ky_pos * n_kx_pos;
        if n_pos < n_src {
            warn!(
                "GRAPPA calibration under-determined: {} positions < {} sources",
                n_pos, n_src
            );
        }
        if n_pos == 0 {
            return Err(GrappaError::AcsTooSmall {
                need: ky_span + kernel_kx,
                got: ny_acs.min(nx_acs),
            });
        }

        info!(
            "GRAPPA calibrate: nc={}, R={}, kernel={}x{}, ACS={}x{}, positions={}, sources={}",
            nc, r, kernel_ky, kernel_kx, ny_acs, nx_acs, n_pos, n_src
        );

        // Build A [n_pos, n_src] and B [n_pos, nc * (R-1)] in column-major
        // friendly layout.
        let n_tgt = nc * (r - 1);
        let mut a = Array2::<Complex32>::zeros((n_pos, n_src));
        let mut b = Array2::<Complex32>::zeros((n_pos, n_tgt));

        let mut p = 0usize;
        for ky0 in 0..n_ky_pos {
            for kx0 in 0..n_kx_pos {
                // Source window in kx: [kx0, kx0+kernel_kx)
                // Source rows in ky: ky0 + k*R for k in 0..kernel_ky
                // Target row for offset d: ky0 + kky_center_src_row*R + d
                for ch in 0..nc {
                    for kky in 0..kernel_ky {
                        let src_y = ky0 + kky * r;
                        for kkx in 0..kernel_kx {
                            let src_x = kx0 + kkx;
                            let col =
                                ch * (kernel_ky * kernel_kx) + kky * kernel_kx + kkx;
                            a[[p, col]] = acs[[ch, src_y, src_x]];
                        }
                    }
                }
                let tgt_x = kx0 + kx_half;
                for d in 1..r {
                    let tgt_y = ky0 + kky_center_src_row * r + d;
                    for ch in 0..nc {
                        let col = (d - 1) * nc + ch;
                        b[[p, col]] = acs[[ch, tgt_y, tgt_x]];
                    }
                }
                p += 1;
            }
        }
        debug_assert_eq!(p, n_pos);

        // Normal equations: solve (A^H A + ridge*I) X = A^H B
        // where X is [n_src, n_tgt]. Then weights[d-1] is X[.., d*nc..(d+1)*nc]^T
        // shaped [nc, n_src].
        let ata = hermitian_gram(&a); // [n_src, n_src]
        let atb = hermitian_mul(&a, &b); // [n_src, n_tgt]

        // Regularize diagonal
        let mut ata_reg = ata;
        let lam = {
            // Scale ridge by mean diagonal magnitude (typical practice).
            let mean_diag = (0..n_src)
                .map(|i| ata_reg[[i, i]].re)
                .sum::<f32>()
                / (n_src as f32).max(1.0);
            ridge * mean_diag.max(f32::EPSILON)
        };
        for i in 0..n_src {
            ata_reg[[i, i]] += Complex32::new(lam, 0.0);
        }

        // Cholesky solve
        let l = cholesky_lower(&ata_reg).ok_or(GrappaError::CholeskyFailed)?;
        let l_inv = invert_lower_triangular(&l).ok_or(GrappaError::CholeskyFailed)?;
        // inv(A^H A + lam I) = L^-H L^-1
        // X = L^-H (L^-1 (A^H B))
        let tmp = matmul(&l_inv, &atb); // L^-1 * AtB
        let l_inv_h = conjugate_transpose(&l_inv);
        let x = matmul(&l_inv_h, &tmp); // [n_src, n_tgt]

        let mut weights = Vec::with_capacity(r - 1);
        for d in 1..r {
            let col_start = (d - 1) * nc;
            let col_end = d * nc;
            // Take X[.., col_start..col_end] and transpose -> [nc, n_src]
            let block = x.slice(s![.., col_start..col_end]);
            let mut w = Array2::<Complex32>::zeros((nc, n_src));
            for ch in 0..nc {
                for j in 0..n_src {
                    w[[ch, j]] = block[[j, ch]];
                }
            }
            weights.push(w);
        }

        Ok(Self {
            r,
            kernel_ky,
            kernel_kx,
            nc,
            weights,
        })
    }

    /// Synthesize missing lines in `kspace[nc, kz, ky, kx]` given the
    /// detected sampling pattern. Per-slice: for each pair of sampled
    /// ky rows separated by exactly `R`, fill in the `R-1` lines between.
    ///
    /// The ACS region itself is already fully sampled and left untouched.
    pub fn synthesize(&self, kspace: &mut Array4<Complex32>, pattern: &SamplingPattern) {
        let _ = pattern; // pattern is implicit in the sampled mask (detected per slice)
        let (nc, nz, ny, nx) = (
            kspace.shape()[0],
            kspace.shape()[1],
            kspace.shape()[2],
            kspace.shape()[3],
        );
        assert_eq!(nc, self.nc, "coil count mismatch");
        let r = self.r;
        let kx_half = self.kernel_kx / 2;
        let kky_center = self.kernel_ky / 2 - 1;

        // For each slice independently.
        for kz in 0..nz {
            // List of sampled ky rows for this slice (outside ACS is the
            // undersampled region; inside ACS all rows are sampled).
            let ky_sampled: Vec<usize> = (0..ny)
                .filter(|&ky| {
                    // Check a representative kx sample in the middle of the readout.
                    kspace[[0, kz, ky, nx / 2]] != Complex32::new(0.0, 0.0)
                        || any_nonzero(kspace.slice(s![0, kz, ky, ..]))
                })
                .collect();

            let mut filled = 0usize;
            // For every group of `kernel_ky` sampled rows whose spacing is R,
            // fill in the `R-1` gap rows between the central pair.
            // We advance along sampled rows; for each missing pattern find
            // R+kernel_ky-1 consecutive rows that match the expected spacing.
            if self.kernel_ky < 2 {
                continue;
            }
            // Build a set of sampled rows for O(1) lookup.
            let mut sampled_set = vec![false; ny];
            for &k in &ky_sampled {
                sampled_set[k] = true;
            }

            // For each candidate "first source row" ky0, check that all
            // kernel_ky source rows exist at spacing R, and the missing
            // targets between the central pair do NOT exist (undersampled).
            for ky0 in 0..ny {
                let ky_span = (self.kernel_ky - 1) * r + 1;
                if ky0 + ky_span > ny {
                    break;
                }
                // All source rows must be sampled.
                let mut ok = true;
                for k in 0..self.kernel_ky {
                    if !sampled_set[ky0 + k * r] {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                // Central missing targets live at ky0 + kky_center*R + d for d in 1..R.
                // Only fill rows that are currently *not* sampled (i.e. outside ACS).
                let any_missing = (1..r).any(|d| !sampled_set[ky0 + kky_center * r + d]);
                if !any_missing {
                    continue;
                }

                // Apply kernel at every kx position [kx0..kx0+kernel_kx).
                for kx0 in 0..=(nx - self.kernel_kx) {
                    // Gather source vector of length n_src.
                    let mut src =
                        Vec::<Complex32>::with_capacity(nc * self.kernel_ky * self.kernel_kx);
                    for ch in 0..nc {
                        for kky in 0..self.kernel_ky {
                            let sy = ky0 + kky * r;
                            for kkx in 0..self.kernel_kx {
                                src.push(kspace[[ch, kz, sy, kx0 + kkx]]);
                            }
                        }
                    }
                    let src = Array1::from(src);
                    // For each missing offset d, apply weights.
                    for d in 1..r {
                        let ty = ky0 + kky_center * r + d;
                        if sampled_set[ty] {
                            continue;
                        }
                        let w = &self.weights[d - 1];
                        let tx = kx0 + kx_half;
                        for ch in 0..nc {
                            // dot product w[ch, :] * src
                            let mut acc = Complex32::new(0.0, 0.0);
                            let row = w.row(ch);
                            for (a, b) in row.iter().zip(src.iter()) {
                                acc += a * b;
                            }
                            kspace[[ch, kz, ty, tx]] = acc;
                        }
                        filled += 1;
                    }
                }
            }
            debug!("GRAPPA synth: slice {} filled {} targets", kz, filled);
        }
    }
}

fn any_nonzero<'a>(row: ndarray::ArrayView1<'a, Complex32>) -> bool {
    row.iter().any(|c| c.re != 0.0 || c.im != 0.0)
}

/// Compute `A^H A` for complex matrix `A`.
fn hermitian_gram(a: &Array2<Complex32>) -> Array2<Complex32> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut out = Array2::<Complex32>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let mut s = Complex32::new(0.0, 0.0);
            for k in 0..m {
                s += a[[k, i]].conj() * a[[k, j]];
            }
            out[[i, j]] = s;
            if i != j {
                out[[j, i]] = s.conj();
            }
        }
    }
    out
}

/// Compute `A^H B`.
fn hermitian_mul(a: &Array2<Complex32>, b: &Array2<Complex32>) -> Array2<Complex32> {
    let (m, n) = (a.nrows(), a.ncols());
    let p = b.ncols();
    debug_assert_eq!(b.nrows(), m);
    let mut out = Array2::<Complex32>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            let mut s = Complex32::new(0.0, 0.0);
            for k in 0..m {
                s += a[[k, i]].conj() * b[[k, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

fn matmul(a: &Array2<Complex32>, b: &Array2<Complex32>) -> Array2<Complex32> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    debug_assert_eq!(b.nrows(), k);
    let mut out = Array2::<Complex32>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = Complex32::new(0.0, 0.0);
            for kk in 0..k {
                s += a[[i, kk]] * b[[kk, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

fn conjugate_transpose(a: &Array2<Complex32>) -> Array2<Complex32> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut out = Array2::<Complex32>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            out[[j, i]] = a[[i, j]].conj();
        }
    }
    out
}

/// Detect a sampling pattern from a `[kz, ky, kx]` mask by projecting to
/// `[ky]` (any-true across kz and kx) and analyzing.
pub fn detect_pattern(mask: &Array3<bool>) -> Option<SamplingPattern> {
    let ny = mask.shape()[1];
    let mut ky_any = vec![false; ny];
    for ky in 0..ny {
        let slab = mask.slice(s![.., ky, ..]);
        if slab.iter().any(|&b| b) {
            ky_any[ky] = true;
        }
    }
    SamplingPattern::detect(&ky_any)
}

/// Extract the ACS region from `kspace[nc, kz, ky, kx]` for a single slice
/// into a `[nc, ky_acs, kx]` tensor.
pub fn extract_acs_slice(
    kspace: &Array4<Complex32>,
    kz: usize,
    pattern: &SamplingPattern,
) -> Array3<Complex32> {
    let (nc, _, _, nx) = (
        kspace.shape()[0],
        kspace.shape()[1],
        kspace.shape()[2],
        kspace.shape()[3],
    );
    let acs = kspace.slice(s![.., kz, pattern.acs_start..pattern.acs_end, ..]);
    let mut out = Array3::<Complex32>::zeros((nc, pattern.acs_len(), nx));
    for c in 0..nc {
        for y in 0..pattern.acs_len() {
            for x in 0..nx {
                out[[c, y, x]] = acs[[c, y, x]];
            }
        }
    }
    let _ = Axis(0); // keep import used
    out
}

/// Errors raised by GRAPPA calibration / synthesis.
#[derive(Debug, thiserror::Error)]
pub enum GrappaError {
    #[error("bad GRAPPA config: {0}")]
    BadConfig(&'static str),
    #[error("ACS region too small (need {need}, got {got})")]
    AcsTooSmall { need: usize, got: usize },
    #[error("Cholesky factorization failed (matrix not positive definite)")]
    CholeskyFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    use num_complex::Complex32;

    fn make_phantom_kspace(nc: usize, ny: usize, nx: usize) -> Array4<Complex32> {
        // A smooth pseudo-k-space with coil-dependent structure. Good enough
        // that the GRAPPA calibration problem is well-posed.
        let mut k = Array4::<Complex32>::zeros((nc, 1, ny, nx));
        for c in 0..nc {
            for y in 0..ny {
                for x in 0..nx {
                    let yy = y as f32 - ny as f32 / 2.0;
                    let xx = x as f32 - nx as f32 / 2.0;
                    let r2 = (yy * yy + xx * xx) / (ny * nx) as f32;
                    let base = (-3.0 * r2).exp();
                    let phase = (c as f32) * 0.4 + 0.02 * (yy + xx);
                    let amp = 1.0 + 0.3 * (c as f32 - nc as f32 / 2.0) * yy / ny as f32;
                    let px = base * amp;
                    k[[c, 0, y, x]] = Complex32::new(px * phase.cos(), px * phase.sin());
                }
            }
        }
        k
    }

    #[test]
    fn pattern_detect_r2_with_acs() {
        // ny=64, R=2, ACS = 16 central lines (24..40), outside every 2nd.
        // Detection returns the longest run of consecutive sampled rows,
        // which may extend by one on each side if a step_by(2) row is adjacent.
        let ny = 64;
        let mut mask = vec![false; ny];
        for ky in (0..ny).step_by(2) {
            mask[ky] = true;
        }
        for ky in 24..40 {
            mask[ky] = true;
        }
        let p = SamplingPattern::detect(&mask).expect("pattern detected");
        assert_eq!(p.r, 2);
        assert!(p.acs_start >= 23 && p.acs_start <= 24);
        assert!(p.acs_end >= 40 && p.acs_end <= 41);
    }

    #[test]
    fn pattern_detect_r3() {
        let ny = 96;
        let mut mask = vec![false; ny];
        for ky in (0..ny).step_by(3) {
            mask[ky] = true;
        }
        for ky in 40..60 {
            mask[ky] = true;
        }
        let p = SamplingPattern::detect(&mask).expect("pattern detected");
        assert_eq!(p.r, 3);
        // ACS block surrounds [40, 60); run may extend by one if step_by
        // happens to sample the neighbouring row.
        assert!(p.acs_start <= 40);
        assert!(p.acs_end >= 60);
        assert!(p.acs_end - p.acs_start <= 22);
    }

    #[test]
    fn pattern_detect_rejects_fully_sampled() {
        let mask = vec![true; 64];
        assert!(SamplingPattern::detect(&mask).is_none());
    }

    #[test]
    fn grappa_reconstructs_r2_from_fully_sampled_phantom() {
        // Build a fully-sampled phantom, calibrate on the *whole* tensor as
        // ACS (so we trivially have enough data), then undersample the
        // tensor (zero out odd ky rows outside the ACS band) and ask
        // GRAPPA to fill them back in. Compare filled rows to truth.
        let nc = 4;
        let ny = 40;
        let nx = 32;
        let truth = make_phantom_kspace(nc, ny, nx);

        // Calibrate using a central ACS block of 24 ky rows.
        let acs_start = 8;
        let acs_end = 32;
        let acs = {
            let mut a = Array3::<Complex32>::zeros((nc, acs_end - acs_start, nx));
            for c in 0..nc {
                for y in acs_start..acs_end {
                    for x in 0..nx {
                        a[[c, y - acs_start, x]] = truth[[c, 0, y, x]];
                    }
                }
            }
            a
        };
        let kernel = GrappaKernel::calibrate(acs.view(), 2, 4, 5, 1e-3)
            .expect("calibrate ok");

        // Build an undersampled copy: keep even ky rows + ACS band.
        let mut us = Array4::<Complex32>::zeros((nc, 1, ny, nx));
        for y in 0..ny {
            let keep = y % 2 == 0 || (y >= acs_start && y < acs_end);
            if keep {
                for c in 0..nc {
                    for x in 0..nx {
                        us[[c, 0, y, x]] = truth[[c, 0, y, x]];
                    }
                }
            }
        }
        let pattern = SamplingPattern {
            r: 2,
            acs_start,
            acs_end,
            ky_lo: 0,
            ky_hi: ny - 1,
        };
        kernel.synthesize(&mut us, &pattern);

        // Check filled rows match truth to within a reasonable tolerance
        // on the central kx region. Boundary rows where the kernel has no
        // valid source neighbourhood (target=1 at the top, 37/39 at the
        // bottom for this kernel size) can't be filled and are excluded.
        let kx_half = 2;
        let kernel_span = (4 - 1) * 2 + 1; // (kernel_ky-1)*R + 1 = 7
        let fill_lo = 3; // kky_center*R + d  with kky_center=1, d=1 -> first fillable
        let fill_hi = ny - (kernel_span - 3) - 1; // last fillable target
        let mut max_err: f32 = 0.0;
        let mut sum_truth: f32 = 0.0;
        let mut sum_err: f32 = 0.0;
        let mut n_checked = 0usize;
        for y in 0..ny {
            if y % 2 == 0 || (y >= acs_start && y < acs_end) {
                continue;
            }
            if y < fill_lo || y > fill_hi {
                continue; // boundary target that can't be filled
            }
            for c in 0..nc {
                for x in kx_half..(nx - kx_half) {
                    let t = truth[[c, 0, y, x]];
                    let g = us[[c, 0, y, x]];
                    let e = (t - g).norm();
                    max_err = max_err.max(e);
                    sum_err += e * e;
                    sum_truth += t.norm_sqr();
                    n_checked += 1;
                }
            }
        }
        assert!(n_checked > 0, "no fillable rows were checked");
        let nrmse = (sum_err / sum_truth.max(1e-20)).sqrt();
        assert!(
            nrmse < 0.1,
            "GRAPPA NRMSE too high: {:.4} (max |err| = {:.4e}, n={})",
            nrmse,
            max_err,
            n_checked
        );
    }
}
