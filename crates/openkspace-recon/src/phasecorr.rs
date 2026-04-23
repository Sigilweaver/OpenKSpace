//! Navigator phase correction for TSE / EPI readouts.
//!
//! ## Background
//!
//! Turbo-spin-echo and echo-planar sequences produce adjacent readouts that
//! are displaced in time (different effective echo times, different eddy-
//! current states, and in TSE typically alternating `ACQ_IS_REVERSE` lines).
//! The result is a systematic phase difference between even and odd echoes
//! that -- if untreated -- produces an N/2 Nyquist ghost and horizontal
//! amplitude striping in the reconstructed image.
//!
//! Vendors record short **navigator** echoes (ISMRMRD flag
//! `ACQ_IS_PHASECORR_DATA`) right after each RF refocusing pulse. These
//! echoes sample the same readout gradient the image lines do, but with no
//! phase-encode gradient -- i.e. they are a clean measurement of the
//! per-sample phase error for that echo. We:
//!
//! 1. Group phase-correction scans by slice and segment (echo position in
//!    the TSE train).
//! 2. Coil-combine each navigator to a single 1-D phase vector by a
//!    complex sum across channels (equivalent to a uniform-sensitivity
//!    assumption -- adequate for a linear phase estimate).
//! 3. For each segment, pick the average complex navigator and invert its
//!    phase to get a correction `c(k) = exp(-iphi(k))`.
//! 4. Multiply every image readout with the correction vector for its
//!    segment before placement in k-space.
//!
//! ## Reference
//!
//! *Bernstein MA, King KF, Zhou XJ.* **Handbook of MRI Pulse Sequences**,
//! Sec.16.4 ("Navigator echoes") & Sec.13.5 ("Phase correction"). Academic Press,
//! 2004.
//!
//! The simple per-segment, linear-phase model used here matches the
//! classical "three-line phase correction" originally described for EPI in:
//!
//! *Buonocore MH, Gao L.* "Ghost artifact reduction for echo planar imaging
//! using image phase correction." **MRM** 38(1):89-100, 1997.

use ndarray::Array1;
use num_complex::Complex32;
use openkspace_io::ismrmrd::Acquisition;
use std::collections::HashMap;
use tracing::{debug, info};

/// Per-(slice, segment) navigator correction vectors in k-space.
///
/// The key is `(slice, segment)`; the value is a length-`ns` complex vector
/// whose magnitude is 1 and whose phase is `-phi_nav(k)`.
#[derive(Debug, Clone, Default)]
pub struct PhaseCorrector {
    /// `(slice, segment) -> correction vector of length ns`
    corrections: HashMap<(u16, u16), Array1<Complex32>>,
}

impl PhaseCorrector {
    /// Build a corrector by averaging the phase-correction scans per
    /// `(slice, segment)` bin.
    ///
    /// Returns an empty corrector (no-op `apply`) if the input is empty.
    pub fn from_phasecorr_acqs(pc: &[Acquisition]) -> Self {
        if pc.is_empty() {
            return Self::default();
        }

        // Bin by (slice, segment). Each bin accumulates a coil-summed 1D
        // k-line, then is averaged and inverted.
        let mut bins: HashMap<(u16, u16), (Array1<Complex32>, u32)> = HashMap::new();

        for a in pc {
            let ns = a.num_samples();
            let nc = a.num_channels();
            if ns == 0 || nc == 0 {
                continue;
            }
            // Coil-sum: Sum_c s_c(k). Produces a per-sample complex value that
            // carries the shared echo phase while averaging out per-coil
            // sensitivity differences.
            let view = a.as_array_view(); // [nc, ns]
            let mut coil_sum = Array1::<Complex32>::zeros(ns);
            for c in 0..nc {
                let row = view.row(c);
                for k in 0..ns {
                    coil_sum[k] += row[k];
                }
            }

            let key = (a.header.idx.slice, a.header.idx.segment);
            let entry = bins
                .entry(key)
                .or_insert_with(|| (Array1::<Complex32>::zeros(ns), 0));
            if entry.0.len() != ns {
                debug!(
                    "phasecorr: dropping scan with ns={ns} (bin expects {})",
                    entry.0.len()
                );
                continue;
            }
            for k in 0..ns {
                entry.0[k] += coil_sum[k];
            }
            entry.1 += 1;
        }

        let mut corrections = HashMap::with_capacity(bins.len());
        for (key, (sum, count)) in bins {
            // Average across bin, then take unit-magnitude conjugate:
            //   c(k) = conj(avg(k)) / |avg(k)|
            let mut corr = sum;
            let inv_n = 1.0 / count as f32;
            corr.mapv_inplace(|v| {
                let avg = v * Complex32::new(inv_n, 0.0);
                let mag = avg.norm();
                if mag < 1e-20 {
                    Complex32::new(1.0, 0.0)
                } else {
                    avg.conj() / Complex32::new(mag, 0.0)
                }
            });
            corrections.insert(key, corr);
        }

        info!(
            "Phase correction: built {} navigator vectors",
            corrections.len()
        );

        Self { corrections }
    }

    /// True if there is nothing to apply (no navigator data available).
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }

    /// Apply the appropriate correction to `acq` (selected by slice/segment).
    ///
    /// If no navigator exists for this `(slice, segment)` the acquisition
    /// is left untouched. Correction is applied in k-space: each complex
    /// sample is multiplied by the corresponding phasor.
    pub fn apply(&self, acq: &mut Acquisition) {
        if self.corrections.is_empty() {
            return;
        }
        let key = (acq.header.idx.slice, acq.header.idx.segment);
        let corr = match self.corrections.get(&key) {
            Some(c) => c,
            // Fall back: some sequences tag only (segment,) ignoring slice.
            None => match self.corrections.get(&(u16::MAX, acq.header.idx.segment)) {
                Some(c) => c,
                None => return,
            },
        };

        let ns = acq.num_samples();
        let copy = ns.min(corr.len());
        if copy == 0 {
            return;
        }

        let nc = acq.num_channels();
        let mut view = acq.as_array_view_mut(); // [nc, ns]
        for c in 0..nc {
            let mut row = view.row_mut(c);
            for k in 0..copy {
                row[k] *= corr[k];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openkspace_io::ismrmrd::AcquisitionHeader;

    fn mk_acq(slice: u16, segment: u16, data: &[Complex32], nc: usize) -> Acquisition {
        let ns = data.len() / nc;
        let mut h: AcquisitionHeader = unsafe { std::mem::zeroed() };
        h.number_of_samples = ns as u16;
        h.active_channels = nc as u16;
        h.idx.slice = slice;
        h.idx.segment = segment;
        let flat: Vec<f32> = data.iter().flat_map(|c| [c.re, c.im]).collect();
        Acquisition::from_raw_f32(h, &flat)
    }

    #[test]
    fn empty_corrector_is_noop() {
        let corr = PhaseCorrector::default();
        assert!(corr.is_empty());
        let mut a = mk_acq(0, 0, &[Complex32::new(1.0, 2.0)], 1);
        corr.apply(&mut a);
        assert_eq!(a.data[0], Complex32::new(1.0, 2.0));
    }

    #[test]
    fn removes_constant_phase_offset() {
        // Navigator sees a constant phase of +pi/3 on all samples.
        // After correction an image line with the same offset should be real.
        let ns = 8;
        let nc = 1;
        let phi = std::f32::consts::PI / 3.0;
        let (c, s) = phi.sin_cos();
        let nav: Vec<Complex32> = (0..ns).map(|_| Complex32::new(s, c)).collect(); // exp(iphi)

        let nav_acq = mk_acq(0, 0, &nav, nc);
        let corr = PhaseCorrector::from_phasecorr_acqs(&[nav_acq]);
        assert!(!corr.is_empty());

        // Image line carries the same phase on a real amplitude.
        let img: Vec<Complex32> = (1..=ns)
            .map(|k| Complex32::new(s, c) * Complex32::new(k as f32, 0.0))
            .collect();
        let mut img_acq = mk_acq(0, 0, &img, nc);
        corr.apply(&mut img_acq);

        for k in 0..ns {
            let got = img_acq.data[k];
            let expected_re = (k + 1) as f32;
            assert!(
                (got.re - expected_re).abs() < 1e-4 && got.im.abs() < 1e-4,
                "k={k}: got {got:?}, expected real {}",
                expected_re
            );
        }
    }
}
