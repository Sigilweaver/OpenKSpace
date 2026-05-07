//! Readout oversampling removal.
//!
//! MRI scanners routinely sample the readout at 2x the nominal matrix
//! resolution along kx. This is done in hardware to push aliasing past
//! the imaged FOV and is not useful for the final image. The standard
//! fix is: 1D IFFT along the readout, centre-crop the image to the
//! recon matrix, then 1D FFT back to k-space. This collapses the
//! oversampled readout to `recon_matrix.x` samples before the main
//! reconstruction runs.
//!
//! Done as a *pre-pass* over each acquisition rather than as a post-IFFT
//! crop because parallel-imaging algorithms (GRAPPA, SENSE) need clean
//! k-space at the recon matrix size as their input. For a plain IFFT+RSS
//! recon the two approaches are mathematically equivalent, but doing it
//! up front keeps the pipeline uniform.

use num_complex::Complex32;
use openkspace_io::ismrmrd::Acquisition;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
use tracing::info;

/// Per-acquisition readout oversampling remover.
///
/// Constructed from the encoded and recon matrix x-dimensions. Returns
/// `None` if no oversampling is present or the ratio is not an integer.
#[derive(Clone)]
pub struct OversamplingRemover {
    encoded_ns: usize,
    recon_ns: usize,
    ratio: usize,
    ifft_plan: Arc<dyn Fft<f32>>,
    fft_plan: Arc<dyn Fft<f32>>,
}

impl std::fmt::Debug for OversamplingRemover {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OversamplingRemover")
            .field("encoded_ns", &self.encoded_ns)
            .field("recon_ns", &self.recon_ns)
            .field("ratio", &self.ratio)
            .finish()
    }
}

impl OversamplingRemover {
    /// Build a remover for a file with `encoded_x` samples per readout
    /// collapsing to `recon_x` samples. Returns `None` if the dimensions
    /// do not describe a valid integer-ratio oversampling.
    pub fn new(encoded_x: usize, recon_x: usize) -> Option<Self> {
        if recon_x == 0 || encoded_x <= recon_x {
            return None;
        }
        if !encoded_x.is_multiple_of(recon_x) {
            return None;
        }
        let ratio = encoded_x / recon_x;
        if ratio > 8 {
            // Unreasonably large ratio; probably a misinterpretation.
            return None;
        }
        let mut planner = FftPlanner::<f32>::new();
        Some(Self {
            encoded_ns: encoded_x,
            recon_ns: recon_x,
            ratio,
            ifft_plan: planner.plan_fft_inverse(encoded_x),
            fft_plan: planner.plan_fft_forward(recon_x),
        })
    }

    /// Oversampling factor (e.g. 2 for Siemens standard readouts).
    #[inline]
    pub fn ratio(&self) -> usize {
        self.ratio
    }

    /// Number of samples per channel after the removal pass.
    #[inline]
    pub fn output_samples(&self) -> usize {
        self.recon_ns
    }

    /// Number of samples per channel expected on input.
    #[inline]
    pub fn input_samples(&self) -> usize {
        self.encoded_ns
    }

    /// Log an info-level summary once.
    pub fn log_summary(&self) {
        info!(
            "Readout oversampling removal: {} -> {} samples (ratio {}x)",
            self.encoded_ns, self.recon_ns, self.ratio
        );
    }

    /// Rewrite `acq` in place: reduce each channel's readout from
    /// `encoded_ns` to `recon_ns` samples via 1-D IFFT / crop / FFT.
    /// If the acquisition's sample count does not match `encoded_ns`
    /// the acquisition is left untouched (e.g. navigator scans of a
    /// different length).
    pub fn apply(&self, acq: &mut Acquisition) {
        let ns = acq.num_samples();
        if ns != self.encoded_ns {
            return;
        }
        let nc = acq.num_channels();

        let ifft_scratch_len = self.ifft_plan.get_inplace_scratch_len();
        let fft_scratch_len = self.fft_plan.get_inplace_scratch_len();
        let mut scratch: Vec<Complex32> =
            vec![Complex32::new(0.0, 0.0); ifft_scratch_len.max(fft_scratch_len)];

        let ifft_scale = 1.0f32 / (self.encoded_ns as f32);
        let crop_start = (self.encoded_ns - self.recon_ns) / 2;

        let mut new_data: Vec<Complex32> = Vec::with_capacity(nc * self.recon_ns);
        let mut lane_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); self.encoded_ns];
        let mut cropped: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); self.recon_ns];

        for ch in 0..nc {
            let src = acq.channel(ch);
            lane_buf.copy_from_slice(src);

            // ifftshift -> IFFT -> scale -> fftshift
            rotate_left(&mut lane_buf, self.encoded_ns.div_ceil(2));
            self.ifft_plan
                .process_with_scratch(&mut lane_buf, &mut scratch[..ifft_scratch_len]);
            for v in lane_buf.iter_mut() {
                *v *= ifft_scale;
            }
            rotate_left(&mut lane_buf, self.encoded_ns / 2);

            // Centre-crop the image to recon_ns samples.
            cropped.copy_from_slice(&lane_buf[crop_start..crop_start + self.recon_ns]);

            // ifftshift -> FFT -> fftshift back to centred k-space.
            rotate_left(&mut cropped, self.recon_ns.div_ceil(2));
            self.fft_plan
                .process_with_scratch(&mut cropped, &mut scratch[..fft_scratch_len]);
            rotate_left(&mut cropped, self.recon_ns / 2);

            new_data.extend_from_slice(&cropped);
        }

        acq.data = new_data;
        acq.header.number_of_samples = self.recon_ns as u16;
        // DC ends up at the geometric centre by construction.
        acq.header.center_sample = (self.recon_ns / 2) as u16;
        // Discard counts referred to the old readout; they no longer apply.
        acq.header.discard_pre = 0;
        acq.header.discard_post = 0;
    }
}

/// Rotate slice left by `k` positions (equivalent to `np.roll(-k)`),
/// matching the shift convention used elsewhere in this crate.
fn rotate_left(buf: &mut [Complex32], k: usize) {
    let n = buf.len();
    if n < 2 || k == 0 || k >= n {
        return;
    }
    let mut tmp: Vec<Complex32> = Vec::with_capacity(n);
    for i in 0..n {
        tmp.push(buf[(i + k) % n]);
    }
    buf.copy_from_slice(&tmp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use openkspace_io::ismrmrd::AcquisitionHeader;

    fn zeroed_header(ns: usize, nc: usize) -> AcquisitionHeader {
        AcquisitionHeader {
            version: 1,
            flags: 0,
            measurement_uid: 0,
            scan_counter: 0,
            acquisition_time_stamp: 0,
            physiology_time_stamp: [0; 3],
            number_of_samples: ns as u16,
            available_channels: nc as u16,
            active_channels: nc as u16,
            channel_mask: [0; 16],
            discard_pre: 0,
            discard_post: 0,
            center_sample: (ns / 2) as u16,
            encoding_space_ref: 0,
            trajectory_dimensions: 0,
            sample_time_us: 0.0,
            position: [0.0; 3],
            read_dir: [0.0; 3],
            phase_dir: [0.0; 3],
            slice_dir: [0.0; 3],
            patient_table_position: [0.0; 3],
            idx: openkspace_io::ismrmrd::EncodingCounters {
                kspace_encode_step_1: 0,
                kspace_encode_step_2: 0,
                average: 0,
                slice: 0,
                contrast: 0,
                phase: 0,
                repetition: 0,
                set: 0,
                segment: 0,
                user: [0; 8],
            },
            user_int: [0; 8],
            user_float: [0.0; 8],
        }
    }

    #[test]
    fn rejects_non_oversampled() {
        assert!(OversamplingRemover::new(256, 256).is_none());
        assert!(OversamplingRemover::new(256, 512).is_none());
        assert!(OversamplingRemover::new(256, 0).is_none());
    }

    #[test]
    fn rejects_non_integer_ratio() {
        assert!(OversamplingRemover::new(300, 256).is_none());
    }

    #[test]
    fn accepts_integer_ratios() {
        let r = OversamplingRemover::new(512, 256).unwrap();
        assert_eq!(r.ratio(), 2);
        assert_eq!(r.input_samples(), 512);
        assert_eq!(r.output_samples(), 256);
    }

    #[test]
    fn preserves_dc_only_signal() {
        // A DC spike at the centre of k-space represents a constant image.
        // After IFFT/crop/FFT at a smaller size we should get a DC spike
        // at the new centre with amplitude scaled by M/N.
        let encoded = 16;
        let recon = 8;
        let nc = 1;

        let header = zeroed_header(encoded, nc);
        let mut data = vec![Complex32::new(0.0, 0.0); encoded];
        data[encoded / 2] = Complex32::new(10.0, 0.0);

        let mut acq = Acquisition { header, data };
        let remover = OversamplingRemover::new(encoded, recon).unwrap();
        remover.apply(&mut acq);

        assert_eq!(acq.num_samples(), recon);

        // Output: only the centre sample should be non-zero.
        let expected = Complex32::new(10.0 * (recon as f32 / encoded as f32), 0.0);
        for i in 0..recon {
            let v = acq.data[i];
            if i == recon / 2 {
                assert!(
                    (v - expected).norm() < 1e-4,
                    "centre: got {:?}, expected {:?}",
                    v,
                    expected
                );
            } else {
                assert!(
                    v.norm() < 1e-4,
                    "off-centre {}: expected ~0, got {:?}",
                    i,
                    v
                );
            }
        }
    }

    #[test]
    fn leaves_mismatched_acquisitions_untouched() {
        // A navigator or noise scan with a different sample count should
        // pass through unchanged.
        let header = zeroed_header(64, 2);
        let data = vec![Complex32::new(1.0, 2.0); 64 * 2];
        let mut acq = Acquisition {
            header,
            data: data.clone(),
        };

        let remover = OversamplingRemover::new(512, 256).unwrap();
        remover.apply(&mut acq);

        assert_eq!(acq.num_samples(), 64);
        assert_eq!(acq.data, data);
    }
}
