//! ISMRMRD acquisition record types.
//!
//! Mirrors the on-disk compound type declared by `ismrmrdlib`. We read the
//! `head` portion via hdf5's compound reading. The variable-length `data`
//! array is read per-row as a `Vec<f32>` of interleaved real/imag samples.

use hdf5_metno::H5Type;
use num_complex::Complex32;

// ---------------------------------------------------------------------------
// EncodingCounters -- packed indices within a scan
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, H5Type)]
pub struct EncodingCounters {
    pub kspace_encode_step_1: u16, // phase-encode  (ky)
    pub kspace_encode_step_2: u16, // 3D partition  (kz)
    pub average: u16,
    pub slice: u16,
    pub contrast: u16,
    pub phase: u16,
    pub repetition: u16,
    pub set: u16,
    pub segment: u16,
    pub user: [u16; 8],
}

// ---------------------------------------------------------------------------
// AcquisitionHeader -- 340-byte fixed part of each record
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, H5Type)]
pub struct AcquisitionHeader {
    pub version: u16,
    pub flags: u64,
    pub measurement_uid: u32,
    pub scan_counter: u32,
    pub acquisition_time_stamp: u32,
    pub physiology_time_stamp: [u32; 3],

    pub number_of_samples: u16, // readout length (kx)
    pub available_channels: u16,
    pub active_channels: u16,
    pub channel_mask: [u64; 16],

    pub discard_pre: u16,
    pub discard_post: u16,
    pub center_sample: u16,
    pub encoding_space_ref: u16,
    pub trajectory_dimensions: u16,
    pub sample_time_us: f32,

    pub position: [f32; 3],
    pub read_dir: [f32; 3],
    pub phase_dir: [f32; 3],
    pub slice_dir: [f32; 3],
    pub patient_table_position: [f32; 3],

    pub idx: EncodingCounters,
    pub user_int: [i32; 8],
    pub user_float: [f32; 8],
}

// ---------------------------------------------------------------------------
// ISMRMRD acquisition flag bits (see ismrmrd.h ISMRMRD_AcquisitionFlags).
// Enum values there are 1-based bit positions; we store them as the raw
// bitmask `1 << (pos - 1)`.
// ---------------------------------------------------------------------------
pub mod flags {
    pub const ACQ_FIRST_IN_ENCODE_STEP1: u64 = 1 << 0; // bit position 1
    pub const ACQ_LAST_IN_ENCODE_STEP1: u64 = 1 << (2 - 1);
    pub const ACQ_FIRST_IN_SLICE: u64 = 1 << (7 - 1);
    pub const ACQ_LAST_IN_SLICE: u64 = 1 << (8 - 1);
    pub const ACQ_IS_NOISE_MEASUREMENT: u64 = 1 << (19 - 1);
    pub const ACQ_IS_PARALLEL_CALIBRATION: u64 = 1 << (20 - 1);
    pub const ACQ_IS_PARALLEL_CALIBRATION_AND_IMG: u64 = 1 << (21 - 1);
    pub const ACQ_IS_REVERSE: u64 = 1 << (22 - 1);
    pub const ACQ_IS_NAVIGATION_DATA: u64 = 1 << (23 - 1);
    pub const ACQ_IS_PHASECORR_DATA: u64 = 1 << (24 - 1);
    pub const ACQ_LAST_IN_MEASUREMENT: u64 = 1 << (25 - 1);
    pub const ACQ_IS_HPFEEDBACK_DATA: u64 = 1 << (26 - 1);
    pub const ACQ_IS_DUMMYSCAN_DATA: u64 = 1 << (27 - 1);
    pub const ACQ_IS_RTFEEDBACK_DATA: u64 = 1 << (28 - 1);
    pub const ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA: u64 = 1 << (29 - 1);

    /// Scans flagged with any of these bits should not be placed into the
    /// image k-space grid -- they are calibration/noise/navigator lines.
    pub const NON_IMAGE_MASK: u64 = ACQ_IS_NOISE_MEASUREMENT
        | ACQ_IS_PHASECORR_DATA
        | ACQ_IS_NAVIGATION_DATA
        | ACQ_IS_RTFEEDBACK_DATA
        | ACQ_IS_HPFEEDBACK_DATA
        | ACQ_IS_DUMMYSCAN_DATA
        | ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA;
}

impl AcquisitionHeader {
    /// True if this acquisition should contribute to the image k-space grid.
    #[inline]
    pub fn is_image_scan(&self) -> bool {
        (self.flags & flags::NON_IMAGE_MASK) == 0
    }

    /// Is this a noise calibration scan? (used for pre-whitening)
    #[inline]
    pub fn is_noise(&self) -> bool {
        (self.flags & flags::ACQ_IS_NOISE_MEASUREMENT) != 0
    }

    /// Is the readout sampled in reverse (e.g. even lines of a bipolar
    /// EPI / bi-directional TSE)? Callers must reverse the sample order
    /// before placing the line into k-space.
    #[inline]
    pub fn is_reverse(&self) -> bool {
        (self.flags & flags::ACQ_IS_REVERSE) != 0
    }
}

// ---------------------------------------------------------------------------
// Assembled acquisition (header + decoded complex samples)
// ---------------------------------------------------------------------------
#[derive(Debug)]
pub struct Acquisition {
    pub header: AcquisitionHeader,
    /// Complex samples in `[channel, sample]` row-major order.
    /// Length = `active_channels * number_of_samples`.
    ///
    /// `channel`/`channel_mut`/`as_array_view`/`as_array_view_mut` all
    /// index or reshape against `header.active_channels *
    /// header.number_of_samples`, so this invariant must hold whenever
    /// those are called. `from_raw_f32` upholds it when fed data whose
    /// length matches the header (as `IsmrmrdFile::for_each` verifies
    /// before constructing an `Acquisition` from file input); direct
    /// struct construction with a mismatched header is a caller bug, not
    /// something these accessors defend against.
    pub data: Vec<Complex32>,
}

impl Acquisition {
    /// Build from a flat `f32` vlen where pairs are (real, imag) and the
    /// storage order on disk is channel-major (all samples for ch 0, then ch 1, ...).
    ///
    /// Capacity is sized from `interleaved` (data already read off disk),
    /// not from the header's `number_of_samples` / `active_channels`
    /// fields -- those are file-controlled, and a caller that skips the
    /// length cross-check (see `IsmrmrdFile::for_each`) must not be able
    /// to force an allocation independent of how much data actually
    /// arrived. This intentionally does not assert `interleaved.len() ==
    /// number_of_samples * active_channels * 2`: that check belongs to
    /// callers that can report a proper parse error, not a panic, on a
    /// mismatched file.
    pub fn from_raw_f32(header: AcquisitionHeader, interleaved: &[f32]) -> Self {
        let mut data = Vec::with_capacity(interleaved.len() / 2);
        for pair in interleaved.chunks_exact(2) {
            data.push(Complex32::new(pair[0], pair[1]));
        }
        Acquisition { header, data }
    }

    #[inline]
    pub fn num_samples(&self) -> usize {
        self.header.number_of_samples as usize
    }
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.header.active_channels as usize
    }

    /// Samples for a single channel as a contiguous slice.
    pub fn channel(&self, ch: usize) -> &[Complex32] {
        let ns = self.num_samples();
        &self.data[ch * ns..(ch + 1) * ns]
    }

    /// Mutable samples for a single channel.
    pub fn channel_mut(&mut self, ch: usize) -> &mut [Complex32] {
        let ns = self.num_samples();
        &mut self.data[ch * ns..(ch + 1) * ns]
    }

    /// View the data as an `ndarray` `[channel, sample]` view.
    pub fn as_array_view(&self) -> ndarray::ArrayView2<'_, Complex32> {
        let ns = self.num_samples();
        let nc = self.num_channels();
        ndarray::ArrayView2::from_shape((nc, ns), &self.data)
            .expect("Acquisition::data shape invariant")
    }

    /// Mutable `[channel, sample]` view.
    pub fn as_array_view_mut(&mut self) -> ndarray::ArrayViewMut2<'_, Complex32> {
        let ns = self.num_samples();
        let nc = self.num_channels();
        ndarray::ArrayViewMut2::from_shape((nc, ns), &mut self.data)
            .expect("Acquisition::data shape invariant")
    }
}

// Note: ISMRMRD's on-disk compound is 340 bytes (packed). Our Rust #[repr(C)]
// layout includes natural alignment padding and is larger -- this is fine
// because HDF5 converts compound types by field name, not by byte offset.

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    /// A header whose `number_of_samples` / `active_channels` claim far more
    /// data than is actually provided must not drive the allocation in
    /// `from_raw_f32` -- capacity should track the real (small) input, not
    /// the file-controlled header fields.
    #[test]
    fn from_raw_f32_caps_capacity_to_actual_data_not_header() {
        let mut header = AcquisitionHeader::default();
        header.number_of_samples = u16::MAX;
        header.active_channels = u16::MAX;

        let interleaved = [1.0f32, 2.0, 3.0, 4.0]; // 2 complex samples
        let acq = Acquisition::from_raw_f32(header, &interleaved);

        assert_eq!(acq.data.len(), 2);
        assert!(acq.data.capacity() < 1_000);
        assert_eq!(acq.data[0], Complex32::new(1.0, 2.0));
        assert_eq!(acq.data[1], Complex32::new(3.0, 4.0));
    }
}
