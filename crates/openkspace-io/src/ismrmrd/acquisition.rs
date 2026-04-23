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
#[derive(Debug, Clone, Copy, H5Type)]
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
#[derive(Debug, Clone, Copy, H5Type)]
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
    pub const ACQ_FIRST_IN_ENCODE_STEP1: u64 = 1 << (1 - 1);
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
    pub data: Vec<Complex32>,
}

impl Acquisition {
    /// Build from a flat `f32` vlen where pairs are (real, imag) and the
    /// storage order on disk is channel-major (all samples for ch 0, then ch 1, ...).
    pub fn from_raw_f32(header: AcquisitionHeader, interleaved: &[f32]) -> Self {
        let ns = header.number_of_samples as usize;
        let nc = header.active_channels as usize;
        debug_assert_eq!(interleaved.len(), ns * nc * 2);

        let mut data = Vec::with_capacity(ns * nc);
        let mut i = 0;
        while i < interleaved.len() {
            data.push(Complex32::new(interleaved[i], interleaved[i + 1]));
            i += 2;
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
