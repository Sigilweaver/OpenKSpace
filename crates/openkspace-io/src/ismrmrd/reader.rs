//! ISMRMRD file reader: opens an HDF5 container, parses the XML header, and
//! streams/assembles the cartesian k-space tensor.

use crate::error::{IoError, IoResult};
use crate::ismrmrd::acquisition::{flags, Acquisition, AcquisitionHeader};
use crate::ismrmrd::header::IsmrmrdHeader;

use hdf5_metno::types::VarLenArray;
use hdf5_metno::{File, H5Type};
use ndarray::Array4;
use num_complex::Complex32;
use std::path::Path;
use tracing::{debug, info, warn};

/// On-disk row layout of `/dataset/data`.
///
/// We derive `H5Type` so hdf5-metno can map the compound by field name.
#[derive(Debug, H5Type)]
#[repr(C)]
struct AcquisitionRow {
    head: AcquisitionHeader,
    traj: VarLenArray<f32>,
    data: VarLenArray<f32>,
}

/// A handle to an opened ISMRMRD file.
pub struct IsmrmrdFile {
    file: File,
    pub header: IsmrmrdHeader,
    pub n_acquisitions: usize,
}

/// Non-image calibration acquisitions extracted from an ISMRMRD file.
///
/// These are the inputs to [`NoisePrewhitener`](../../../openkspace_recon/prewhiten/struct.NoisePrewhitener.html)
/// and the phase-correction calibrator: noise-only scans (acquired with
/// RF off) and TSE/EPI navigator echoes used to align even/odd readouts.
#[derive(Default, Debug)]
pub struct CalibrationScans {
    pub noise: Vec<Acquisition>,
    pub phasecorr: Vec<Acquisition>,
}

impl IsmrmrdFile {
    /// Open the HDF5 file and parse the XML header.
    pub fn open<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;

        // Header: /dataset/xml is a variable-length ASCII string.
        // (mridata.org files use H5T_CSET_ASCII, so we can't use VarLenUnicode.)
        let xml_ds = file.dataset("dataset/xml")?;
        let xml: Vec<hdf5_metno::types::VarLenAscii> = xml_ds.read_raw()?;
        let xml_str = xml
            .first()
            .ok_or(IoError::MissingField("dataset/xml"))?
            .as_str()
            .to_string();

        let header = IsmrmrdHeader::parse(&xml_str)?;

        let data_ds = file.dataset("dataset/data")?;
        let n = data_ds.size();

        info!(
            "Opened {}  -- {} acquisitions, encoded {}x{}x{}, recon {}x{}x{}, {} channels ({})",
            path.display(),
            n,
            header.encoding.encoded_matrix.x,
            header.encoding.encoded_matrix.y,
            header.encoding.encoded_matrix.z,
            header.encoding.recon_matrix.x,
            header.encoding.recon_matrix.y,
            header.encoding.recon_matrix.z,
            header.receiver_channels,
            header.encoding.trajectory,
        );

        Ok(IsmrmrdFile {
            file,
            header,
            n_acquisitions: n,
        })
    }

    /// Iterate over every acquisition in the file.
    ///
    /// Each row is read individually to keep peak memory bounded -- this
    /// works for multi-GB files without loading the whole compound array.
    pub fn for_each<F>(&self, mut f: F) -> IoResult<()>
    where
        F: FnMut(Acquisition) -> IoResult<()>,
    {
        let ds = self.file.dataset("dataset/data")?;
        let n = ds.size();

        // Read in chunks to amortize HDF5 overhead while keeping memory low.
        const CHUNK: usize = 256;

        let mut start = 0usize;
        while start < n {
            let end = (start + CHUNK).min(n);
            let arr = ds.read_slice_1d::<AcquisitionRow, _>(start..end)?;

            for row in arr.into_iter() {
                // hdf5-metno's VarLenArray<T> derefs to &[T].
                let samples: &[f32] = &row.data;
                // number_of_samples / active_channels come straight from the
                // file; guard the multiply so a crafted header can't wrap
                // around to a small `expected` and slip past the length
                // check below.
                let expected = (row.head.number_of_samples as usize)
                    .checked_mul(row.head.active_channels as usize)
                    .and_then(|v| v.checked_mul(2));
                if expected != Some(samples.len()) {
                    return Err(IoError::Inconsistent(format!(
                        "acquisition {}: data len {} != expected {:?} \
                         ({} samples x {} channels x 2)",
                        start,
                        samples.len(),
                        expected,
                        row.head.number_of_samples,
                        row.head.active_channels,
                    )));
                }
                let acq = Acquisition::from_raw_f32(row.head, samples);
                f(acq)?;
            }

            start = end;
        }
        Ok(())
    }

    /// Assemble a cartesian k-space tensor of shape `[channels, kz, ky, kx]`.
    ///
    /// Non-image scans (noise, navigators, dummy, phasecorr) are skipped.
    /// For multi-slice 2D sequences (where `kspace_encode_step_2` is not
    /// used), the slice index from `idx.slice` is used as the 3rd axis.
    /// All acquisitions sharing a coordinate are overwritten by the last.
    ///
    /// # Sizing strategy
    ///
    /// Many vendor-converted ISMRMRD files have inconsistent headers: the
    /// declared `encoded_matrix` may not match the observed ranges of the
    /// per-acquisition `idx.*` fields (e.g. encoded_matrix.z=30 but only
    /// 15 slices present, or ky indices exceeding encoded_matrix.y).
    ///
    /// To be robust, we perform a cheap first pass to determine the actual
    /// observed ranges of ky, kz, slice, and channel count. The tensor is
    /// sized to `max(header, observed) + 1` on each axis.
    /// Assemble a cartesian k-space tensor of shape `[channels, kz, ky, kx]`.
    ///
    /// Non-image scans (noise, navigators, dummy, phasecorr) are skipped.
    /// See [`read_kspace_with`](Self::read_kspace_with) for the corrections-aware
    /// variant.
    pub fn read_kspace(&self) -> IoResult<Array4<Complex32>> {
        self.read_kspace_with(|_| {})
    }

    /// Collect all calibration / non-image scans into memory.
    ///
    /// This is the path used by noise pre-whitening and navigator
    /// phase-correction: a single pass over the file pulls out the sparse
    /// set of scans that feed those calibrators, without allocating a
    /// full k-space tensor.
    pub fn read_calibration(&self) -> IoResult<CalibrationScans> {
        let mut out = CalibrationScans::default();
        self.for_each(|acq| {
            if (acq.header.flags & flags::ACQ_IS_NOISE_MEASUREMENT) != 0 {
                out.noise.push(acq);
            } else if (acq.header.flags & flags::ACQ_IS_PHASECORR_DATA) != 0 {
                out.phasecorr.push(acq);
            }
            Ok(())
        })?;
        info!(
            "Calibration scans: {} noise, {} phase-correction",
            out.noise.len(),
            out.phasecorr.len()
        );
        Ok(out)
    }

    /// Returns `true` if this dataset uses the 3D partition axis
    /// (`kspace_encode_step_2`) rather than discrete 2D slices.
    pub fn is_3d_encoding(&self) -> IoResult<bool> {
        // Cheap heuristic: scan until we see any nonzero kspace_encode_step_2
        // on an image acquisition. 3D sequences drive this every line;
        // 2D multi-slice sequences leave it at 0.
        let mut three_d = false;
        self.for_each(|acq| {
            if three_d {
                return Ok(());
            }
            if acq.header.is_image_scan() && acq.header.idx.kspace_encode_step_2 > 0 {
                three_d = true;
            }
            Ok(())
        })?;
        Ok(three_d)
    }

    /// Same as [`read_kspace`](Self::read_kspace), but invokes `preprocess`
    /// on each *image* acquisition before it is placed into the grid.
    ///
    /// This is the extension point used by the reconstruction pipeline to
    /// apply noise pre-whitening, navigator phase correction, or any other
    /// per-line transform. Non-image scans (noise, phasecorr, dummy) are
    /// never forwarded to `preprocess` -- if you need those, call
    /// [`read_calibration`](Self::read_calibration) first.
    pub fn read_kspace_with<F>(&self, preprocess: F) -> IoResult<Array4<Complex32>>
    where
        F: FnMut(&mut Acquisition),
    {
        self.read_kspace_with_mask(preprocess).map(|(k, _m)| k)
    }

    /// Same as [`read_kspace_with`](Self::read_kspace_with), but also returns
    /// a per-cell `filled` mask of shape `[kz, ky, kx]`. A cell is `true` iff
    /// at least one acquisition was placed at that location.
    ///
    /// GRAPPA and other parallel-imaging strategies use the mask to detect
    /// the undersampling pattern and the auto-calibration region.
    pub fn read_kspace_with_mask<F>(
        &self,
        mut preprocess: F,
    ) -> IoResult<(Array4<Complex32>, ndarray::Array3<bool>)>
    where
        F: FnMut(&mut Acquisition),
    {
        let enc = &self.header.encoding;

        // --- First pass: probe observed ranges ------------------------------
        let mut max_ky: u16 = 0;
        let mut max_kz_step: u16 = 0;
        let mut max_slice: u16 = 0;
        let mut max_samples: u16 = 0;
        let mut nc_observed: u16 = 0;
        let mut n_image: usize = 0;
        let mut n_nonimage: usize = 0;

        self.for_each(|acq| {
            if !acq.header.is_image_scan() {
                n_nonimage += 1;
                return Ok(());
            }
            n_image += 1;
            let i = &acq.header.idx;
            max_ky = max_ky.max(i.kspace_encode_step_1);
            max_kz_step = max_kz_step.max(i.kspace_encode_step_2);
            max_slice = max_slice.max(i.slice);
            max_samples = max_samples.max(acq.header.number_of_samples);
            nc_observed = nc_observed.max(acq.header.active_channels);
            Ok(())
        })?;

        if n_image == 0 {
            return Err(IoError::MissingField("no image acquisitions found"));
        }

        // --- Resolve final tensor shape (max of header and observed) --------
        let use_slice_axis = max_kz_step == 0 && max_slice > 0;

        let nx = (enc.encoded_matrix.x as usize).max(max_samples as usize);
        let ny = (enc.encoded_matrix.y as usize)
            .max(max_ky as usize + 1)
            .max(1);
        let nz = if use_slice_axis {
            (max_slice as usize + 1).max(1)
        } else {
            (enc.encoded_matrix.z as usize)
                .max(max_kz_step as usize + 1)
                .max(1)
        };

        let nc = {
            let declared = self.header.receiver_channels as usize;
            let observed = nc_observed as usize;
            if declared != 0 && declared != observed {
                warn!(
                    "header declares {} channels, observed {} -- using {}",
                    declared, observed, observed
                );
            }
            if observed == 0 {
                return Err(IoError::MissingField("active_channels (0)"));
            }
            observed
        };

        info!(
            "Sizing: header {}x{}x{} -> observed ky={}, kz_step={}, slice={}, samples={}",
            enc.encoded_matrix.x,
            enc.encoded_matrix.y,
            enc.encoded_matrix.z,
            max_ky + 1,
            max_kz_step + 1,
            max_slice + 1,
            max_samples,
        );
        if use_slice_axis {
            info!("Multi-slice 2D dataset: using idx.slice as 3rd axis.");
        }

        info!(
            "Allocating k-space tensor [ch={}, kz={}, ky={}, kx={}]  (~{:.2} GiB)",
            nc,
            nz,
            ny,
            nx,
            (nc * nz * ny * nx * std::mem::size_of::<Complex32>()) as f64
                / (1024.0 * 1024.0 * 1024.0),
        );

        let mut kspace = Array4::<Complex32>::zeros((nc, nz, ny, nx));
        // Per-cell hit counter. We sum repeat acquisitions and divide by the
        // hit count at the end (i.e. take the mean) -- correct for true NEX
        // signal averaging and harmless on uniquely-sampled data.
        //
        // NOTE: when the repeats are phase-incoherent (frequency drift
        // between NEXes, or different echoes of a TSE train being written
        // to the same cell), complex summation is destructive. The repeat
        // logic here therefore intentionally takes the **first** acquisition
        // for any cell and discards subsequent writers. This is equivalent
        // to a "one-hit" reconstruction and avoids the trouble entirely;
        // true complex averaging should only be applied after appropriate
        // phase alignment, which requires vendor-specific calibration data.
        let mut filled = ndarray::Array3::<bool>::from_elem((nz, ny, nx), false);

        let mut placed = 0usize;
        let mut skipped = 0usize;
        let mut reversed = 0usize;

        // --- DC centering offsets -------------------------------------------
        //
        // ISMRMRD encodes idx.kspace_encode_step_1 as a zero-based index into
        // the encoded matrix, with DC at `ky_limit.center`. For a centred
        // IFFT, DC must land at `ny/2`. We shift each placement by the
        // delta so that DC ends up at the array centre, regardless of
        // asymmetric sampling / partial-Fourier acquisition.
        let ky_dc_offset: isize = (ny as isize / 2) - (enc.ky_limit.center as isize);
        let kz_dc_offset: isize = if use_slice_axis {
            0 // slice axis is not Fourier-transformed
        } else {
            (nz as isize / 2) - (enc.kz_limit.center as isize)
        };

        // For kx, the acquisition's `center_sample` gives the kx=0 location
        // within the readout. Apply the same centering logic per-acquisition.
        info!(
            "DC offsets: ky+={}, kz+={}  (ky_center={}, kz_center={})",
            ky_dc_offset, kz_dc_offset, enc.ky_limit.center, enc.kz_limit.center,
        );

        // --- Second pass: place each image acquisition ----------------------
        self.for_each(|mut acq| {
            if !acq.header.is_image_scan() {
                skipped += 1;
                return Ok(());
            }

            // Per-line corrections (prewhitening, phase correction, ...).
            preprocess(&mut acq);

            let ky_src = acq.header.idx.kspace_encode_step_1 as isize;
            let kz_src = if use_slice_axis {
                acq.header.idx.slice as isize
            } else {
                acq.header.idx.kspace_encode_step_2 as isize
            };

            let ky = ky_src + ky_dc_offset;
            let kz = kz_src + kz_dc_offset;

            if ky < 0 || ky >= ny as isize || kz < 0 || kz >= nz as isize {
                debug!("out-of-range acq (kz={}, ky={}) -- skipped", kz, ky);
                skipped += 1;
                return Ok(());
            }
            let ky = ky as usize;
            let kz = kz as usize;

            let ns = acq.num_samples();
            let discard_pre = acq.header.discard_pre as usize;
            let discard_post = acq.header.discard_post as usize;
            let keep = ns.saturating_sub(discard_pre + discard_post);

            // Centre kx by mapping the acquisition's center_sample to nx/2.
            // Fall back to centring the kept readout within nx when
            // center_sample is 0 (unset).
            let cs = acq.header.center_sample as usize;
            let kx_dst_center = nx / 2;
            let dst_off: usize = if cs > 0 && cs >= discard_pre {
                let cs_in_kept = cs - discard_pre;
                kx_dst_center.saturating_sub(cs_in_kept)
            } else {
                nx.saturating_sub(keep) / 2
            };

            let nc_here = acq.num_channels();
            let nc_fit = nc_here.min(nc);

            let copy_len_common = {
                let src_len = keep.min(acq.channel(0).len().saturating_sub(discard_pre));
                src_len.min(nx.saturating_sub(dst_off))
            };
            if copy_len_common == 0 {
                skipped += 1;
                return Ok(());
            }

            // First-wins collision check at the centre of the readout
            let centre_kx = dst_off + copy_len_common / 2;
            if filled[[kz, ky, centre_kx]] {
                skipped += 1;
                return Ok(());
            }

            // Bipolar / reverse-readout lines must be time-reversed before
            // placement -- their samples run from +kx_max to -kx_max rather
            // than the usual -kx_max to +kx_max. Vendors sometimes leave
            // this as-is on alternating TSE echoes, producing ghosts and
            // horizontal amplitude stripes in the image if not handled.
            let is_reverse = acq.header.is_reverse();
            if is_reverse {
                reversed += 1;
            }

            for ch in 0..nc_fit {
                let src = acq.channel(ch);
                let avail = src.len().saturating_sub(discard_pre);
                let src_len = keep.min(avail);
                let src_slice = &src[discard_pre..discard_pre + src_len];

                let copy_len = src_slice.len().min(nx.saturating_sub(dst_off));
                if copy_len == 0 {
                    continue;
                }

                let mut row =
                    kspace.slice_mut(ndarray::s![ch, kz, ky, dst_off..dst_off + copy_len]);
                if is_reverse {
                    for (i, &v) in src_slice[..copy_len].iter().rev().enumerate() {
                        row[i] = v;
                    }
                } else {
                    for (i, &v) in src_slice[..copy_len].iter().enumerate() {
                        row[i] = v;
                    }
                }
            }

            // Mark this line as filled so later duplicates are skipped.
            let mut f_row =
                filled.slice_mut(ndarray::s![kz, ky, dst_off..dst_off + copy_len_common]);
            f_row.mapv_inplace(|_| true);

            placed += 1;
            Ok(())
        })?;

        info!(
            "Placed {} acquisitions, skipped {} (of which {} non-image). {} reversed readouts flipped.",
            placed, skipped, n_nonimage, reversed
        );
        Ok((kspace, filled))
    }
}
