//! FastMRI HDF5 reader.
//!
//! FastMRI files (from fastmri.med.nyu.edu) store a pre-assembled k-space
//! tensor and a ground-truth RSS reconstruction directly in the HDF5 root:
//!
//! ```text
//! /kspace              complex64  [slices, coils, ky, kx]
//! /reconstruction_rss  float32    [slices, recon_y, recon_x]   (ground truth)
//! /ismrmrd_header      str        ISMRMRD-compatible XML
//!
//! File-level attrs:
//!   acquisition   str   e.g. "CORPDFS_FBK", "AXT2"
//!   patient_id    str   SHA-256 anonymised subject ID
//!   max           str   string-encoded f32 max value
//!   norm          str   string-encoded f32 norm
//! ```
//!
//! The `ismrmrd_header` XML is structurally identical to that in an ISMRMRD
//! file, so [`IsmrmrdHeader`] is reused here for metadata.
//!
//! The k-space tensor axis order `[slices, coils, ky, kx]` is converted to
//! the pipeline-standard `[coils, kz/slices, ky, kx]` on load, matching the
//! shape returned by [`IsmrmrdFile::read_kspace`].

use crate::error::{IoError, IoResult};
use crate::ismrmrd::header::IsmrmrdHeader;

use hdf5_metno::{File, H5Type};
use ndarray::{Array2, Array3, Array4};
use num_complex::Complex32;
use std::path::Path;
use tracing::info;

/// HDF5 complex64 on-disk layout: two contiguous f32s (real, imag).
///
/// We derive `H5Type` so hdf5-metno maps the compound dtype by field name,
/// then convert to `num_complex::Complex32` ourselves. This avoids the
/// ndarray version mismatch that arises when asking hdf5-metno to return
/// `ndarray::Array<Complex32, _>` directly (hdf5-metno 0.9 bundles its own
/// ndarray 0.16 while the workspace uses 0.15).
#[derive(Debug, Clone, Copy, H5Type)]
#[repr(C)]
struct Cf32 {
    re: f32,
    im: f32,
}

impl From<Cf32> for Complex32 {
    #[inline]
    fn from(c: Cf32) -> Self {
        Complex32::new(c.re, c.im)
    }
}

// ── Helpers to read flat Vecs from HDF5 ──────────────────────────────────────

fn read_complex_flat(file: &File, path: &str) -> IoResult<(Vec<Complex32>, Vec<usize>)> {
    let ds    = file.dataset(path)?;
    let shape = ds.shape();
    let raw: Vec<Cf32> = ds.read_raw()?;
    let data: Vec<Complex32> = raw.into_iter().map(Into::into).collect();
    Ok((data, shape))
}

fn read_f32_flat(file: &File, path: &str) -> IoResult<(Vec<f32>, Vec<usize>)> {
    let ds    = file.dataset(path)?;
    let shape = ds.shape();
    let data: Vec<f32> = ds.read_raw()?;
    Ok((data, shape))
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Metadata parsed from a FastMRI file.
#[derive(Debug, Clone)]
pub struct FastmriMeta {
    /// ISMRMRD-compatible XML header.
    pub header: IsmrmrdHeader,
    /// Acquisition type label, e.g. `"CORPDFS_FBK"`, `"AXT2"`.
    pub acquisition: String,
    /// Anonymised patient identifier (SHA-256 hex string).
    pub patient_id: String,
    /// Number of slices in this file.
    pub n_slices: usize,
    /// Number of coils.
    pub n_coils: usize,
    /// Encoded ky dimension.
    pub n_ky: usize,
    /// Encoded kx dimension.
    pub n_kx: usize,
    /// Recon y dimension (from `reconstruction_rss`).
    pub recon_y: usize,
    /// Recon x dimension (from `reconstruction_rss`).
    pub recon_x: usize,
}

/// A handle to an opened FastMRI file.
pub struct FastmriFile {
    file: File,
    pub meta: FastmriMeta,
}

impl FastmriFile {
    /// Open the HDF5 file, parse the header, and validate the tensor shapes.
    pub fn open<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;

        // ── XML header ────────────────────────────────────────────────────
        let xml_ds  = file.dataset("ismrmrd_header")?;
        let xml_str = xml_ds
            .read_scalar::<hdf5_metno::types::VarLenUnicode>()
            .map(|s| s.as_str().to_string())
            .or_else(|_| {
                xml_ds
                    .read_scalar::<hdf5_metno::types::VarLenAscii>()
                    .map(|s| s.as_str().to_string())
            })
            .map_err(|_| IoError::MissingField("ismrmrd_header"))?;

        let header = IsmrmrdHeader::parse(&xml_str)?;

        // ── k-space shape ─────────────────────────────────────────────────
        let kshape = file.dataset("kspace")?.shape();
        if kshape.len() != 4 {
            return Err(IoError::Unsupported(format!(
                "kspace has {} dims (expected 4: [slices, coils, ky, kx])",
                kshape.len()
            )));
        }
        let (n_slices, n_coils, n_ky, n_kx) =
            (kshape[0], kshape[1], kshape[2], kshape[3]);

        // ── RSS reconstruction shape ──────────────────────────────────────
        let rshape = file.dataset("reconstruction_rss")?.shape();
        if rshape.len() != 3 {
            return Err(IoError::Unsupported(format!(
                "reconstruction_rss has {} dims (expected 3: [slices, y, x])",
                rshape.len()
            )));
        }
        if rshape[0] != n_slices {
            return Err(IoError::Inconsistent(format!(
                "kspace has {} slices but reconstruction_rss has {}",
                n_slices, rshape[0]
            )));
        }
        let (recon_y, recon_x) = (rshape[1], rshape[2]);

        // ── File-level attributes ─────────────────────────────────────────
        let attr_str = |key: &str| -> String {
            file.attr(key)
                .and_then(|a| a.read_scalar::<hdf5_metno::types::VarLenUnicode>())
                .map(|s| s.as_str().to_string())
                .unwrap_or_default()
        };
        let acquisition = attr_str("acquisition");
        let patient_id  = attr_str("patient_id");

        info!(
            "Opened {}  -- {} slices, {} coils, ky={}, kx={}, recon={}x{}, acq={:?}",
            path.display(),
            n_slices, n_coils, n_ky, n_kx, recon_y, recon_x, acquisition,
        );

        Ok(FastmriFile {
            file,
            meta: FastmriMeta {
                header,
                acquisition,
                patient_id,
                n_slices,
                n_coils,
                n_ky,
                n_kx,
                recon_y,
                recon_x,
            },
        })
    }

    /// Read the full k-space tensor and return it as `[coils, slices, ky, kx]`.
    ///
    /// This matches the axis order produced by [`IsmrmrdFile::read_kspace`]
    /// so both can be fed into the same reconstruction pipeline unchanged.
    ///
    /// The raw HDF5 layout is `[slices, coils, ky, kx]`; we permute axes on load.
    pub fn read_kspace(&self) -> IoResult<Array4<Complex32>> {
        let m = &self.meta;
        let (data, _) = read_complex_flat(&self.file, "kspace")?;
        // data is in [slices, coils, ky, kx] row-major order.
        // Build output in [coils, slices, ky, kx] order.
        let (ns, nc, nky, nkx) = (m.n_slices, m.n_coils, m.n_ky, m.n_kx);
        let coil_stride = nky * nkx;
        let slice_stride = nc * coil_stride;

        let mut out = Array4::<Complex32>::zeros((nc, ns, nky, nkx));
        for s in 0..ns {
            for c in 0..nc {
                let src_off = s * slice_stride + c * coil_stride;
                out.slice_mut(ndarray::s![c, s, .., ..])
                   .iter_mut()
                   .zip(&data[src_off..src_off + coil_stride])
                   .for_each(|(dst, src)| *dst = *src);
            }
        }
        Ok(out)
    }

    /// Read a single slice's k-space as `[coils, ky, kx]`.
    ///
    /// More memory-efficient than [`read_kspace`](Self::read_kspace) when
    /// only one slice is needed - still loads the full file but avoids
    /// allocating the full permuted tensor.
    pub fn read_kspace_slice(&self, slice_idx: usize) -> IoResult<Array3<Complex32>> {
        let m = &self.meta;
        if slice_idx >= m.n_slices {
            return Err(IoError::Inconsistent(format!(
                "slice {} out of range (file has {} slices)",
                slice_idx, m.n_slices
            )));
        }
        let (data, _) = read_complex_flat(&self.file, "kspace")?;
        let coil_stride  = m.n_ky * m.n_kx;
        let slice_stride = m.n_coils * coil_stride;
        let slice_base   = slice_idx * slice_stride;

        let mut out = Array3::<Complex32>::zeros((m.n_coils, m.n_ky, m.n_kx));
        for c in 0..m.n_coils {
            let src_off = slice_base + c * coil_stride;
            out.slice_mut(ndarray::s![c, .., ..])
               .iter_mut()
               .zip(&data[src_off..src_off + coil_stride])
               .for_each(|(dst, src)| *dst = *src);
        }
        Ok(out)
    }

    /// Read the ground-truth RSS reconstruction as `[slices, recon_y, recon_x]`.
    pub fn read_reconstruction_rss(&self) -> IoResult<Array3<f32>> {
        let m = &self.meta;
        let (data, _) = read_f32_flat(&self.file, "reconstruction_rss")?;
        Array3::from_shape_vec((m.n_slices, m.recon_y, m.recon_x), data)
            .map_err(|e| IoError::Inconsistent(e.to_string()))
    }

    /// Read the ground-truth RSS for a single slice as `[recon_y, recon_x]`.
    pub fn read_reconstruction_rss_slice(&self, slice_idx: usize) -> IoResult<Array2<f32>> {
        let m = &self.meta;
        if slice_idx >= m.n_slices {
            return Err(IoError::Inconsistent(format!(
                "slice {} out of range (file has {} slices)",
                slice_idx, m.n_slices
            )));
        }
        let (data, _) = read_f32_flat(&self.file, "reconstruction_rss")?;
        let stride = m.recon_y * m.recon_x;
        let off    = slice_idx * stride;
        Array2::from_shape_vec((m.recon_y, m.recon_x), data[off..off + stride].to_vec())
            .map_err(|e| IoError::Inconsistent(e.to_string()))
    }

    /// `true` if this file appears to be a brain scan based on acquisition label.
    pub fn is_brain(&self) -> bool {
        self.meta.acquisition.to_ascii_uppercase().contains("AX")
    }
}
