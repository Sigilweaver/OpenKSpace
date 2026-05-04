//! OpenKSpace command-line interface.
//!
//! Subcommands:
//!   info   -- print header metadata for an ISMRMRD or FastMRI .h5 file
//!   recon  -- run a cartesian IFFT + RSS reconstruction -> PNG(s)

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array2, Axis};
use openkspace_io::ismrmrd::IsmrmrdFile;
use openkspace_io::{is_fastmri, FastmriFile};
use openkspace_recon::{
    center_crop_3d, ifft2_inplace, rss_combine_4d, CsRss, FftMode, GrappaRss, IfftRss,
    ReconStrategy, SenseMapSource, SenseRss,
};
use serde_json::json;
use std::path::PathBuf;
use std::time::Duration;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(
    name = "openkspace",
    about = "OpenKSpace: MRI k-space reconstruction engine",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,

    /// Verbose logging (can be repeated: -v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Print header metadata for an ISMRMRD or FastMRI file.
    Info {
        /// Path to .h5 file
        file: PathBuf,

        /// Emit metadata as JSON (for tooling / scripting integration)
        #[arg(long)]
        json: bool,
    },

    /// Probe index ranges across all acquisitions (diagnostic).
    Probe {
        /// Path to .h5 file
        file: PathBuf,
    },

    /// Reconstruct image(s) from a cartesian k-space dataset.
    Recon {
        /// Path to .h5 file
        file: PathBuf,

        /// Output directory for PNG image(s)
        #[arg(short, long, default_value = "recon_out")]
        out: PathBuf,

        /// Only reconstruct this single slice index (else all slices)
        #[arg(short, long)]
        slice: Option<usize>,

        /// Lower percentile for contrast windowing (default 0.5)
        #[arg(long, default_value_t = 0.5)]
        pct_low: f32,

        /// Upper percentile for contrast windowing (default 99.5)
        #[arg(long, default_value_t = 99.5)]
        pct_high: f32,

        /// Do not crop to the recon matrix -- keep the full oversampled FOV
        #[arg(long)]
        no_crop: bool,

        /// Disable noise pre-whitening (on by default if noise scans exist)
        #[arg(long)]
        no_prewhiten: bool,

        /// Disable navigator phase correction (on by default if navigators exist)
        #[arg(long)]
        no_phasecorr: bool,

        /// Disable readout oversampling removal
        /// (on by default when encoded_x > recon_x)
        #[arg(long)]
        no_oversampling_removal: bool,

        /// Disable partial-Fourier / homodyne reconstruction along ky
        /// (on by default when the ky mask is asymmetric around DC)
        #[arg(long)]
        no_partial_fourier: bool,

        /// FFT mode: auto (2D/3D from data), 2d, 3d
        #[arg(long, value_enum, default_value_t = FftModeArg::Auto)]
        fft: FftModeArg,

        /// Reconstruction strategy
        #[arg(long, value_enum, default_value_t = StrategyArg::IfftRss)]
        strategy: StrategyArg,

        /// GRAPPA: number of source ky rows per kernel (even, >=2)
        #[arg(long, default_value_t = 4)]
        grappa_kernel_ky: usize,

        /// GRAPPA: number of kx taps per kernel (odd, >=1)
        #[arg(long, default_value_t = 5)]
        grappa_kernel_kx: usize,

        /// GRAPPA: Tikhonov ridge (relative to mean diagonal of A^H A)
        #[arg(long, default_value_t = 1e-3)]
        grappa_ridge: f32,

        /// SENSE: sensitivity-map source
        #[arg(long, value_enum, default_value_t = SenseMapSourceArg::Walsh)]
        sense_maps: SenseMapSourceArg,

        /// SENSE: half-size (in voxels) of the Walsh covariance window
        #[arg(long, default_value_t = 3)]
        sense_walsh_window: usize,

        /// SENSE: number of Walsh power-iteration steps per voxel
        #[arg(long, default_value_t = 6)]
        sense_walsh_iters: usize,

        /// SENSE/ESPIRiT: k-space kernel size (odd)
        #[arg(long, default_value_t = 5)]
        espirit_kernel: usize,

        /// SENSE/ESPIRiT: singular-value threshold (fraction of sigma_max)
        #[arg(long, default_value_t = 0.02)]
        espirit_threshold: f32,

        /// SENSE/ESPIRiT: power-iteration steps
        #[arg(long, default_value_t = 30)]
        espirit_iters: usize,

        /// SENSE: Tikhonov ridge added to C^H C in the unfolding solve
        #[arg(long, default_value_t = 1e-4)]
        sense_ridge: f32,

        /// SENSE: also compute the g-factor map (requires --write-gfactor
        /// to emit images; cheap to enable either way)
        #[arg(long, default_value_t = false)]
        sense_gfactor: bool,

        /// Write g-factor PNGs (requires SENSE with --sense-gfactor)
        #[arg(long, default_value_t = false)]
        write_gfactor: bool,

        /// CS: number of FISTA iterations
        #[arg(long, default_value_t = 60)]
        cs_iters: usize,

        /// CS: L1-wavelet regularisation strength
        #[arg(long, default_value_t = 0.01)]
        cs_lambda: f32,

        /// Output format: per-slice PNG, NIfTI-1 volume, or both
        #[arg(long, value_enum, default_value_t = OutputFormat::Png)]
        format: OutputFormat,
    },
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum SenseMapSourceArg {
    Walsh,
    Espirit,
}

impl From<SenseMapSourceArg> for SenseMapSource {
    fn from(m: SenseMapSourceArg) -> Self {
        match m {
            SenseMapSourceArg::Walsh => SenseMapSource::Walsh,
            SenseMapSourceArg::Espirit => SenseMapSource::Espirit,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum StrategyArg {
    #[value(name = "ifft-rss")]
    IfftRss,
    #[value(name = "grappa")]
    Grappa,
    #[value(name = "sense")]
    Sense,
    #[value(name = "cs")]
    Cs,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
enum FftModeArg {
    Auto,
    #[value(name = "2d")]
    TwoD,
    #[value(name = "3d")]
    ThreeD,
}

impl From<FftModeArg> for FftMode {
    fn from(m: FftModeArg) -> Self {
        match m {
            FftModeArg::Auto => FftMode::Auto,
            FftModeArg::TwoD => FftMode::TwoD,
            FftModeArg::ThreeD => FftMode::ThreeD,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    /// Per-slice PNGs with percentile contrast windowing (default)
    Png,
    /// Single NIfTI-1 volume (.nii)
    Nifti,
    /// Both PNG slices and NIfTI volume
    Both,
}

fn init_tracing(verbosity: u8) {
    let level = match verbosity {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                format!(
                    "openkspace_cli={lvl},openkspace_io={lvl},openkspace_recon={lvl}",
                    lvl = level
                )
                .into()
            }),
        )
        .with_target(false)
        .compact()
        .init();
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    let verbose = cli.verbose;

    match cli.cmd {
        Cmd::Info { file, json } => cmd_info(&file, json),
        Cmd::Probe { file } => cmd_probe(&file),
        Cmd::Recon {
            file,
            out,
            slice,
            pct_low,
            pct_high,
            no_crop,
            no_prewhiten,
            no_phasecorr,
            no_oversampling_removal,
            no_partial_fourier,
            fft,
            strategy,
            grappa_kernel_ky,
            grappa_kernel_kx,
            grappa_ridge,
            sense_maps,
            sense_walsh_window,
            sense_walsh_iters,
            espirit_kernel,
            espirit_threshold,
            espirit_iters,
            sense_ridge,
            sense_gfactor,
            write_gfactor,
            cs_iters,
            cs_lambda,
            format,
        } => cmd_recon(
            &file,
            &out,
            slice,
            pct_low,
            pct_high,
            no_crop,
            no_prewhiten,
            no_phasecorr,
            no_oversampling_removal,
            no_partial_fourier,
            fft.into(),
            strategy,
            grappa_kernel_ky,
            grappa_kernel_kx,
            grappa_ridge,
            sense_maps.into(),
            sense_walsh_window,
            sense_walsh_iters,
            espirit_kernel,
            espirit_threshold,
            espirit_iters,
            sense_ridge,
            sense_gfactor,
            write_gfactor,
            cs_iters,
            cs_lambda,
            format,
            verbose,
        ),
    }
}

fn cmd_info(path: &PathBuf, json: bool) -> Result<()> {
    match detect_format(path)? {
        FileFormat::Ismrmrd => cmd_info_ismrmrd(path, json),
        FileFormat::FastMri => cmd_info_fastmri(path, json),
    }
}

// ── Format detection ──────────────────────────────────────────────────────────

enum FileFormat {
    Ismrmrd,
    FastMri,
}

/// Probe the HDF5 root to distinguish ISMRMRD from FastMRI.
///
/// ISMRMRD has `/dataset/data` (compound acquisition records).
/// FastMRI has `/kspace` (pre-assembled complex tensor).
/// We probe by attempting to open each in turn.
fn detect_format(path: &PathBuf) -> Result<FileFormat> {
    if is_fastmri(path) {
        Ok(FileFormat::FastMri)
    } else if IsmrmrdFile::open(path).is_ok() {
        Ok(FileFormat::Ismrmrd)
    } else {
        bail!(
            "{}: cannot determine file format (not a valid ISMRMRD or FastMRI HDF5 file)",
            path.display()
        )
    }
}

// ── info ──────────────────────────────────────────────────────────────────────

fn cmd_info_ismrmrd(path: &PathBuf, json: bool) -> Result<()> {
    let f = IsmrmrdFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let h = &f.header;

    if json {
        let v = json!({
            "format": "ISMRMRD",
            "file": path.display().to_string(),
            "acquisitions": f.n_acquisitions,
            "vendor": h.system_vendor,
            "model": h.system_model,
            "field_strength_t": h.field_strength_t,
            "channels": h.receiver_channels,
            "trajectory": h.encoding.trajectory,
            "encoded_matrix": {
                "x": h.encoding.encoded_matrix.x,
                "y": h.encoding.encoded_matrix.y,
                "z": h.encoding.encoded_matrix.z
            },
            "recon_matrix": {
                "x": h.encoding.recon_matrix.x,
                "y": h.encoding.recon_matrix.y,
                "z": h.encoding.recon_matrix.z
            },
            "encoded_fov_mm": {
                "x": h.encoding.encoded_fov.x,
                "y": h.encoding.encoded_fov.y,
                "z": h.encoding.encoded_fov.z
            },
            "ky_range": {
                "min": h.encoding.ky_limit.minimum,
                "max": h.encoding.ky_limit.maximum,
                "center": h.encoding.ky_limit.center
            },
            "slice_range": {
                "min": h.encoding.slice_limit.minimum,
                "max": h.encoding.slice_limit.maximum
            }
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }

    println!("Format        : ISMRMRD");
    println!("File          : {}", path.display());
    println!("Acquisitions  : {}", f.n_acquisitions);
    println!("Vendor        : {}", h.system_vendor);
    println!("Model         : {}", h.system_model);
    println!("Field [T]     : {:.3}", h.field_strength_t);
    println!("Channels      : {}", h.receiver_channels);
    println!("Trajectory    : {}", h.encoding.trajectory);
    println!(
        "Encoded matrix: {} x {} x {}",
        h.encoding.encoded_matrix.x, h.encoding.encoded_matrix.y, h.encoding.encoded_matrix.z
    );
    println!(
        "Recon matrix  : {} x {} x {}",
        h.encoding.recon_matrix.x, h.encoding.recon_matrix.y, h.encoding.recon_matrix.z
    );
    println!(
        "Encoded FOV   : {:.1} x {:.1} x {:.1} mm",
        h.encoding.encoded_fov.x, h.encoding.encoded_fov.y, h.encoding.encoded_fov.z
    );
    println!(
        "ky range      : [{}, {}] centre {}",
        h.encoding.ky_limit.minimum, h.encoding.ky_limit.maximum, h.encoding.ky_limit.center
    );
    println!(
        "Slices        : [{}, {}]",
        h.encoding.slice_limit.minimum, h.encoding.slice_limit.maximum
    );
    Ok(())
}

fn cmd_info_fastmri(path: &PathBuf, json: bool) -> Result<()> {
    let f = FastmriFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let m = &f.meta;
    let h = &m.header;

    if json {
        let v = json!({
            "format": "FastMRI",
            "file": path.display().to_string(),
            "acquisition": m.acquisition,
            "patient_id": m.patient_id,
            "n_slices": m.n_slices,
            "n_coils": m.n_coils,
            "encoded_matrix": { "kx": m.n_kx, "ky": m.n_ky },
            "recon_matrix": { "x": m.recon_x, "y": m.recon_y },
            "vendor": h.system_vendor,
            "model": h.system_model,
            "field_strength_t": h.field_strength_t,
            "trajectory": h.encoding.trajectory
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }

    println!("Format        : FastMRI");
    println!("File          : {}", path.display());
    println!("Acquisition   : {}", m.acquisition);
    println!("Patient ID    : {}", m.patient_id);
    println!("Slices        : {}", m.n_slices);
    println!("Coils         : {}", m.n_coils);
    println!("Encoded matrix: {} x {}", m.n_kx, m.n_ky);
    println!("Recon matrix  : {} x {}", m.recon_x, m.recon_y);
    println!("Vendor        : {}", h.system_vendor);
    println!("Model         : {}", h.system_model);
    println!("Field [T]     : {:.3}", h.field_strength_t);
    println!("Trajectory    : {}", h.encoding.trajectory);
    Ok(())
}

fn cmd_probe(path: &PathBuf) -> Result<()> {
    match detect_format(path)? {
        FileFormat::FastMri => {
            let f = FastmriFile::open(path)
                .with_context(|| format!("opening {}", path.display()))?;
            println!(
                "FastMRI file: pre-assembled [{slices}, {coils}, {ky}, {kx}] tensor -- \
                 no per-acquisition index to probe.",
                slices = f.meta.n_slices,
                coils  = f.meta.n_coils,
                ky     = f.meta.n_ky,
                kx     = f.meta.n_kx,
            );
            return Ok(());
        }
        FileFormat::Ismrmrd => {}
    }

    let f = IsmrmrdFile::open(path).with_context(|| format!("opening {}", path.display()))?;

    use std::collections::{BTreeMap, BTreeSet};
    let mut ky = BTreeSet::<u16>::new();
    let mut kz = BTreeSet::<u16>::new();
    let mut avg = BTreeSet::<u16>::new();
    let mut slc = BTreeSet::<u16>::new();
    let mut con = BTreeSet::<u16>::new();
    let mut pha = BTreeSet::<u16>::new();
    let mut rep = BTreeSet::<u16>::new();
    let mut set = BTreeSet::<u16>::new();
    let mut seg = BTreeSet::<u16>::new();
    // (slice, ky) -> count of acquisitions touching that line
    let mut hits: BTreeMap<(u16, u16), u32> = BTreeMap::new();
    // (slice, ky) -> BTreeSet of segment ids that touched it
    let mut seg_at: BTreeMap<(u16, u16), BTreeSet<u16>> = BTreeMap::new();
    let mut n_img = 0usize;
    let mut n_noise = 0usize;
    let mut n_other = 0usize;
    let mut n_reverse = 0usize;
    let mut flag_union: u64 = 0;
    let mut flag_intersect: u64 = !0u64;

    f.for_each(|a| {
        if a.header.is_noise() {
            n_noise += 1;
            return Ok(());
        }
        if !a.header.is_image_scan() {
            n_other += 1;
            return Ok(());
        }
        n_img += 1;
        if a.header.is_reverse() {
            n_reverse += 1;
        }
        flag_union |= a.header.flags;
        flag_intersect &= a.header.flags;
        let i = &a.header.idx;
        ky.insert(i.kspace_encode_step_1);
        kz.insert(i.kspace_encode_step_2);
        avg.insert(i.average);
        slc.insert(i.slice);
        con.insert(i.contrast);
        pha.insert(i.phase);
        rep.insert(i.repetition);
        set.insert(i.set);
        seg.insert(i.segment);
        *hits.entry((i.slice, i.kspace_encode_step_1)).or_insert(0) += 1;
        seg_at
            .entry((i.slice, i.kspace_encode_step_1))
            .or_default()
            .insert(i.segment);
        Ok(())
    })?;

    let fmt = |s: &BTreeSet<u16>| {
        if s.is_empty() {
            "(none)".into()
        } else {
            format!(
                "{} unique: [{}..{}]",
                s.len(),
                s.iter().next().unwrap(),
                s.iter().next_back().unwrap()
            )
        }
    };

    println!("Image acqs        : {}", n_img);
    println!("Noise acqs        : {}", n_noise);
    println!("Other non-image   : {}", n_other);
    println!("Reverse readouts  : {}", n_reverse);
    println!(
        "Flag bits always set over image acqs : {:064b}",
        flag_intersect
    );
    println!("Flag bits ever   set over image acqs : {:064b}", flag_union);
    println!("kspace_encode_1 ky: {}", fmt(&ky));
    println!("kspace_encode_2 kz: {}", fmt(&kz));
    println!("idx.slice         : {}", fmt(&slc));
    println!("idx.average       : {}", fmt(&avg));
    println!("idx.contrast      : {}", fmt(&con));
    println!("idx.phase         : {}", fmt(&pha));
    println!("idx.repetition    : {}", fmt(&rep));
    println!("idx.set           : {}", fmt(&set));
    println!("idx.segment       : {}", fmt(&seg));

    // Hit-count histogram over (slice, ky) pairs
    let mut hist: BTreeMap<u32, u32> = BTreeMap::new();
    for &c in hits.values() {
        *hist.entry(c).or_insert(0) += 1;
    }
    println!("\nHits per (slice, ky):");
    for (c, n) in &hist {
        println!("  {:>3} hits: {:>6} lines", c, n);
    }

    // Segment-ID histogram
    if seg.len() > 1 {
        let mut seg_hist: BTreeMap<Vec<u16>, u32> = BTreeMap::new();
        for s in seg_at.values() {
            let key: Vec<u16> = s.iter().copied().collect();
            *seg_hist.entry(key).or_insert(0) += 1;
        }
        println!("\nSegment-id sets covering each (slice, ky):");
        for (k, n) in &seg_hist {
            println!("  segments {:?}: {:>6} lines", k, n);
        }
    }

    Ok(())
}

fn cmd_recon(
    path: &PathBuf,
    out_dir: &PathBuf,
    slice_sel: Option<usize>,
    pct_low: f32,
    pct_high: f32,
    no_crop: bool,
    no_prewhiten: bool,
    no_phasecorr: bool,
    no_oversampling_removal: bool,
    no_partial_fourier: bool,
    fft_mode: FftMode,
    strategy_arg: StrategyArg,
    grappa_kernel_ky: usize,
    grappa_kernel_kx: usize,
    grappa_ridge: f32,
    sense_map_source: SenseMapSource,
    sense_walsh_window: usize,
    sense_walsh_iters: usize,
    espirit_kernel: usize,
    espirit_threshold: f32,
    espirit_iters: usize,
    sense_ridge: f32,
    sense_gfactor: bool,
    write_gfactor: bool,
    cs_iters: usize,
    cs_lambda: f32,
    format: OutputFormat,
    verbose: u8,
) -> Result<()> {
    if !(0.0..100.0).contains(&pct_low) || !(0.0..=100.0).contains(&pct_high) || pct_high <= pct_low
    {
        bail!("invalid percentile window: [{pct_low}, {pct_high}]");
    }

    match detect_format(path)? {
        FileFormat::FastMri => {
            // Calibration flags have no effect on FastMRI files (fully-sampled
            // tensor with pre-computed RSS reference); warn so users know.
            if no_prewhiten || no_phasecorr || no_oversampling_removal || no_partial_fourier {
                warn!(
                    "FastMRI file detected: --no-prewhiten / --no-phasecorr / \
                     --no-oversampling-removal / --no-partial-fourier have no effect \
                     (FastMRI tensors do not carry calibration data)."
                );
            }
            return cmd_recon_fastmri(
                path, out_dir, slice_sel, pct_low, pct_high, no_crop,
                strategy_arg, format, verbose,
            );
        }
        FileFormat::Ismrmrd => {}
    }

    let f = IsmrmrdFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    if f.header.encoding.trajectory != "cartesian" {
        bail!(
            "only cartesian reconstruction is implemented (got {})",
            f.header.encoding.trajectory
        );
    }

    let spinner = if verbose == 0 {
        let b = ProgressBar::new_spinner();
        b.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap(),
        );
        b.enable_steady_tick(Duration::from_millis(100));
        b.set_message("Reconstructing...");
        Some(b)
    } else {
        None
    };

    let volume = match strategy_arg {
        StrategyArg::IfftRss => {
            let strategy = IfftRss {
                remove_oversampling: !no_oversampling_removal,
                prewhiten: !no_prewhiten,
                phase_correct: !no_phasecorr,
                partial_fourier: !no_partial_fourier,
                fft_mode,
                crop_to_recon_matrix: !no_crop,
            };
            info!(
                "Strategy: {} (oversampling_removal={}, prewhiten={}, phasecorr={}, partial_fourier={}, fft={:?})",
                strategy.name(),
                strategy.remove_oversampling,
                strategy.prewhiten,
                strategy.phase_correct,
                strategy.partial_fourier,
                strategy.fft_mode,
            );
            strategy
                .reconstruct(&f)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .context("reconstruction")?
        }
        StrategyArg::Grappa => {
            let strategy = GrappaRss {
                remove_oversampling: !no_oversampling_removal,
                prewhiten: !no_prewhiten,
                phase_correct: !no_phasecorr,
                kernel_ky: grappa_kernel_ky,
                kernel_kx: grappa_kernel_kx,
                ridge: grappa_ridge,
                fft_mode,
                crop_to_recon_matrix: !no_crop,
            };
            info!(
                "Strategy: {} (oversampling_removal={}, prewhiten={}, phasecorr={}, \
                 kernel={}x{}, ridge={}, fft={:?})",
                strategy.name(),
                strategy.remove_oversampling,
                strategy.prewhiten,
                strategy.phase_correct,
                strategy.kernel_ky,
                strategy.kernel_kx,
                strategy.ridge,
                strategy.fft_mode,
            );
            strategy
                .reconstruct(&f)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .context("reconstruction")?
        }
        StrategyArg::Sense => {
            let strategy = SenseRss {
                remove_oversampling: !no_oversampling_removal,
                prewhiten: !no_prewhiten,
                phase_correct: !no_phasecorr,
                map_source: sense_map_source,
                walsh_window: sense_walsh_window,
                walsh_iters: sense_walsh_iters,
                espirit_kernel,
                espirit_threshold,
                espirit_iters,
                ridge: sense_ridge,
                compute_gfactor: sense_gfactor || write_gfactor,
                fft_mode,
                crop_to_recon_matrix: !no_crop,
            };
            info!(
                "Strategy: {} (oversampling_removal={}, prewhiten={}, phasecorr={}, \
                 map_source={:?}, ridge={}, fft={:?})",
                strategy.name(),
                strategy.remove_oversampling,
                strategy.prewhiten,
                strategy.phase_correct,
                strategy.map_source,
                strategy.ridge,
                strategy.fft_mode,
            );
            strategy
                .reconstruct(&f)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .context("reconstruction")?
        }
        StrategyArg::Cs => {
            let strategy = CsRss {
                remove_oversampling: !no_oversampling_removal,
                prewhiten: !no_prewhiten,
                phase_correct: !no_phasecorr,
                iters: cs_iters,
                lambda: cs_lambda,
                fft_mode,
                crop_to_recon_matrix: !no_crop,
            };
            info!(
                "Strategy: {} (oversampling_removal={}, prewhiten={}, phasecorr={}, \
                 iters={}, lambda={:.3e}, fft={:?})",
                strategy.name(),
                strategy.remove_oversampling,
                strategy.prewhiten,
                strategy.phase_correct,
                strategy.iters,
                strategy.lambda,
                strategy.fft_mode,
            );
            strategy
                .reconstruct(&f)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .context("reconstruction")?
        }
    };
    if let Some(b) = spinner {
        b.finish_and_clear();
    }
    let magnitude = volume.data;
    let gfactor = volume.gfactor;

    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "recon".into());

    let file_out_dir = out_dir.join(&stem);
    std::fs::create_dir_all(&file_out_dir)
        .with_context(|| format!("creating {}", file_out_dir.display()))?;

    let (nz, ny, nx) = magnitude.dim();
    info!("Image volume: {} slices x {}x{}", nz, ny, nx);

    let slices: Vec<usize> = match slice_sel {
        Some(s) if s < nz => vec![s],
        Some(s) => bail!("slice {s} out of range (0..{nz})"),
        None => (0..nz).collect(),
    };

    // ── NIfTI output ─────────────────────────────────────────────────────
    if matches!(format, OutputFormat::Nifti | OutputFormat::Both) {
        let nii_vol: ndarray::Array3<f32> = if slices.len() == nz {
            magnitude.view().to_owned()
        } else {
            let mut flat = Vec::with_capacity(slices.len() * ny * nx);
            for &s in &slices {
                flat.extend(magnitude.index_axis(Axis(0), s).iter().copied());
            }
            ndarray::Array3::from_shape_vec((slices.len(), ny, nx), flat)
                .context("building NIfTI sub-volume")?
        };
        let nii_path = file_out_dir.join(format!("{stem}.nii"));
        write_nifti_volume(&nii_vol, &nii_path)
            .with_context(|| format!("writing {}", nii_path.display()))?;
        info!("Wrote {} (NIfTI)", nii_path.display());
    }

    // ── PNG output ────────────────────────────────────────────────────────
    let write_pngs =
        matches!(format, OutputFormat::Png | OutputFormat::Both) || write_gfactor;
    if write_pngs {
        let write_pb = if verbose == 0 && slices.len() > 1 {
            let b = ProgressBar::new(slices.len() as u64);
            b.set_style(
                ProgressStyle::default_bar()
                    .template("[{bar:40.cyan/blue}] {pos}/{len} slices  {elapsed_precise}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            Some(b)
        } else {
            None
        };

        for iz in &slices {
            let iz = *iz;
            let slice: Array2<f32> = magnitude.index_axis(Axis(0), iz).to_owned();

            if verbose >= 1 {
                let (mn, mx, mean) = slice_stats(&slice);
                info!("Slice {iz}: min={mn:.3e}  max={mx:.3e}  mean={mean:.3e}");
            }

            if matches!(format, OutputFormat::Png | OutputFormat::Both) {
                let png_path = file_out_dir.join(format!("slice_{iz:04}.png"));
                write_png_windowed(&slice, &png_path, pct_low, pct_high)
                    .with_context(|| format!("writing {}", png_path.display()))?;
                if verbose >= 1 {
                    info!("Wrote {}", png_path.display());
                }
            }

            if write_gfactor {
                match gfactor.as_ref() {
                    Some(gv) => {
                        let gslice: Array2<f32> = gv.index_axis(Axis(0), iz).to_owned();
                        let gpath =
                            file_out_dir.join(format!("gfactor_slice_{iz:04}.png"));
                        let mut sorted: Vec<f32> = gslice.iter().copied().collect();
                        sorted.sort_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let hi = percentile(&sorted, 99.0).max(2.0);
                        write_png_linear(&gslice, &gpath, 1.0, hi)
                            .with_context(|| format!("writing {}", gpath.display()))?;
                        if verbose >= 1 {
                            info!(
                                "Wrote {} (g-factor, window [1.0, {:.2}])",
                                gpath.display(),
                                hi
                            );
                        }
                    }
                    None => {
                        warn!("--write-gfactor requested but strategy produced no g-factor map");
                    }
                }
            }

            if let Some(ref b) = write_pb {
                b.inc(1);
            }
        }

        if let Some(b) = write_pb {
            b.finish_and_clear();
        }
    }

    Ok(())
}

fn cmd_recon_fastmri(
    path: &PathBuf,
    out_dir: &PathBuf,
    slice_sel: Option<usize>,
    pct_low: f32,
    pct_high: f32,
    no_crop: bool,
    strategy_arg: StrategyArg,
    format: OutputFormat,
    verbose: u8,
) -> Result<()> {
    if !matches!(strategy_arg, StrategyArg::IfftRss) {
        warn!(
            "Strategy {:?} is not supported for FastMRI files (fully-sampled tensors). \
             Falling back to ifft-rss.",
            strategy_arg
        );
    }

    let f = FastmriFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let m = &f.meta;

    info!(
        "FastMRI recon: {} slices, {} coils, ky={}, kx={}",
        m.n_slices, m.n_coils, m.n_ky, m.n_kx
    );

    let spinner = if verbose == 0 {
        let b = ProgressBar::new_spinner();
        b.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap(),
        );
        b.enable_steady_tick(Duration::from_millis(100));
        b.set_message("Reading k-space and reconstructing...");
        Some(b)
    } else {
        None
    };

    let mut kspace = f
        .read_kspace()
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("reading kspace")?;

    ifft2_inplace(&mut kspace, (2, 3));
    let mut magnitude = rss_combine_4d(&kspace);

    if !no_crop && (m.recon_y != m.n_ky || m.recon_x != m.n_kx) {
        magnitude = center_crop_3d(&magnitude, (m.n_slices, m.recon_y, m.recon_x));
    }

    if let Some(b) = spinner {
        b.finish_and_clear();
    }

    let (nz, ny, nx) = magnitude.dim();
    info!("Image volume: {} slices x {}x{}", nz, ny, nx);

    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "recon".into());
    let file_out_dir = out_dir.join(&stem);
    std::fs::create_dir_all(&file_out_dir)
        .with_context(|| format!("creating {}", file_out_dir.display()))?;

    let slices: Vec<usize> = match slice_sel {
        Some(s) if s < nz => vec![s],
        Some(s) => bail!("slice {s} out of range (0..{nz})"),
        None => (0..nz).collect(),
    };

    // ── NIfTI output ─────────────────────────────────────────────────────
    if matches!(format, OutputFormat::Nifti | OutputFormat::Both) {
        let nii_vol: ndarray::Array3<f32> = if slices.len() == nz {
            magnitude.view().to_owned()
        } else {
            let mut flat = Vec::with_capacity(slices.len() * ny * nx);
            for &s in &slices {
                flat.extend(magnitude.index_axis(Axis(0), s).iter().copied());
            }
            ndarray::Array3::from_shape_vec((slices.len(), ny, nx), flat)
                .context("building NIfTI sub-volume")?
        };
        let nii_path = file_out_dir.join(format!("{stem}.nii"));
        write_nifti_volume(&nii_vol, &nii_path)
            .with_context(|| format!("writing {}", nii_path.display()))?;
        info!("Wrote {} (NIfTI)", nii_path.display());
    }

    // ── PNG output ────────────────────────────────────────────────────────
    if matches!(format, OutputFormat::Png | OutputFormat::Both) {
        let write_pb = if verbose == 0 && slices.len() > 1 {
            let b = ProgressBar::new(slices.len() as u64);
            b.set_style(
                ProgressStyle::default_bar()
                    .template("[{bar:40.cyan/blue}] {pos}/{len} slices  {elapsed_precise}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            Some(b)
        } else {
            None
        };

        for iz in &slices {
            let iz = *iz;
            let slice: Array2<f32> = magnitude.index_axis(Axis(0), iz).to_owned();
            let png_path = file_out_dir.join(format!("slice_{iz:04}.png"));
            write_png_windowed(&slice, &png_path, pct_low, pct_high)
                .with_context(|| format!("writing {}", png_path.display()))?;
            if let Some(ref b) = write_pb {
                b.inc(1);
            } else {
                info!("Wrote {}", png_path.display());
            }
        }

        if let Some(b) = write_pb {
            b.finish_and_clear();
        }
    }

    Ok(())
}

/// Write an `Array3<f32>` as a NIfTI-1 single-file volume (.nii).
///
/// The input array has shape `[nz, ny, nx]` in C order (x varies fastest in
/// memory), which maps to NIfTI dim = [3, nx, ny, nz] (Fortran / x-fastest).
/// Voxel sizes are written as 1.0 mm isotropic; no spatial transform is set.
fn write_nifti_volume(vol: &ndarray::Array3<f32>, path: &std::path::Path) -> Result<()> {
    let (nz, ny, nx) = vol.dim();

    // 348-byte NIfTI-1 header, then 4-byte extension block, then data.
    let mut hdr = [0u8; 348];

    // sizeof_hdr = 348
    hdr[0..4].copy_from_slice(&348i32.to_le_bytes());
    // dim[0..7] at byte 40: [ndims, nx, ny, nz, 1, 1, 1, 1]
    let nx16 = i16::try_from(nx).with_context(|| format!("NIfTI nx={nx} exceeds i16::MAX"))?;
    let ny16 = i16::try_from(ny).with_context(|| format!("NIfTI ny={ny} exceeds i16::MAX"))?;
    let nz16 = i16::try_from(nz).with_context(|| format!("NIfTI nz={nz} exceeds i16::MAX"))?;
    let dims: [i16; 8] = [3, nx16, ny16, nz16, 1, 1, 1, 1];
    for (i, d) in dims.iter().enumerate() {
        hdr[40 + i * 2..40 + i * 2 + 2].copy_from_slice(&d.to_le_bytes());
    }
    // datatype = 16 (DT_FLOAT32) at byte 70
    hdr[70..72].copy_from_slice(&16i16.to_le_bytes());
    // bitpix = 32 at byte 72
    hdr[72..74].copy_from_slice(&32i16.to_le_bytes());
    // pixdim[0..7] at byte 76: qfac=1, voxel sizes = 1 mm
    let pixdim: [f32; 8] = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    for (i, p) in pixdim.iter().enumerate() {
        hdr[76 + i * 4..76 + i * 4 + 4].copy_from_slice(&p.to_le_bytes());
    }
    // vox_offset = 352.0 (header + extension block) at byte 108
    hdr[108..112].copy_from_slice(&352.0f32.to_le_bytes());
    // scl_slope = 1.0 at byte 112
    hdr[112..116].copy_from_slice(&1.0f32.to_le_bytes());
    // magic = "n+1\0" at byte 344
    hdr[344..348].copy_from_slice(b"n+1\0");

    use std::io::Write;
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    file.write_all(&hdr)?;
    file.write_all(&[0u8; 4])?; // no extensions
    for &v in vol.iter() {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Compute (min, max, mean) of a 2-D slice for diagnostic logging.
fn slice_stats(slice: &Array2<f32>) -> (f32, f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let n = slice.len();
    for &v in slice.iter() {
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v as f64;
    }
    (min, max, (sum / n as f64) as f32)
}

/// Write a 2D f32 array to PNG with a fixed linear window `[lo, hi]`.
fn write_png_linear(img: &Array2<f32>, path: &std::path::Path, lo: f32, hi: f32) -> Result<()> {
    let (h, w) = img.dim();
    let hi = hi.max(lo + f32::EPSILON);
    let mut buf = vec![0u8; h * w];
    for y in 0..h {
        for x in 0..w {
            let v = img[[y, x]];
            let norm = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            let byte = (norm * 255.0).round() as u8;
            buf[slice_index(w, y, x)] = byte;
        }
    }
    image::save_buffer(path, &buf, w as u32, h as u32, image::ColorType::L8)?;
    Ok(())
}

/// Write a 2D f32 array to PNG with percentile windowing.
fn write_png_windowed(
    img: &Array2<f32>,
    path: &std::path::Path,
    pct_low: f32,
    pct_high: f32,
) -> Result<()> {
    let (h, w) = img.dim();

    // Gather magnitudes, compute percentile bounds.
    let mut vals: Vec<f32> = img.iter().copied().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lo = percentile(&vals, pct_low);
    let hi = percentile(&vals, pct_high).max(lo + f32::EPSILON);

    let mut buf = vec![0u8; h * w];
    for y in 0..h {
        for x in 0..w {
            let v = img[[y, x]];
            let norm = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            // Slight gamma for perceptual brightness.
            let byte = (norm.powf(0.9) * 255.0).round() as u8;
            buf[slice_index(w, y, x)] = byte;
        }
    }

    image::save_buffer(path, &buf, w as u32, h as u32, image::ColorType::L8)?;
    Ok(())
}

#[inline]
fn slice_index(w: usize, y: usize, x: usize) -> usize {
    y * w + x
}

fn percentile(sorted: &[f32], pct: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let k = ((pct / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[k.min(sorted.len() - 1)]
}
