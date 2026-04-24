//! OpenKSpace command-line interface.
//!
//! Subcommands:
//!   info   -- print header metadata for an ISMRMRD .h5 file
//!   recon  -- run a cartesian IFFT + RSS reconstruction -> PNG(s)

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use ndarray::{Array2, Axis};
use openkspace_io::ismrmrd::IsmrmrdFile;
use openkspace_recon::{
    CsRss, FftMode, GrappaRss, IfftRss, ReconStrategy, SenseMapSource, SenseRss,
};
use std::path::PathBuf;
use tracing::info;

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
    /// Print ISMRMRD header metadata without loading k-space.
    Info {
        /// Path to .h5 file
        file: PathBuf,
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

        /// CS: number of FISTA iterations
        #[arg(long, default_value_t = 60)]
        cs_iters: usize,

        /// CS: L1-wavelet regularisation strength
        #[arg(long, default_value_t = 0.01)]
        cs_lambda: f32,
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

    match cli.cmd {
        Cmd::Info { file } => cmd_info(&file),
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
            cs_iters,
            cs_lambda,
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
            cs_iters,
            cs_lambda,
        ),
    }
}

fn cmd_info(path: &PathBuf) -> Result<()> {
    let f = IsmrmrdFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    let h = &f.header;

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

fn cmd_probe(path: &PathBuf) -> Result<()> {
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
    cs_iters: usize,
    cs_lambda: f32,
) -> Result<()> {
    if !(0.0..100.0).contains(&pct_low) || !(0.0..=100.0).contains(&pct_high) || pct_high <= pct_low
    {
        bail!("invalid percentile window: [{pct_low}, {pct_high}]");
    }

    let f = IsmrmrdFile::open(path).with_context(|| format!("opening {}", path.display()))?;
    if f.header.encoding.trajectory != "cartesian" {
        bail!(
            "only cartesian reconstruction is implemented (got {})",
            f.header.encoding.trajectory
        );
    }

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
    let magnitude = volume.data;

    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let (nz, ny, nx) = magnitude.dim();
    info!("Image volume: {} slices x {}x{}", nz, ny, nx);

    let slices: Vec<usize> = match slice_sel {
        Some(s) if s < nz => vec![s],
        Some(s) => bail!("slice {s} out of range (0..{nz})"),
        None => (0..nz).collect(),
    };

    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "recon".into());

    for iz in slices {
        let slice: Array2<f32> = magnitude.index_axis(Axis(0), iz).to_owned();

        // Diagnostic stats
        let (mn, mx, mean) = {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0f64;
            let n = slice.len();
            for &v in slice.iter() {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
                sum += v as f64;
            }
            (min, max, (sum / n as f64) as f32)
        };
        info!("Slice {iz}: min={mn:.3e}  max={mx:.3e}  mean={mean:.3e}");

        let png_path = out_dir.join(format!("{stem}_slice_{iz:04}.png"));
        write_png_windowed(&slice, &png_path, pct_low, pct_high)
            .with_context(|| format!("writing {}", png_path.display()))?;
        info!("Wrote {}", png_path.display());
    }

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
            buf[slice_index(w, h, y, x)] = byte;
        }
    }

    image::save_buffer(path, &buf, w as u32, h as u32, image::ColorType::L8)?;
    Ok(())
}

#[inline]
fn slice_index(w: usize, h: usize, y: usize, x: usize) -> usize {
    // Flip vertically so the image faces the conventional radiological orientation.
    let _ = h;
    y * w + x
}

fn percentile(sorted: &[f32], pct: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let k = ((pct / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[k.min(sorted.len() - 1)]
}
