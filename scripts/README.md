# Validation scripts

Tools for validating the OpenKSpace reconstruction against reference
implementations. These are developer aids; they are not part of the
published crate.

## `validate.py`

Runs `openkspace recon` on an ISMRMRD file, computes a numpy reference
reconstruction on the same file, applies identical percentile windowing
to both, and reports the structural similarity index (SSIM).

### Requirements

```sh
pip install numpy h5py pillow scikit-image
```

### Usage

```sh
# via the wrapper (builds the release binary if needed)
./scripts/validate.sh path/to/file.h5 --slice 15

# or directly
python scripts/validate.py path/to/file.h5 --slice 15 --threshold 0.95
```

Exit codes:
- `0` - PASS (SSIM >= threshold)
- `1` - FAIL (SSIM < threshold)
- `2` - error (missing file, missing binary, etc.)

## `validate_all.sh`

Batch wrapper for CI. `validate.py` also accepts directories and
multiple files directly; this wrapper just sets `--binary` from the
`BINARY` environment variable (default `./target/release/openkspace`)
and forwards the rest.

### Usage

```sh
cargo build --release

# every .h5 under a directory (recursive)
./scripts/validate_all.sh corpus/MRIData-org/knee/siemens/fully_sampled \
    --slice 15 --threshold 0.95 --report report.json

# or an explicit list
./scripts/validate_all.sh a.h5 b.h5 c.h5 --slice 0
```

Behaviour:
- Batch mode prints a per-file progress line, then a summary table with
  SSIM, status, and wall-clock time.
- `--report path.json` optionally emits a machine-readable report.
- Exit code is `1` if any file is `FAIL` or `ERROR`; `0` if all pass.
