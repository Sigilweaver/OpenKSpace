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
