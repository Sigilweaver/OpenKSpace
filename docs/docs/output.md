---
title: Output
sidebar_label: Output
---

# Output formats

Selectable via `--format png|nifti|both` (default `png`).

## PNG

8-bit greyscale PNG per slice with percentile contrast windowing.
The contrast window is set per-volume from the `[--pct-low,
--pct-high]` percentiles (default 1.0 / 99.0).

Filenames follow the pattern:

```
recon_out/slice_000.png
recon_out/slice_001.png
...
```

## NIfTI-1

Single-file `.nii` volume containing the magnitude reconstruction
in float32. The header includes voxel dimensions extracted from
the ISMRMRD `encodedSpace.fieldOfView_mm`. Suitable for downstream
tools such as FSL, AFNI, ITK-SNAP, or MITK.

```sh
openkspace recon scan.h5 --format nifti --out recon_out/
# -> recon_out/volume.nii
```

## g-factor maps (SENSE only)

When `--sense-gfactor` is set, the noise amplification map is
computed alongside the unfolded image. With `--write-gfactor`,
the map is also emitted as PNG using a linear window `[1, p99]`:

```
recon_out/gfactor_000.png
recon_out/gfactor_001.png
```
