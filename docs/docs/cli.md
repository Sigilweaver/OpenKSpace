---
title: CLI reference
sidebar_label: CLI reference
---

# CLI reference

```text
openkspace info  <file.h5>                   # print header metadata
openkspace probe <file.h5>                   # scan acquisition index ranges
openkspace recon <file.h5>                   # reconstruct all slices -> recon_out/
    [--out <dir>]                            # output directory (default: recon_out/)
    [--slice <N>]                            # reconstruct one slice only
    [--fft auto|2d|3d]                       # FFT mode (default: auto)
    [--strategy ifft-rss|grappa|sense|cs]    # reconstruction strategy
    [--no-prewhiten]                         # skip noise pre-whitening
    [--no-phasecorr]                         # skip navigator phase correction
    [--no-oversampling-removal]              # keep 2x kx samples
    [--no-partial-fourier]                   # skip homodyne
    [--no-crop]                              # keep full oversampled FOV
    [--pct-low <f>] [--pct-high <f>]         # contrast window percentiles
    [--format png|nifti|both]                # output format (default: png)

  GRAPPA:
    [--grappa-kernel-ky <k>] [--grappa-kernel-kx <k>]
    [--grappa-ridge <lam>]

  SENSE:
    [--sense-maps walsh|espirit]
    [--sense-walsh-window <w>] [--sense-walsh-iters <n>]
    [--espirit-kernel <k>] [--espirit-threshold <f>] [--espirit-iters <n>]
    [--sense-ridge <lam>]
    [--sense-gfactor]                        # compute g-factor
    [--write-gfactor]                        # also emit g-factor PNGs

  CS:
    [--cs-iters <n>] [--cs-lambda <lam>] [--cs-wavelet-levels <n>]
```

For source-of-truth defaults and behaviour, see the
[README](https://github.com/Sigilweaver/OpenKSpace/blob/main/README.md)
and the rustdoc on docs.rs.
