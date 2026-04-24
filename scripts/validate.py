#!/usr/bin/env python3
"""OpenKSpace validation harness.

Runs the OpenKSpace `openkspace recon` CLI on one or more ISMRMRD files
and compares the resulting PNG against a numpy reference reconstruction
using SSIM.

Usage:
    python scripts/validate.py <file.h5>            # single file
    python scripts/validate.py <dir>                # batch: all *.h5 under dir
    python scripts/validate.py <f1.h5> <f2.h5> ...  # explicit list

Exits 0 when every file meets the SSIM threshold, 1 when any file fails,
and 2 on invocation errors. In batch mode a summary table is printed at
the end and optionally written to JSON via --report.

Dependencies: numpy, h5py, pillow, scikit-image.
Install with: pip install numpy h5py pillow scikit-image
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ISMRMRD AcquisitionFlags (1-based bit positions, mirrored from Rust).
_FLAG_NOISE = 1 << (19 - 1)
_FLAG_CALIB = 1 << (20 - 1)
_FLAG_CALIB_IMG = 1 << (21 - 1)
_FLAG_REVERSE = 1 << (22 - 1)
_FLAG_NAVIGATION = 1 << (23 - 1)
_FLAG_PHASECORR = 1 << (24 - 1)
_FLAG_HPFEEDBACK = 1 << (26 - 1)
_FLAG_DUMMY = 1 << (27 - 1)
_FLAG_RTFEEDBACK = 1 << (28 - 1)
_FLAG_SURFACE = 1 << (29 - 1)
_NON_IMAGE_MASK = (
    _FLAG_NOISE
    | _FLAG_PHASECORR
    | _FLAG_NAVIGATION
    | _FLAG_RTFEEDBACK
    | _FLAG_HPFEEDBACK
    | _FLAG_DUMMY
    | _FLAG_SURFACE
)


def _parse_xml(xml: str) -> dict:
    """Extract the subset of header fields the reference recon needs."""

    def _int(pat: str) -> int:
        m = re.search(pat, xml, re.S)
        if m is None:
            raise ValueError(f"xml: pattern not found: {pat}")
        return int(m.group(1))

    return {
        "enc_x": _int(r"<encodedSpace>.*?<matrixSize>.*?<x>(\d+)</x>"),
        "enc_y": _int(r"<encodedSpace>.*?<matrixSize>.*?<y>(\d+)</y>"),
        "enc_z": _int(r"<encodedSpace>.*?<matrixSize>.*?<z>(\d+)</z>"),
        "rec_x": _int(r"<reconSpace>.*?<matrixSize>.*?<x>(\d+)</x>"),
        "rec_y": _int(r"<reconSpace>.*?<matrixSize>.*?<y>(\d+)</y>"),
        "ky_center": _int(
            r"<kspace_encoding_step_1>\s*<minimum>\d+</minimum>"
            r"\s*<maximum>\d+</maximum>\s*<center>(\d+)"
        ),
    }


def ref_recon(h5_path: str, slice_idx: int) -> np.ndarray:
    """Numpy reference: IFFT + RSS for a single slice. Returns magnitude float32."""
    with h5py.File(h5_path, "r") as f:
        d = f["dataset/data"]
        xml = f["dataset/xml"][()][0].decode()
        hdr = _parse_xml(xml)

        mx, my = hdr["enc_x"], hdr["enc_y"]
        rx, ry = hdr["rec_x"], hdr["rec_y"]
        ky_center = hdr["ky_center"]

        nc = None
        kspace = None
        filled = None
        ky_shift = my // 2 - ky_center

        for i in range(len(d)):
            row = d[i]
            h = row["head"]
            flags = int(h["flags"])
            if flags & _NON_IMAGE_MASK:
                continue
            if int(h["idx"]["slice"]) != slice_idx:
                continue

            ns = int(h["number_of_samples"])
            if nc is None:
                nc = int(h["active_channels"])
                kspace = np.zeros((nc, my, mx), dtype=np.complex64)
                filled = np.zeros((my, mx), dtype=bool)

            ky = int(h["idx"]["kspace_encode_step_1"]) + ky_shift
            if not (0 <= ky < my):
                continue

            cx = np.array(row["data"]).reshape(nc, ns, 2)
            cx = cx[..., 0] + 1j * cx[..., 1]

            cs = int(h["center_sample"])
            dst_off = mx // 2 - cs if cs > 0 else (mx - ns) // 2
            end = dst_off + ns
            if dst_off < 0 or end > mx:
                continue

            # First-wins collision at the centre of the readout.
            ctr_x = dst_off + ns // 2
            if filled[ky, ctr_x]:
                continue

            if flags & _FLAG_REVERSE:
                cx = cx[:, ::-1]

            kspace[:, ky, dst_off:end] = cx
            filled[ky, dst_off:end] = True

    if kspace is None:
        raise RuntimeError(f"no image acquisitions found for slice {slice_idx}")

    img = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1))),
        axes=(-2, -1),
    )
    rss = np.sqrt((np.abs(img) ** 2).sum(axis=0))

    # Crop to the recon FOV (clamped to input size to match openkspace).
    ty = min(ry, my) if ry >= 1 else my
    tx = min(rx, mx) if rx >= 1 else mx
    y0 = (my - ty) // 2
    x0 = (mx - tx) // 2
    return rss[y0 : y0 + ty, x0 : x0 + tx].astype(np.float32)


def apply_window(img: np.ndarray, pct_low: float, pct_high: float) -> np.ndarray:
    """Mirror openkspace's percentile windowing and gamma."""
    vals = img.ravel()
    lo = np.percentile(vals, pct_low)
    hi = max(np.percentile(vals, pct_high), lo + 1e-9)
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return np.power(norm, 0.9)


def run_openkspace(
    binary: str, h5_path: str, slice_idx: int, out_dir: str
) -> np.ndarray:
    """Invoke the openkspace CLI and load the resulting PNG back as float32.

    We pass `--no-prewhiten --no-phasecorr --no-oversampling-removal` so
    the Rust pipeline matches the naive numpy reference (plain IFFT+RSS
    with post-IFFT recon-matrix crop).
    """
    cmd = [
        binary,
        "recon",
        h5_path,
        "--out",
        out_dir,
        "--slice",
        str(slice_idx),
        "--no-prewhiten",
        "--no-phasecorr",
        "--no-oversampling-removal",
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    pngs = sorted(p for p in os.listdir(out_dir) if p.endswith(".png"))
    if not pngs:
        raise RuntimeError(f"openkspace produced no PNGs in {out_dir}")
    im = np.array(Image.open(os.path.join(out_dir, pngs[0])).convert("L"))
    return im.astype(np.float32) / 255.0


def validate_one(
    h5_path: str,
    slice_idx: int,
    threshold: float,
    pct_low: float,
    pct_high: float,
    binary: str,
    verbose: bool,
) -> dict:
    """Run the full validation for one file. Returns a result dict."""
    t0 = time.monotonic()
    result = {
        "file": h5_path,
        "slice": slice_idx,
        "threshold": threshold,
        "ssim": None,
        "status": "ERROR",
        "error": None,
        "elapsed_s": 0.0,
    }
    try:
        if verbose:
            print(f"  reference recon (numpy) ...", flush=True)
        ref = ref_recon(h5_path, slice_idx)
        ref_w = apply_window(ref, pct_low, pct_high)

        if verbose:
            print(f"  openkspace recon (Rust) ...", flush=True)
        with tempfile.TemporaryDirectory() as td:
            ours = run_openkspace(binary, h5_path, slice_idx, td)

        if ref_w.shape != ours.shape:
            h = min(ref_w.shape[0], ours.shape[0])
            w = min(ref_w.shape[1], ours.shape[1])
            ref_w = ref_w[:h, :w]
            ours = ours[:h, :w]

        score = float(ssim(ref_w, ours, data_range=1.0))
        result["ssim"] = score
        result["status"] = "PASS" if score >= threshold else "FAIL"
    except subprocess.CalledProcessError as e:
        result["error"] = f"openkspace failed: rc={e.returncode}"
    except Exception as e:  # noqa: BLE001 - report anything the harness hits
        result["error"] = f"{type(e).__name__}: {e}"
    result["elapsed_s"] = time.monotonic() - t0
    return result


def _discover_inputs(paths: list[str]) -> list[str]:
    """Expand a mix of files and directories into a deduplicated file list."""
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "**", "*.h5"), recursive=True))
            for f in found:
                if f not in seen:
                    seen.add(f)
                    out.append(f)
        elif os.path.isfile(p):
            if p not in seen:
                seen.add(p)
                out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def _format_summary(results: list[dict], threshold: float) -> str:
    lines: list[str] = []
    name_w = max((len(os.path.basename(r["file"])) for r in results), default=4)
    name_w = max(name_w, 20)
    header = f"{'file':<{name_w}}  {'slice':>5}  {'ssim':>7}  {'status':>6}  {'time':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        ssim_str = f"{r['ssim']:.4f}" if r["ssim"] is not None else "  n/a"
        status = r["status"]
        lines.append(
            f"{os.path.basename(r['file']):<{name_w}}  "
            f"{r['slice']:>5}  {ssim_str:>7}  {status:>6}  {r['elapsed_s']:>6.1f}s"
        )
    n = len(results)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_err = sum(1 for r in results if r["status"] == "ERROR")
    lines.append("-" * len(header))
    lines.append(
        f"{n} file(s): {n_pass} pass, {n_fail} fail, {n_err} error "
        f"(threshold {threshold:.4f})"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate OpenKSpace against a numpy reference."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="one or more ISMRMRD .h5 files, or directories to recurse into",
    )
    ap.add_argument("--slice", type=int, default=0, help="slice index to compare")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="minimum acceptable SSIM (default: 0.95)",
    )
    ap.add_argument(
        "--pct-low", type=float, default=0.5, help="low percentile for windowing"
    )
    ap.add_argument(
        "--pct-high", type=float, default=99.5, help="high percentile for windowing"
    )
    ap.add_argument(
        "--binary",
        default="./target/release/openkspace",
        help="path to the openkspace CLI binary",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="optional path to write a JSON report of per-file results",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="in batch mode, continue after a FAIL/ERROR (default: true)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        print(
            f"error: openkspace binary not found at {args.binary} "
            "(run `cargo build --release` first)",
            file=sys.stderr,
        )
        return 2

    try:
        files = _discover_inputs(args.inputs)
    except FileNotFoundError as e:
        print(f"error: no such file or directory: {e}", file=sys.stderr)
        return 2
    if not files:
        print("error: no .h5 files found in inputs", file=sys.stderr)
        return 2

    batch = len(files) > 1 or os.path.isdir(args.inputs[0])
    results: list[dict] = []
    for i, path in enumerate(files, 1):
        if batch:
            print(f"[{i}/{len(files)}] {path}", flush=True)
        r = validate_one(
            path,
            args.slice,
            args.threshold,
            args.pct_low,
            args.pct_high,
            args.binary,
            verbose=not batch,
        )
        results.append(r)
        if batch:
            ssim_str = f"{r['ssim']:.4f}" if r["ssim"] is not None else "n/a"
            err = f" ({r['error']})" if r["error"] else ""
            print(f"    -> {r['status']}  ssim={ssim_str}{err}", flush=True)
        else:
            if r["ssim"] is not None:
                print(f"SSIM                            : {r['ssim']:.4f}")
            print(f"Threshold                       : {args.threshold:.4f}")
            print(f"File                            : {os.path.basename(r['file'])}")
            if r["error"]:
                print(f"ERROR                           : {r['error']}")
            print(f"RESULT                          : {r['status']}")

    if batch:
        print()
        print(_format_summary(results, args.threshold))

    if args.report:
        with open(args.report, "w") as f:
            json.dump(
                {"threshold": args.threshold, "results": results}, f, indent=2
            )
        print(f"wrote JSON report to {args.report}")

    # Exit non-zero on any FAIL or ERROR.
    if any(r["status"] != "PASS" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
