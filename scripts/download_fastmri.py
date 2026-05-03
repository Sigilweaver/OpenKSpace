#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx[http2]",
#   "rich",
# ]
# ///
"""
download_fastmri.py
-------------------
Download, extract, and verify fastMRI dataset archives.

Reads presigned S3 URLs from corpus/FastMRI.html (the access email from NYU).
Each archive is downloaded to corpus/FastMRI/.downloads/, extracted to corpus/FastMRI/,
then the archive is deleted.  State is tracked in corpus/FastMRI/status.json so
interrupted runs resume cleanly.

Presets:
  minimal    knee_multicoil_train_batch_0 + brain_multicoil_train_batch_0  (~190 GB)
  knee       all knee k-space batches (no DICOMs)                          (~640 GB)
  brain      all brain k-space batches (no DICOMs)                         (~1.4 TB)
  all-kspace knee + brain k-space, no DICOMs                               (~2.0 TB)
  dicoms     knee + brain + prostate + breast DICOMs only                  (~320 GB)
  all        everything                                                     (~6.3 TB)

Examples:
  uv run scripts/download_fastmri.py --preset minimal
  uv run scripts/download_fastmri.py --preset minimal --dry-run
  uv run scripts/download_fastmri.py --include knee_multicoil_val
  uv run scripts/download_fastmri.py --include 'knee_multicoil_train_batch_[012]'
  uv run scripts/download_fastmri.py --preset all-kspace --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

REPO_ROOT   = Path(__file__).resolve().parent.parent
HTML_FILE   = REPO_ROOT / "corpus" / "FastMRI.html"
DEST_ROOT   = REPO_ROOT / "corpus" / "FastMRI"
STATUS_FILE = DEST_ROOT / "status.json"
WORK_DIR    = DEST_ROOT / ".downloads"

console = Console()

# HDF5 magic bytes for post-extraction spot-check.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# Minimal preset: one knee batch + one brain batch.
_MINIMAL = {"knee_multicoil_train_batch_0", "brain_multicoil_train_batch_0"}


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

def _category(name: str) -> str:
    n = name.lower()
    if n.startswith("knee"):
        return "knee"
    if n.startswith("brain"):
        return "brain"
    if "prostate" in n:
        return "prostate"
    if "breast" in n:
        return "breast"
    return "other"


def _is_dicom(name: str) -> bool:
    n = name.lower()
    return "dicom" in n or n.endswith("_dcm") or "_dcm_" in n or "_dcm." in n


def _is_kspace(name: str) -> bool:
    """True when the archive likely contains raw k-space HDF5 (knee/brain only)."""
    return not _is_dicom(name) and _category(name) in ("knee", "brain")


def parse_html(path: Path) -> list[dict]:
    """
    Extract (name, url, size_gb) triples from the FastMRI access HTML.

    The HTML structure is:
        <a href="SIGNED_URL">NAME</a>\n        (~XX.X GB)<br />
    """
    text = path.read_text(encoding="utf-8")
    pat = re.compile(
        r'href="([^"]+)"[^>]*>([^<]+)</a>\s*\(~?([\d.]+)\s*(GB|MB|KB)\)',
        re.IGNORECASE,
    )
    entries = []
    for m in pat.finditer(text):
        url       = m.group(1)
        name      = m.group(2).strip()
        size_val  = float(m.group(3))
        unit      = m.group(4).upper()

        if not url.startswith("https://"):
            continue
        # Normalise whitespace (some link texts span multiple lines in the HTML).
        name = " ".join(name.split())
        if name in ("SHA256 Hash", "SHA256"):
            continue

        size_gb = (
            size_val / (1024 * 1024) if unit == "KB"
            else size_val / 1024     if unit == "MB"
            else size_val
        )
        entries.append({
            "name":      name,
            "url":       url,
            "size_gb":   size_gb,
            "category":  _category(name),
            "is_dicom":  _is_dicom(name),
            "is_kspace": _is_kspace(name),
        })
    return entries


# ---------------------------------------------------------------------------
# Preset filtering
# ---------------------------------------------------------------------------

def apply_preset(entries: list[dict], preset: str) -> list[dict]:
    if preset == "minimal":
        return [e for e in entries if e["name"] in _MINIMAL]
    if preset == "knee":
        return [e for e in entries if e["category"] == "knee" and not e["is_dicom"]]
    if preset == "brain":
        return [e for e in entries if e["category"] == "brain" and not e["is_dicom"]]
    if preset == "all-kspace":
        return [e for e in entries if e["is_kspace"]]
    if preset == "dicoms":
        return [e for e in entries if e["is_dicom"]]
    if preset == "all":
        return list(entries)
    raise ValueError(f"Unknown preset: {preset!r}")


# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------

def load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    return {}


def save_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_archive(entry: dict, dest: Path) -> None:
    """Stream-download with resume support and a rich progress bar."""
    url  = entry["url"]
    part = dest.with_suffix(dest.suffix + ".part")

    resume_from = part.stat().st_size if part.exists() else 0
    headers     = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with httpx.stream("GET", url, follow_redirects=True,
                      timeout=600, headers=headers) as r:
        if r.status_code == 416:
            # Server says range not satisfiable - file was already complete.
            part.rename(dest)
            return
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0)) + resume_from
        mode  = "ab" if resume_from else "wb"

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold]{entry['name']}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as bar:
            task = bar.add_task("dl", total=total or None)
            bar.update(task, advance=resume_from)
            with part.open(mode) as fh:
                for chunk in r.iter_bytes(chunk_size=1 << 17):  # 128 KB
                    fh.write(chunk)
                    bar.update(task, advance=len(chunk))

    part.rename(dest)


# ---------------------------------------------------------------------------
# Extract and verify
# ---------------------------------------------------------------------------

def extract_archive(archive: Path, dest: Path) -> list[Path]:
    """
    Extract archive to dest/.  Returns list of .h5 files found after extraction.
    tar validates CRC on the fly; a non-zero exit means the archive is corrupt.
    """
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(dest)],
        capture_output=True,
    )
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[:300]
        raise RuntimeError(f"tar extraction failed: {err}")
    return list(dest.rglob("*.h5"))


def spot_check_hdf5(h5_files: list[Path], n: int = 5) -> None:
    """
    Verify HDF5 magic bytes on up to n files.
    A cheap sanity check that doesn't require h5py.
    """
    for path in h5_files[:n]:
        with path.open("rb") as fh:
            magic = fh.read(8)
        if magic != _HDF5_MAGIC:
            raise RuntimeError(
                f"HDF5 magic mismatch in {path.name}: got {magic!r}"
            )


# ---------------------------------------------------------------------------
# Per-archive pipeline
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def process_one(entry: dict, status: dict, dry_run: bool) -> str:
    name  = entry["name"]
    state = status.get(name, {}).get("state", "pending")

    if state == "extracted":
        console.print(f"[green]skip[/green]  {name}  (already extracted)")
        return "skipped"

    url = entry["url"]
    ext = ".tar.xz" if ".tar.xz" in url else ".tar.gz"
    archive = WORK_DIR / (name + ext)

    if dry_run:
        console.print(
            f"[cyan]dry-run[/cyan]  {name}  "
            f"({entry['category']}, ~{entry['size_gb']:.1f} GB, "
            f"{'k-space' if entry['is_kspace'] else 'DICOM' if entry['is_dicom'] else 'other'})"
        )
        return "dry_run"

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Download -------------------------------------------------------
        if not archive.exists():
            console.print(
                f"\n[bold blue]Downloading[/bold blue]  {name}  "
                f"(~{entry['size_gb']:.1f} GB)"
            )
            status[name] = {"state": "downloading", "started": _ts()}
            save_status(status)
            download_archive(entry, archive)
            status[name].update({
                "state":      "downloaded",
                "size_bytes": archive.stat().st_size,
            })
            save_status(status)
        else:
            console.print(
                f"[dim]Archive already present, skipping download: {archive.name}[/dim]"
            )

        # 2. Extract --------------------------------------------------------
        console.print(f"  [bold]Extracting[/bold]  {archive.name}")
        status[name]["state"] = "extracting"
        save_status(status)

        h5_files = extract_archive(archive, DEST_ROOT)

        # 3. Spot-check HDF5 magic ------------------------------------------
        if h5_files:
            spot_check_hdf5(h5_files)

        # 4. Delete archive -------------------------------------------------
        archive.unlink()
        console.print(f"  [dim]Deleted archive[/dim]")

        status[name] = {
            "state":           "extracted",
            "extracted_files": len(h5_files),
            "completed":       _ts(),
            "category":        entry["category"],
            "size_gb":         entry["size_gb"],
        }
        save_status(status)
        console.print(
            f"  [green]Done[/green]  "
            f"{len(h5_files)} .h5 file{'s' if len(h5_files) != 1 else ''}"
        )
        return "ok"

    except Exception as exc:
        status[name] = {**status.get(name, {}), "state": "failed", "error": str(exc)}
        save_status(status)
        console.print(f"  [red]Failed[/red]: {exc}")
        # Leave partial archive in .downloads/ so the next run can resume.
        return "failed"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global DEST_ROOT, STATUS_FILE, WORK_DIR

    p = argparse.ArgumentParser(
        description="Download, extract, and verify fastMRI archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--html",
        default=str(HTML_FILE),
        help=f"Path to FastMRI.html (default: {HTML_FILE})",
    )
    p.add_argument(
        "--preset",
        choices=["minimal", "knee", "brain", "all-kspace", "dicoms", "all"],
        default="minimal",
        help="Dataset preset (default: minimal, ~190 GB)",
    )
    p.add_argument(
        "--include",
        metavar="PATTERN",
        help=(
            "Comma-separated glob patterns matched against archive names "
            "(overrides --preset)"
        ),
    )
    p.add_argument(
        "--dest",
        default=str(DEST_ROOT),
        help=f"Extraction root directory (default: {DEST_ROOT})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without fetching anything",
    )
    args = p.parse_args()

    DEST_ROOT   = Path(args.dest)
    STATUS_FILE = DEST_ROOT / "status.json"
    WORK_DIR    = DEST_ROOT / ".downloads"

    html_path = Path(args.html)
    if not html_path.exists():
        console.print(
            f"[red]HTML file not found:[/red] {html_path}\n"
            "Request a new download link at https://fastmri.med.nyu.edu/"
        )
        sys.exit(2)

    entries = parse_html(html_path)
    if not entries:
        console.print(
            "[red]No entries parsed from HTML.[/red]  "
            "The link format may have changed, or the file is empty."
        )
        sys.exit(2)

    # Apply filter.
    if args.include:
        patterns = [pat.strip() for pat in args.include.split(",")]
        entries = [e for e in entries if any(fnmatch.fnmatch(e["name"], pat) for pat in patterns)]
    else:
        entries = apply_preset(entries, args.preset)

    if not entries:
        console.print("[yellow]No archives match the selected filter.[/yellow]")
        sys.exit(0)

    # Summary table.
    table = Table(title="fastMRI Download Plan", show_header=True, header_style="bold")
    table.add_column("Archive")
    table.add_column("Category")
    table.add_column("Type")
    table.add_column("Size", justify="right")

    total_gb = 0.0
    for e in entries:
        kind     = "DICOM" if e["is_dicom"] else ("k-space" if e["is_kspace"] else "other")
        total_gb += e["size_gb"]
        table.add_row(e["name"], e["category"], kind, f"~{e['size_gb']:.1f} GB")
    table.add_row(
        "[bold]TOTAL[/bold]", "", "",
        f"[bold]~{total_gb:.1f} GB[/bold]",
    )
    console.print(table)

    if args.dry_run:
        return

    status  = load_status()
    counts: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0, "dry_run": 0}
    for entry in entries:
        result = process_one(entry, status, dry_run=False)
        counts[result] = counts.get(result, 0) + 1

    console.print(
        f"\n[bold]Done.[/bold]  "
        f"ok={counts['ok']}  "
        f"skipped={counts['skipped']}  "
        f"failed={counts['failed']}"
    )
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
