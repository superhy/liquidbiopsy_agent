#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# -----------------------------------------------------------------------------
# Beginner Quickstart Notes (important for first run)
# -----------------------------------------------------------------------------
# 1) This dataset (GSE243474) is human data. The reference genome should be a
#    human assembly FASTA, typically:
#      - hg38 / GRCh38: newer human reference (recommended for new projects)
#      - hg19 / GRCh37: older human reference (legacy studies/tools)
# 2) BED coordinates must match the FASTA assembly. If BED is hg38 but FASTA is
#    hg19 (or the opposite), embeddings are biologically misaligned.
# 3) If you are not sure yet, start with hg38 FASTA and run smoke tests first.
#
# How to use beginner mode:
#   - Set RUN_PRESET_IN_MAIN = True
#   - Pick one case name in PRESET_CASE_IN_MAIN
#   - Fill PRESET_INPUT_DIR and PRESET_FASTA_PATH
#   - Run: python scripts/encode_bed_to_embedding.py
# -----------------------------------------------------------------------------
# CLI Cookbook (copy-paste examples)
# -----------------------------------------------------------------------------
# A) Inspect one compressed BED file only (no embedding run):
#    python scripts/encode_bed_to_embedding.py ^
#      --inspect_file "F:\liquid-agent-data\GSE243474\breast\GSM7787780_BC287K27R1.bed.gz" ^
#      --inspect_lines 8 ^
#      --inspect_only
#
# B) Run embedding with default peak mode (mask_from_raw):
#    python scripts/encode_bed_to_embedding.py ^
#      --input_dir "F:\liquid-agent-data\GSE243474\breast" ^
#      --fasta "F:\genome\hg38.fa" ^
#      --model ntv2
#
# C) Run dual-mode comparison in one command:
#    python scripts/encode_bed_to_embedding.py ^
#      --input_dir "F:\liquid-agent-data\GSE243474\breast" ^
#      --fasta "F:\genome\hg38.fa" ^
#      --model ntv2 ^
#      --peak_mode both
#
# D) Inspect one file, then continue to run embedding:
#    python scripts/encode_bed_to_embedding.py ^
#      --inspect_file "GSM7787780_BC287K27R1.bed.gz" ^
#      --input_dir "F:\liquid-agent-data\GSE243474\breast" ^
#      --fasta "F:\genome\hg38.fa" ^
#      --model ntv2
#
# E) Use the beginner preset switch in this file:
#    1) RUN_PRESET_IN_MAIN = True
#    2) PRESET_CASE_IN_MAIN = "smoke_mask_from_raw" (or other preset)
#    3) python scripts/encode_bed_to_embedding.py
# -----------------------------------------------------------------------------
RUN_PRESET_IN_MAIN = False
PRESET_CASE_IN_MAIN = "smoke_mask_from_raw"
PRESET_INPUT_DIR = r"F:\liquid-agent-data\GSE243474\breast"
PRESET_FASTA_PATH = r"F:\genome\hg38.fa"  # TODO: replace with your local FASTA path
PRESET_OUTPUT_BASE = r"F:\liquid-agent-data\GSE243474\embeddings_beginner_tests"

# Quick smoke-test defaults (for fast local sanity checks on macOS data layout).
QUICK_SMOKE_DEFAULT_INPUT_REL = "GSE243474/breast"
QUICK_SMOKE_DEFAULT_OUTPUT_REL = "GSE243474/encoder_smoke_test"
QUICK_SMOKE_DEFAULT_MODELS = ("ntv2", "dnabert2", "hyenadna", "caduceus", "epibert", "enformer")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode BED/BED.GZ files into fixed-size sample embeddings using DNA foundation models."
    )
    parser.add_argument("--input_dir", default=None, type=str, help="Folder containing BED/BED.GZ files.")
    parser.add_argument("--fasta", default=None, type=str, help="Local reference genome FASTA path.")
    parser.add_argument("--output_dir", default=None, type=str, help="Output directory. Default: <input_dir>/embeddings")
    parser.add_argument(
        "--model",
        default="ntv2",
        choices=["ntv2", "dnabert2", "hyenadna", "caduceus", "epibert", "epcot", "enformer"],
        help="Model key. ntv2/dnabert2/hyenadna/caduceus/epibert/epcot/enformer.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Optional explicit model repo/path. If omitted, loader first tries <model_root>/<model_key>.",
    )
    parser.add_argument(
        "--model_root",
        default="models",
        type=str,
        help="Models root relative to configured data root (default: models).",
    )
    parser.add_argument(
        "--peak_mode",
        default="mask_from_raw",
        choices=["direct_narrow_peak", "mask_from_raw", "both"],
        help=(
            "Peak handling mode: mask_from_raw (default, use raw BED intervals with peak-focus mask), "
            "direct_narrow_peak (read narrowPeak files directly), or both (run both modes for comparison)."
        ),
    )
    parser.add_argument(
        "--window_size",
        default=None,
        type=int,
        help="Sequence window size around interval center. Default is model-specific.",
    )
    parser.add_argument(
        "--max_intervals",
        default=None,
        type=int,
        help="Max intervals sampled per file. Default is model-specific.",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size for model inference.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for interval sampling.")
    parser.add_argument("--device", default="auto", type=str, help="Device: auto/cpu/cuda/cuda:0 ...")
    parser.add_argument(
        "--preview_samples",
        default=3,
        type=int,
        help="Print first N sample embeddings (truncated) for quick inspection. Use 0 to disable.",
    )
    parser.add_argument(
        "--preview_dims",
        default=8,
        type=int,
        help="Number of leading embedding dimensions to print in preview logs.",
    )
    parser.add_argument(
        "--allow_model_download",
        action="store_true",
        help="Allow downloading model weights if missing locally. Default is local-only mode.",
    )
    parser.add_argument(
        "--inspect_file",
        default=None,
        type=str,
        help="Inspect one BED/BED.GZ file content summary (supports .gz).",
    )
    parser.add_argument(
        "--inspect_lines",
        default=8,
        type=int,
        help="Number of content lines to print when using --inspect_file.",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect file and exit without running embedding encoding.",
    )
    parser.add_argument(
        "--quick_smoke_test",
        action="store_true",
        help="Run quick multi-encoder smoke test on 1-2 BED files from GSE243474/breast.",
    )
    parser.add_argument(
        "--quick_input_dir",
        default=None,
        type=str,
        help="Optional input directory for quick smoke test. Default: GSE243474/breast under data root.",
    )
    parser.add_argument(
        "--quick_output_dir",
        default=None,
        type=str,
        help="Optional output directory for quick smoke test. Default: GSE243474/encoder_smoke_test under data root.",
    )
    parser.add_argument(
        "--quick_fasta",
        default="auto",
        type=str,
        help="FASTA path for quick smoke test, or 'auto' to search common locations under data root.",
    )
    parser.add_argument(
        "--quick_models",
        nargs="+",
        default=None,
        help=(
            "Model keys for quick smoke test. Default: "
            f"{', '.join(QUICK_SMOKE_DEFAULT_MODELS)}"
        ),
    )
    parser.add_argument(
        "--quick_max_files",
        default=2,
        type=int,
        help="How many BED files to sample in quick smoke test (default: 2).",
    )
    parser.add_argument(
        "--quick_allow_model_download",
        action="store_true",
        help="Allow model download in quick smoke test (default is local-only mode).",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging.")
    return parser


def _open_text_auto(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def inspect_bed_file(file_path: str | Path, *, max_lines: int = 8, max_chars: int = 180) -> dict:
    """
    Inspect one BED/BED.GZ file and return a compact summary.

    This is designed for quick CLI inspection so beginners can see:
    - whether the file is readable
    - what the first valid lines look like
    - basic coordinate stats from a small prefix scan
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Inspect target not found: {p}")

    total_scanned = 0
    valid_bed_lines = 0
    chrom_counts: dict[str, int] = {}
    widths: list[int] = []
    preview_lines: list[str] = []

    with _open_text_auto(p) as f:
        for line in f:
            total_scanned += 1
            s = line.rstrip("\n")
            if not s:
                continue
            if s.startswith("#") or s.startswith("track") or s.startswith("browser"):
                if len(preview_lines) < max_lines:
                    preview_lines.append(s[:max_chars])
                continue

            parts = s.split("\t")
            if len(parts) >= 3:
                chrom = parts[0].strip()
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    start, end = 0, 0
                if chrom and end >= start:
                    valid_bed_lines += 1
                    chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
                    widths.append(end - start)
            if len(preview_lines) < max_lines:
                preview_lines.append(s[:max_chars])
            if len(preview_lines) >= max_lines and valid_bed_lines >= max_lines:
                # Prefix scan is enough for quick inspection.
                break

    widths_arr = widths if widths else [0]
    top_chrom = sorted(chrom_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "file_path": str(p),
        "total_lines_scanned_prefix": total_scanned,
        "valid_bed_lines_prefix": valid_bed_lines,
        "interval_width_min_prefix": int(min(widths_arr)),
        "interval_width_max_prefix": int(max(widths_arr)),
        "interval_width_mean_prefix": float(sum(widths_arr) / max(1, len(widths_arr))),
        "top_chrom_prefix": top_chrom,
        "preview_lines": preview_lines,
    }


def _is_narrow_peak_name(file_name: str) -> bool:
    lower = file_name.lower()
    return ("narrowpeak" in lower) or ("sorted_peaks" in lower)


def summarize_input_folder(input_dir: str | Path, *, preview_files: int = 2, preview_lines: int = 3) -> dict:
    """Print and return a compact summary of input BED/BED.GZ files."""
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    files = sorted([p for p in root.iterdir() if p.is_file() and p.name.lower().endswith((".bed", ".bed.gz"))])
    narrow = [p for p in files if _is_narrow_peak_name(p.name)]
    raw = [p for p in files if not _is_narrow_peak_name(p.name)]
    out = {
        "input_dir": str(root),
        "total_bed_files": len(files),
        "raw_bed_files": len(raw),
        "narrow_peak_files": len(narrow),
        "preview": [],
    }

    for p in files[: max(0, preview_files)]:
        out["preview"].append(inspect_bed_file(p, max_lines=preview_lines))
    return out


def _resolve_fasta_path_for_beginner(fasta_value: str) -> str:
    """
    Resolve FASTA path for beginner mode.

    - If `fasta_value` is a valid existing file, use it directly.
    - If `fasta_value` is 'auto', try common hg38/hg19 locations on Windows.
    """
    candidate = Path(fasta_value)
    if fasta_value.lower() != "auto" and candidate.exists():
        return str(candidate)

    auto_candidates = [
        Path(r"F:\genome\hg38.fa"),
        Path(r"F:\genome\GRCh38.fa"),
        Path(r"F:\liquid-agent-data\genome\hg38.fa"),
        Path(r"F:\liquid-agent-data\reference\hg38.fa"),
        Path(r"F:\genome\hg19.fa"),
        Path(r"F:\genome\GRCh37.fa"),
        Path(r"F:\liquid-agent-data\genome\hg19.fa"),
    ]
    for p in auto_candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "FASTA file not found. Set PRESET_FASTA_PATH to your local hg38/hg19 FASTA "
        "or pass --fasta <path> in CLI mode."
    )


def _build_beginner_case(case_name: str) -> dict:
    """
    Build one beginner test case.

    Available cases:
      - smoke_mask_from_raw: fastest sanity test from raw BED + peak mask
      - smoke_direct_narrow_peak: fastest sanity test from narrowPeak files
      - full_mask_from_raw: full run using raw BED + peak mask
      - full_direct_narrow_peak: full run using narrowPeak directly
      - compare_both_modes: run both modes in one command (best for comparison)
    """
    common = {
        "model": "ntv2",
        "device": "auto",
        "allow_model_download": False,
        "preview_samples": 3,
        "preview_dims": 8,
        "seed": 42,
        "quiet": False,
        "model_name": None,
        "model_root": "models",
        "window_size": None,
        "batch_size": None,
    }
    presets = {
        "smoke_mask_from_raw": {
            **common,
            "peak_mode": "mask_from_raw",
            "max_intervals": 32,
        },
        "smoke_direct_narrow_peak": {
            **common,
            "peak_mode": "direct_narrow_peak",
            "max_intervals": 32,
        },
        "full_mask_from_raw": {
            **common,
            "peak_mode": "mask_from_raw",
            "max_intervals": None,
        },
        "full_direct_narrow_peak": {
            **common,
            "peak_mode": "direct_narrow_peak",
            "max_intervals": None,
        },
        "compare_both_modes": {
            **common,
            "peak_mode": "both",
            "max_intervals": 64,
        },
    }
    if case_name not in presets:
        raise ValueError(f"Unknown PRESET_CASE_IN_MAIN='{case_name}'. Supported: {list(presets.keys())}")
    return presets[case_name]


def _run_encoding_from_args(args) -> dict:
    from liquidbiopsy_agent.multimodal.bed_embedding import encode_bed_folder_to_embeddings

    if not args.quiet:
        input_summary = summarize_input_folder(args.input_dir, preview_files=2, preview_lines=3)
        print("[INPUT_SUMMARY]")
        print(json.dumps(input_summary, ensure_ascii=False, indent=2))

    if args.peak_mode == "both":
        base_output = Path(args.output_dir) if args.output_dir else (Path(args.input_dir) / "embeddings_peak_mode_compare")
        summaries: dict[str, dict] = {}
        for mode in ("mask_from_raw", "direct_narrow_peak"):
            run_output = base_output / mode
            summaries[mode] = encode_bed_folder_to_embeddings(
                input_dir=args.input_dir,
                fasta_path=args.fasta,
                output_dir=run_output,
                model_key=args.model,
                model_name=args.model_name,
                model_root=args.model_root,
                window_size=args.window_size,
                max_intervals_per_file=args.max_intervals,
                batch_size=args.batch_size,
                seed=args.seed,
                local_files_only=not args.allow_model_download,
                device=args.device,
                verbose=not args.quiet,
                preview_samples=args.preview_samples,
                preview_dims=args.preview_dims,
                peak_mode=mode,
            )
        return {"peak_mode": "both", "runs": summaries}

    return encode_bed_folder_to_embeddings(
        input_dir=args.input_dir,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        model_key=args.model,
        model_name=args.model_name,
        model_root=args.model_root,
        window_size=args.window_size,
        max_intervals_per_file=args.max_intervals,
        batch_size=args.batch_size,
        seed=args.seed,
        local_files_only=not args.allow_model_download,
        device=args.device,
        verbose=not args.quiet,
        preview_samples=args.preview_samples,
        preview_dims=args.preview_dims,
        peak_mode=args.peak_mode,
    )


def _resolve_quick_smoke_fasta(fasta_value: str) -> Path:
    from liquidbiopsy_agent.utils.storage import get_data_root, resolve_data_path

    raw = (fasta_value or "auto").strip()
    if raw.lower() != "auto":
        direct = Path(raw).expanduser()
        if direct.exists() and direct.is_file():
            return direct.resolve()
        resolved = resolve_data_path(raw, path_kind="quick smoke FASTA path", must_exist=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"quick smoke FASTA path is not a file: {resolved}")
        return resolved

    data_root = get_data_root()
    auto_candidates = [
        data_root / "genome" / "hg38.fa",
        data_root / "reference" / "hg38.fa",
        data_root / "genome" / "GRCh38.fa",
        data_root / "reference" / "GRCh38.fa",
        data_root / "genome" / "hg19.fa",
        data_root / "reference" / "hg19.fa",
        Path("/Volumes/US202/liquid-agent-data/genome/hg38.fa"),
        Path("/Volumes/US202/liquid-agent-data/reference/hg38.fa"),
    ]
    for p in auto_candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "Quick smoke FASTA not found. Please pass --quick_fasta /abs/path/to/hg38.fa "
        "or place FASTA under <data_root>/genome or <data_root>/reference."
    )


def _prepare_quick_smoke_subset(input_dir: Path, subset_dir: Path, max_files: int) -> list[Path]:
    bed_files = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file()
            and not p.name.startswith("._")
            and p.name.lower().endswith((".bed", ".bed.gz"))
        ]
    )
    if not bed_files:
        raise ValueError(f"No BED/BED.GZ files found in quick smoke input: {input_dir}")

    n = max(1, int(max_files))
    chosen = bed_files[:n]

    subset_dir.mkdir(parents=True, exist_ok=True)
    for old in subset_dir.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()

    for src in chosen:
        dst = subset_dir / src.name
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)

    return chosen


def _embedding_basic_stats(embeddings_csv: Path, *, head_dims: int = 8) -> dict[str, Any]:
    df = pd.read_csv(embeddings_csv)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if df.empty or not emb_cols:
        return {
            "samples": int(len(df)),
            "embedding_dim": 0,
            "value_mean": None,
            "value_std": None,
            "value_min": None,
            "value_max": None,
            "first_sample_head": [],
        }

    arr = df[emb_cols].to_numpy(dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    first_head = arr[0, : min(head_dims, arr.shape[1])]
    first_head_list = [float(v) if np.isfinite(v) else None for v in first_head.tolist()]

    if finite.size == 0:
        mean_v = std_v = min_v = max_v = None
    else:
        mean_v = float(finite.mean())
        std_v = float(finite.std())
        min_v = float(finite.min())
        max_v = float(finite.max())

    return {
        "samples": int(arr.shape[0]),
        "embedding_dim": int(arr.shape[1]),
        "value_mean": mean_v,
        "value_std": std_v,
        "value_min": min_v,
        "value_max": max_v,
        "first_sample_head": first_head_list,
    }


def run_quick_encoder_smoke_test(
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    fasta_path: str = "auto",
    model_keys: Sequence[str] | None = None,
    max_files: int = 2,
    max_intervals: int = 64,
    allow_model_download: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    """
    Quick smoke test for multiple encoders on 1-2 BED files.

    Default flow:
    - Input: GSE243474/breast under data root
    - Select first 1-2 BED files
    - Run multiple encoder backbones (mask_from_raw mode)
    - Print basic embedding stats per encoder
    """
    from liquidbiopsy_agent.multimodal.bed_embedding import encode_bed_folder_to_embeddings
    from liquidbiopsy_agent.utils.storage import resolve_data_path

    models = list(model_keys) if model_keys else list(QUICK_SMOKE_DEFAULT_MODELS)
    input_path = resolve_data_path(
        input_dir or QUICK_SMOKE_DEFAULT_INPUT_REL,
        path_kind="quick smoke input dir",
        must_exist=True,
    )
    output_base = resolve_data_path(
        output_dir or QUICK_SMOKE_DEFAULT_OUTPUT_REL,
        path_kind="quick smoke output dir",
        must_exist=False,
    )
    fasta_file = _resolve_quick_smoke_fasta(fasta_path)
    output_base.mkdir(parents=True, exist_ok=True)

    subset_dir = output_base / "_subset_bed_files"
    chosen_files = _prepare_quick_smoke_subset(
        input_dir=input_path,
        subset_dir=subset_dir,
        max_files=max_files,
    )

    print("[QUICK_SMOKE] Running encoder smoke test.")
    print(f"[QUICK_SMOKE] input_dir={input_path}")
    print(f"[QUICK_SMOKE] fasta={fasta_file}")
    print(f"[QUICK_SMOKE] output_base={output_base}")
    print(f"[QUICK_SMOKE] chosen_files={[p.name for p in chosen_files]}")
    print(f"[QUICK_SMOKE] models={models}")

    results: list[dict[str, Any]] = []
    for model_key in models:
        run_output = output_base / f"{model_key}_mask_from_raw"
        print(f"[QUICK_SMOKE] >>> model={model_key}")
        try:
            summary = encode_bed_folder_to_embeddings(
                input_dir=subset_dir,
                fasta_path=fasta_file,
                output_dir=run_output,
                model_key=model_key,
                max_intervals_per_file=max(8, int(max_intervals)),
                batch_size=8,
                seed=42,
                local_files_only=not allow_model_download,
                device=device,
                verbose=True,
                preview_samples=min(len(chosen_files), 2),
                preview_dims=8,
                peak_mode="mask_from_raw",
            )
            stats = _embedding_basic_stats(Path(summary["embeddings_csv"]), head_dims=8)
            result_item = {
                "model_key": model_key,
                "status": "ok",
                "summary": {
                    "embedding_dim": summary.get("embedding_dim"),
                    "files_processed": summary.get("files_processed"),
                    "status_counts": summary.get("status_counts"),
                    "embeddings_csv": summary.get("embeddings_csv"),
                    "metadata_csv": summary.get("metadata_csv"),
                },
                "basic_stats": stats,
            }
            print(
                f"[QUICK_SMOKE] model={model_key} ok "
                f"emb_dim={stats['embedding_dim']} samples={stats['samples']} "
                f"mean={stats['value_mean']} std={stats['value_std']}"
            )
        except Exception as exc:
            result_item = {
                "model_key": model_key,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[QUICK_SMOKE] model={model_key} error: {result_item['error']}")
        results.append(result_item)

    summary_out = {
        "input_dir": str(input_path),
        "subset_dir": str(subset_dir),
        "subset_files": [p.name for p in chosen_files],
        "fasta_path": str(fasta_file),
        "output_base_dir": str(output_base),
        "models": models,
        "results": results,
    }
    summary_path = output_base / "quick_smoke_summary.json"
    summary_path.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[QUICK_SMOKE] summary_json={summary_path}")
    return summary_out


def main() -> None:
    if RUN_PRESET_IN_MAIN:
        preset = _build_beginner_case(PRESET_CASE_IN_MAIN)
        fasta_path = _resolve_fasta_path_for_beginner(PRESET_FASTA_PATH)
        args = SimpleNamespace(
            input_dir=PRESET_INPUT_DIR,
            fasta=fasta_path,
            output_dir=str(Path(PRESET_OUTPUT_BASE) / PRESET_CASE_IN_MAIN),
            **preset,
        )
        print("[BEGINNER] Running preset case from main()")
        print(f"[BEGINNER] case={PRESET_CASE_IN_MAIN}")
        print(f"[BEGINNER] input_dir={args.input_dir}")
        print(f"[BEGINNER] fasta={args.fasta}")
        print(f"[BEGINNER] output_dir={args.output_dir}")
        summary = _run_encoding_from_args(args)
        print("[SUMMARY]")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    args = _build_arg_parser().parse_args()
    if args.quick_smoke_test:
        summary = run_quick_encoder_smoke_test(
            input_dir=args.quick_input_dir,
            output_dir=args.quick_output_dir,
            fasta_path=args.quick_fasta,
            model_keys=args.quick_models,
            max_files=args.quick_max_files,
            allow_model_download=args.quick_allow_model_download,
            device=args.device,
        )
        print("[QUICK_SMOKE_SUMMARY]")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.inspect_file:
        inspect_target = Path(args.inspect_file)
        if not inspect_target.exists() and args.input_dir:
            joined = Path(args.input_dir) / args.inspect_file
            if joined.exists():
                inspect_target = joined
        inspection = inspect_bed_file(inspect_target, max_lines=max(1, int(args.inspect_lines)))
        print("[INSPECT]")
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        if args.inspect_only:
            return

    if not args.input_dir or not args.fasta:
        raise ValueError("For encoding mode, both --input_dir and --fasta are required.")
    summary = _run_encoding_from_args(args)
    if not args.quiet:
        print("[SUMMARY]")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
