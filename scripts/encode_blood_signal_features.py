#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liquidbiopsy_agent.multimodal.blood_signal_encoding import (
    encode_blood_signal_dataset,
    list_supported_blood_signal_specs,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode blood-biopsy signals into packed PT features. "
            "Supports interval signals (cfChIP/cfMeDIP/MeDIP), LPWGS (CNV-profile default; foundation optional), "
            "and VCF signatures."
        )
    )
    parser.add_argument(
        "--signal",
        required=True,
        choices=sorted(list_supported_blood_signal_specs().keys()),
        help="Blood signal type to encode.",
    )
    parser.add_argument(
        "--input_format",
        required=True,
        type=str,
        help="Input format for the selected signal, e.g. bed.gz / cnv_parquet / vcf.",
    )
    parser.add_argument("--input_dir", required=True, type=str, help="Input directory under data root.")
    parser.add_argument("--output_dir", default=None, type=str, help="Optional explicit output directory.")

    parser.add_argument(
        "--encoder",
        default=None,
        type=str,
        help="Encoder key. Default follows signal-specific recommendation.",
    )
    parser.add_argument("--fasta_path", default=None, type=str, help="Reference FASTA path (required for interval signals).")
    parser.add_argument("--model_name", default=None, type=str, help="Optional explicit model path/name.")
    parser.add_argument("--model_root", default="models", type=str, help="Model root under data root.")
    parser.add_argument("--peak_mode", default="mask_from_raw", type=str, help="Peak mode for interval signals.")
    parser.add_argument("--window_size", default=None, type=int, help="Interval sequence window size.")
    parser.add_argument("--max_intervals_per_file", default=None, type=int, help="Max sampled intervals per file.")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size for model inference.")

    parser.add_argument("--bin_size", default=1_000_000, type=int, help="Bin size for LPWGS bed -> CNV bins.")
    parser.add_argument("--target_bins", default=256, type=int, help="Target contour bins for LPWGS profile vector.")

    parser.add_argument("--seed", default=42, type=int, help="Random seed for interval sampling.")
    parser.add_argument("--device", default="auto", type=str, help="Torch device (auto/cpu/cuda/mps).")
    parser.add_argument(
        "--allow_model_download",
        action="store_true",
        help="Allow model download if local model files are missing.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    summary = encode_blood_signal_dataset(
        signal=args.signal,
        input_dir=args.input_dir,
        input_format=args.input_format,
        output_dir=args.output_dir,
        encoder=args.encoder,
        fasta_path=args.fasta_path,
        model_name=args.model_name,
        model_root=args.model_root,
        peak_mode=args.peak_mode,
        window_size=args.window_size,
        max_intervals_per_file=args.max_intervals_per_file,
        batch_size=args.batch_size,
        bin_size=args.bin_size,
        target_bins=args.target_bins,
        seed=args.seed,
        local_files_only=not args.allow_model_download,
        trust_remote_code=True,
        device=args.device,
        verbose=not args.quiet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
