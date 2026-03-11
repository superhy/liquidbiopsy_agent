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

from liquidbiopsy_agent.visualization import run_cfdna_plot_suite


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "cfDNA visualization suite: feature-space scatter(+hyperplane), "
            "fragmentomics, methylation-proxy heatmap, CNV heatmap."
        )
    )
    parser.add_argument(
        "--output_dir",
        default="GSE243474/visualisation/cfdna",
        type=str,
        help="Output directory under data root for generated figures/tables.",
    )
    parser.add_argument(
        "--cfdna_features_pt",
        default=None,
        type=str,
        help="cfDNA packed feature PT (sample_id -> embedding tensor).",
    )
    parser.add_argument(
        "--labels_table",
        default=None,
        type=str,
        help=(
            "Optional labels table for coloring and hyperplane fitting. "
            "Expected columns include sample id + label."
        ),
    )
    parser.add_argument("--labels_sample_col", default="sample_id", type=str, help="Sample id column in labels table.")
    parser.add_argument("--labels_label_col", default="her2_status", type=str, help="Label column in labels table.")
    parser.add_argument(
        "--frag_dir",
        default=None,
        type=str,
        help="Optional fragmentomics directory (expects *_length_hist.parquet and optional frag_summary.parquet).",
    )
    parser.add_argument(
        "--meth_summary_path",
        default=None,
        type=str,
        help="Optional methylation proxy summary table path (e.g. meth_proxy_summary.parquet).",
    )
    parser.add_argument(
        "--cnv_dir",
        default=None,
        type=str,
        help="Optional CNV directory (expects *_bin_counts.parquet).",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed for hyperplane fitting.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    summary = run_cfdna_plot_suite(
        output_dir=args.output_dir,
        cfdna_features_pt=args.cfdna_features_pt,
        labels_table=args.labels_table,
        labels_sample_col=args.labels_sample_col,
        labels_label_col=args.labels_label_col,
        frag_dir=args.frag_dir,
        meth_summary_path=args.meth_summary_path,
        cnv_dir=args.cnv_dir,
        random_seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
