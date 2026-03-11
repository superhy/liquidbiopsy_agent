#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liquidbiopsy_agent.multimodal.feature_contrastive import train_feature_contrastive_from_config
from liquidbiopsy_agent.utils.storage import get_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train feature-level cross-modal contrastive model from precomputed cfDNA/PT and WSI/PT features "
            "(slide-level or tile-level)."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file, e.g. configs/multimodal_feature_her2_demo.yaml",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    logging.getLogger("train_feature_contrastive").info("Using data root: %s", get_data_root())
    summary = train_feature_contrastive_from_config(Path(args.config))
    print("Feature-level contrastive training finished")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
