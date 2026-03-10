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

from liquidbiopsy_agent.multimodal.wsi_encoding import (
    TANGLE_PRETRAINED_DRIVE_URL,
    encode_tcga_brca_wsi,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode TCGA-BRCA pathology WSIs into slide-level embeddings: "
            "TRIDENT preprocessing + UNI-V2 tile features + TANGLE slide encoder."
        )
    )
    parser.add_argument(
        "--slides_dir",
        default="TCGA-BRCA/slides",
        type=str,
        help="WSI input directory under data root (default: TCGA-BRCA/slides).",
    )
    parser.add_argument(
        "--output_root",
        default="TCGA-BRCA/wsi_embeddings",
        type=str,
        help="Output root under data root.",
    )
    parser.add_argument(
        "--models_root",
        default="models",
        type=str,
        help="Models root under data root.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        type=str,
        help="Device: auto/cpu/cuda:0/mps. auto follows project unified policy.",
    )

    parser.add_argument("--trident_repo_dir", default=None, type=str, help="Optional local TRIDENT repo path.")
    parser.add_argument("--tangle_repo_dir", default=None, type=str, help="Optional local TANGLE repo path.")
    parser.add_argument("--uni_v2_ckpt_path", default=None, type=str, help="Optional local UNI-V2 checkpoint path.")
    parser.add_argument(
        "--hf_token",
        default=None,
        type=str,
        help="Optional Hugging Face token for gated UNI-V2 download.",
    )
    parser.add_argument("--tangle_checkpoint_dir", default=None, type=str, help="Optional local TANGLE checkpoint dir.")
    parser.add_argument(
        "--tangle_pretrained_root_dir",
        default=None,
        type=str,
        help="Directory used to store downloaded TANGLE checkpoints.",
    )
    parser.add_argument(
        "--tangle_drive_url",
        default=TANGLE_PRETRAINED_DRIVE_URL,
        type=str,
        help="Google Drive folder URL for pretrained TANGLE checkpoints.",
    )
    parser.add_argument(
        "--tangle_checkpoint_keyword",
        default="tangle_brca",
        type=str,
        help="Preferred keyword used when auto-selecting a TANGLE checkpoint.",
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Disable all model/checkpoint auto-downloads (local files only).",
    )

    parser.add_argument("--skip_trident", action="store_true", help="Skip TRIDENT preprocessing + UNI-V2 feature extraction.")
    parser.add_argument("--skip_tangle", action="store_true", help="Skip TANGLE slide embedding stage.")
    parser.add_argument(
        "--patch_features_dir",
        default=None,
        type=str,
        help="Patch feature directory (required when --skip_trident is set).",
    )

    parser.add_argument("--reader_type", default="openslide", choices=["openslide", "image", "cucim", "sdpc"])
    parser.add_argument("--segmenter", default="hest", choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg_conf_thresh", default=0.5, type=float)
    parser.add_argument("--remove_holes", action="store_true", help="Treat segmentation holes as non-tissue.")
    parser.add_argument(
        "--disable_artifact_removal",
        action="store_true",
        help="Disable artifact removal model during segmentation.",
    )
    parser.add_argument("--remove_penmarks", action="store_true", help="Enable pen-mark-specific artifact removal.")

    parser.add_argument("--mag", default=20.0, type=float, help="Target magnification for patching.")
    parser.add_argument("--patch_size", default=256, type=int, help="Patch size for tiling.")
    parser.add_argument("--overlap", default=0, type=int, help="Patch overlap (pixels).")
    parser.add_argument("--min_tissue_proportion", default=0.0, type=float, help="Minimum tissue proportion per patch.")
    parser.add_argument("--seg_batch_size", default=16, type=int)
    parser.add_argument("--feat_batch_size", default=256, type=int)
    parser.add_argument("--max_workers", default=None, type=int)
    parser.add_argument("--skip_errors", action="store_true", help="Skip failed slides and continue.")
    parser.add_argument("--search_nested", action="store_true", help="Recursively search nested WSI folders.")
    parser.add_argument("--custom_list_of_wsis", default=None, type=str, help="CSV with column `wsi` for slide subset.")

    parser.add_argument("--extension", default=".h5", type=str, help="Patch feature extension consumed by TANGLE.")
    parser.add_argument("--tangle_num_workers", default=0, type=int)
    parser.add_argument(
        "--feature_dim_policy",
        default="truncate_or_pad",
        choices=["strict", "truncate", "pad", "truncate_or_pad"],
        help=(
            "How to handle patch-feature dim mismatch vs TANGLE checkpoint "
            "(UNI-V2 is often 1536 while BRCA TANGLE checkpoints are 1024)."
        ),
    )

    parser.add_argument(
        "--quick_smoke_test",
        action="store_true",
        help="Use lighter defaults for a faster functional check.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.skip_trident and not args.patch_features_dir:
        parser.error("--patch_features_dir is required when --skip_trident is set")

    if args.quick_smoke_test:
        args.patch_size = 224
        args.seg_batch_size = min(args.seg_batch_size, 4)
        args.feat_batch_size = min(args.feat_batch_size, 32)

    summary = encode_tcga_brca_wsi(
        slides_dir=args.slides_dir,
        output_root=args.output_root,
        device=args.device,
        models_root=args.models_root,
        trident_repo_dir=args.trident_repo_dir,
        tangle_repo_dir=args.tangle_repo_dir,
        uni_v2_ckpt_path=args.uni_v2_ckpt_path,
        hf_token=args.hf_token,
        tangle_checkpoint_dir=args.tangle_checkpoint_dir,
        tangle_pretrained_root_dir=args.tangle_pretrained_root_dir,
        allow_model_download=not args.local_only,
        tangle_drive_url=args.tangle_drive_url,
        tangle_checkpoint_keyword=args.tangle_checkpoint_keyword,
        run_trident=not args.skip_trident,
        run_tangle=not args.skip_tangle,
        patch_features_dir=args.patch_features_dir,
        reader_type=args.reader_type,
        segmenter=args.segmenter,
        seg_conf_thresh=args.seg_conf_thresh,
        remove_holes=args.remove_holes,
        remove_artifacts=not args.disable_artifact_removal,
        remove_penmarks=args.remove_penmarks,
        mag=args.mag,
        patch_size=args.patch_size,
        overlap=args.overlap,
        min_tissue_proportion=args.min_tissue_proportion,
        seg_batch_size=args.seg_batch_size,
        feat_batch_size=args.feat_batch_size,
        max_workers=args.max_workers,
        skip_errors=args.skip_errors,
        search_nested=args.search_nested,
        custom_list_of_wsis=args.custom_list_of_wsis,
        extension=args.extension,
        tangle_num_workers=args.tangle_num_workers,
        feature_dim_policy=args.feature_dim_policy,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
