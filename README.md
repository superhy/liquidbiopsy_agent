# LiquidBiopsy Agent (BED-only MVP)

Agentised, cache-aware cfDNA pipeline that ingests BED(.bed.gz) fragment coordinates and produces QC, CNV, fragmentomics, methylation-proxy features, cohort summaries, and an HTML report. Runs on CPU-only workstations; LLM integration is optional and degrades gracefully. The pipeline is orchestrated with LangGraph for extensible agent-style execution without LangChain.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install all optional features (reporting, multimodal, WSI, local/openai LLM, tools):
```bash
pip install -e ".[report,ml,local-llm,llm-openai,multimodal,wsi,tools]"
```

## Data Root Policy
This repository is code-only. Do not store datasets or model checkpoints in git.
All dataset/checkpoint/output paths are resolved under one server root:
- global override (highest priority): `LIQUID_BIOPSY_DATA_ROOT`
- Ubuntu default (placeholder in code): `/home/YOUR_UBUNTU_ROOT_PREFIX/liquid-agent-data`
- macOS default: `/Volumes/US202/liquid-agent-data`
- Windows default: `F:\liquid-agent-data`
- optional OS-specific overrides:
  - `LIQUID_BIOPSY_DATA_ROOT_UBUNTU`
  - `LIQUID_BIOPSY_DATA_ROOT_MACOS`
  - `LIQUID_BIOPSY_DATA_ROOT_WINDOWS`
- current top-level dataset folders under root: `GSE243474`, `TCGA-BRCA`
- run-operation logs (when written by scripts): `GSE243474/log`

Any runtime data or weight path must be inside that root.

## Quickstart
```bash
# Linux/macOS example:
export LIQUID_BIOPSY_DATA_ROOT=/your/fixed/server/path/liquid-agent-data
# Windows PowerShell example:
# $env:LIQUID_BIOPSY_DATA_ROOT='C:\your\fixed\server\path\liquid-agent-data'
python -m liquidbiopsy_agent run --input datasets/bed/my_cohort --output runs/bed_mvp --config configs/default.yaml
python -m liquidbiopsy_agent run --input datasets/bed/my_cohort --output runs/bed_mvp --instruction "开启所有图"
python -m liquidbiopsy_agent status --run-dir runs/bed_mvp/run_YYYYMMDD_HHMMSS
python -m liquidbiopsy_agent resume --run-dir runs/bed_mvp/run_YYYYMMDD_HHMMSS --instruction "no cnv"
```
Supports tar bundles too: `--input datasets/tar/GSE243474_RAW.tar`.

## Inputs
- BED or BED.GZ fragment coordinates (3+ columns); FASTQ/BAM not required.
- Genome build and assay labels are inferred heuristically from filenames (LLM-assisted if enabled).
- Default methylation proxy panels live in `ref/region_sets/` (tiny demos). Replace with real panels for analysis.

## Outputs (per run directory)
- `data/manifest/manifest.parquet`
- `data/processed_bed/*.bed.gz` (normalised 3-col)
- `qc/sample_qc.parquet` + `qc/plots/*.png`
- Extra plots: `qc/plots/coverage_uniformity_heatmap.png`, `features/cnv/plots/*_cnv_profile.png`, `features/frag/plots/length_peak_comparison.png`
- Optional advanced plots (toggle via config/LLM): `qc/plots/coverage_bin_heatmap.png`, `features/cnv/plots/*_cnv_chrom_plot.png`, `features/frag/plots/length_violin.png`
- `features/{cnv,frag,meth_proxy,merged}/...`
- `analysis/cohort_summary.json`, `analysis/drift_flags.json`
- `analysis/agent/agent_summary.json` (parameter tuning + rerun plan + explanation)
- `analysis/agent/explanation.txt` (British English narrative)
- `reports/report.html`

## What this MVP does / does not do
- Uses BED only; no end-motif, SNV, or full methylation calling.
- CNV via coarse bin counting with simple smoothing; GC correction optional via config.
- Fragmentomics via length histograms and entropy proxies.
- Methylation-proxy uses region panels to count fragment midpoints.

## LLM decisions (optional, local first)
Set `llm.enable: true` in `configs/default.yaml` and configure a local model. Decisions fall back to heuristics when LLM is disabled or responses are invalid.

Example (llama.cpp local model):
```bash
pip install -e ".[local-llm]"
export LIQUIDBIOPSY_LLM_PROVIDER=local_llama_cpp
export LIQUIDBIOPSY_LLM_MODEL_PATH=weights/llm/your_model.gguf
```

Example (OpenAI API):
```bash
pip install -e ".[llm-openai]"
export LIQUIDBIOPSY_LLM_PROVIDER=openai
export LIQUIDBIOPSY_LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=your_api_key
```

## Agent interpretation outputs
The agent writes intermediate, interpretable outputs to:
- `analysis/agent/parameter_tuning.json`
- `analysis/agent/rerun_plan.json`
- `analysis/agent/explanation.txt`
These are intended to support biomedical review and parameter refinement between runs.

## New module: tissue + blood multimodal contrastive learning (HER2 example)
This repository now includes an independent module at `src/liquidbiopsy_agent/multimodal/` for learning cross-modal alignment between:
- Tissue pathology images (image foundation backbone; e.g. ResNet/EfficientNet),
- Blood-based features (e.g. methylation proxy vectors).

The module applies two encoder/projection heads and a subtype-aware contrastive objective:
- Pull together image/blood embeddings from the same molecular subtype (e.g. HER2+ with HER2+),
- Push apart embeddings from different subtypes (e.g. HER2+ vs HER2-).

### Install
```bash
pip install -e ".[multimodal]"
```

### Train (example config)
```bash
python scripts/train_multimodal.py --config configs/multimodal_her2_demo.yaml
```

### Train from pre-encoded features (cfDNA PT + WSI PT)
```bash
python scripts/train_feature_contrastive.py --config configs/multimodal_feature_her2_demo.yaml
```

### cfDNA visualisation suite
Generate cfDNA-side plots from precomputed features and optional analysis tables:
- feature-space scatter (PCA 2D) + linear hyperplane attempt (binary labels)
- fragment-length distribution + short-fragment ratio plot
- methylation-proxy heatmap
- CNV heatmap

```bash
python scripts/run_cfdna_plot_suite.py \
  --output_dir GSE243474/visualisation/cfdna \
  --cfdna_features_pt GSE243474/features/ntv2/breast/mask_from_raw/cfdna_features__encoder-ntv2__mode-mask_from_raw.pt \
  --labels_table GSE243474/path/to/her2_labels.csv \
  --labels_sample_col sample_id \
  --labels_label_col her2_status \
  --frag_dir runs/example_run/features/frag \
  --meth_summary_path runs/example_run/features/meth_proxy/meth_proxy_summary.parquet \
  --cnv_dir runs/example_run/features/cnv
```

Legacy alias still works:
```bash
python scripts/visualize_cfdna.py ...
```

All paths are resolved under the configured data root.

Default output is grouped by analysis type under `GSE243474/visualisation/cfdna/`:
- `feature_space/`:
  - `cfdna_feature_space__pca2d__linear_hyperplane.png`
  - `cfdna_feature_space__projected_points.csv`
- `fragmentomics/`:
  - `cfdna_fragmentomics__length_distribution_by_label.png`
  - `cfdna_fragmentomics__length_density_by_label.csv`
  - `cfdna_fragmentomics__short_fragment_ratio_table.csv`
  - `cfdna_fragmentomics__short_fragment_ratio_boxplot.png`
- `methylation_proxy/`:
  - `cfdna_methylation_proxy__heatmap.png`
  - `cfdna_methylation_proxy__matrix.csv`
- `cnv/`:
  - `cfdna_cnv__heatmap.png`
  - `cfdna_cnv__matrix.csv`
- root summary:
  - `cfdna_visualisation_summary.json`

### Quick encoder smoke test (1-2 cfDNA BED samples)
```bash
python scripts/encode_bed_to_embedding.py --quick_smoke_test

# If FASTA is not in <data_root>/genome or <data_root>/reference, pass explicit path:
python scripts/encode_bed_to_embedding.py \
  --quick_smoke_test \
  --quick_fasta /absolute/path/to/hg38.fa
```

Optional flags:
- `--quick_models ntv2 dnabert2 hyenadna` (choose specific encoders)
- `--quick_max_files 1` (use only one BED sample)
- `--quick_allow_model_download` (allow weight download if local weights are missing)

cfDNA feature package naming (sample_id -> embedding tensor):
- `cfdna_features__encoder-<encoder_name>__mode-<peak_mode>.pt`
- Stored under `GSE243474/features/<encoder>/<cohort>/<mode>/`

### Multi-signal blood feature encoding (cfChIP / cfMeDIP / LPWGS / VCF)
Unified entrypoint:
```bash
python scripts/encode_blood_signal_features.py \
  --signal cfchip_seq \
  --input_format bed.gz \
  --input_dir GSE243474/breast \
  --fasta_path genome/hg38.fa \
  --encoder ntv2 \
  --peak_mode mask_from_raw
```

LPWGS encoding (correctness-first default: CNV profile encoder):
```bash
python scripts/encode_blood_signal_features.py \
  --signal lpwgs \
  --input_format bed.gz \
  --input_dir GSE243474/breast \
  --encoder lpwgs_cnv_profile \
  --bin_size 1000000 \
  --target_bins 256
```

LPWGS foundation-model encoding (optional exploratory path):
```bash
python scripts/encode_blood_signal_features.py \
  --signal lpwgs \
  --input_format bed.gz \
  --input_dir GSE243474/breast \
  --fasta_path genome/hg38.fa \
  --encoder ntv2
```

VCF signature encoding:
```bash
python scripts/encode_blood_signal_features.py \
  --signal ctdna_variant \
  --input_format vcf \
  --input_dir GSE243474/variants \
  --encoder vcf_signature
```

Packed feature naming:
- `blood_features__signal-<signal>__encoder-<encoder>__mode-<mode>.pt`

Encoder applicability rules are documented in:
- `src/liquidbiopsy_agent/multimodal/blood_signal_encoding.py`

### WSI pathology encoder (TRIDENT + UNI-V2 + TANGLE)
WSI pipeline entrypoint:
```bash
python scripts/encode_wsi_to_slide_embedding.py
```

Role split in this project:
- TRIDENT: pathology WSI preprocessing (segmentation + tiling + coord management).
- UNI-V2: tile-level encoding (all tile features and representative tile features are UNI-based).
- TANGLE: slide-level encoding only (aggregates patch features into one slide embedding).

Default data paths (resolved under data root):
- slides: `TCGA-BRCA/slides`
- models root: `models`
- outputs: `TCGA-BRCA/wsi_embeddings`

Important implementation detail:
- TRIDENT/TANGLE are loaded as source repositories from `models/third_party/` at runtime.
- They are not imported from pip-installed `trident`/`tangle` packages in the active Python environment.
- Default clone targets:
  - `<data_root>/models/third_party/TRIDENT`
  - `<data_root>/models/third_party/TANGLE`
- You can override repo locations per run:
  - `--trident_repo_dir /absolute/path/to/TRIDENT`
  - `--tangle_repo_dir /absolute/path/to/TANGLE`

Why `third_party` is kept under data root:
- Keeps upstream code isolated from this project source tree.
- Makes updating/replacing upstream repos explicit.
- Avoids coupling runtime behavior to whichever packages happen to be installed in an env.

Intermediate files and whether they are needed:
- TRIDENT stage writes segmentation/coords/patch features (e.g. `contours*`, `*_overlap/patches`, `features_uni_v2`).
- Optional representative-tile stage writes selected tile sets (`tile_features/selected_tiles/*.npz`).
- Slide-level package: `slide_features/wsi_slide_features__encoder-<encoder>__mode-<mode>.pt`
- Tile-level package: `tile_features/wsi_tile_features__selector-<tile_selector>__encoder-<tile_encoder>.pt`
- TANGLE stage also keeps tabular exports (`slide_embeddings.{parquet,csv,pkl}`) for inspection.
- These files are used for resume/debug/reuse (e.g. rerun TANGLE without recomputing TRIDENT).
- They can be deleted, but deleted stages must be recomputed in future runs.

Run representative tile selection (SPLICE-style, no tile merge):
```bash
python scripts/encode_wsi_to_slide_embedding.py \
  --run_tile_selection \
  --tile_selection_method splice \
  --tile_selection_top_k 32
```

Supported tile selectors:
- `splice` (default): SPLICE-style representativeness + non-redundancy selection.
- `fps`: diversity-first farthest-point sampling.

Non-CUDA safety guard:
- On non-CUDA devices (Apple MPS/CPU), artifact-removal substage is auto-disabled to avoid TRIDENT CUDA-path crashes.
- This auto-guard is applied even if artifact-removal-related flags are enabled.

Expected input tables:
- `data.pair_table` (CSV/Parquet under data root): `patient_id`, `blood_sample_id`, `tissue_image_path`, `her2_status`, and optional `split` (`train`/`val`).
- `data.blood_feature_table` (CSV/Parquet under data root): one row per blood sample, with `sample_id` + numeric methylation-feature columns.

Key outputs are written to `train.output_dir`:
- `best_model.pt`
- `training_history.csv`
- `embeddings_train.parquet`
- `embeddings_val.parquet`
- `summary.json`

## Config knobs
See `configs/default.yaml` for bin sizes, chromosome lists, QC thresholds, retry counts, and paths.

## Adding real panels
Drop BED files into `ref/region_sets/` and rerun. The methylation proxy node will auto-detect them.

## Caching and resume
Each node fingerprints inputs and config. On cache hit, the node is marked SKIPPED. `resume` reruns failed nodes only; `clean-cache` clears node records.
