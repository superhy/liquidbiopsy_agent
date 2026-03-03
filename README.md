# LiquidBiopsy Agent (BED-only MVP)

Agentised, cache-aware cfDNA pipeline that ingests BED(.bed.gz) fragment coordinates and produces QC, CNV, fragmentomics, methylation-proxy features, cohort summaries, and an HTML report. Runs on CPU-only workstations; LLM integration is optional and degrades gracefully. The pipeline is orchestrated with LangGraph for extensible agent-style execution without LangChain.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data Root Policy
This repository is code-only. Do not store datasets or model checkpoints in git.
All dataset/checkpoint/output paths are resolved under one server root:
- global override (highest priority): `LIQUID_BIOPSY_DATA_ROOT`
- Ubuntu placeholder: `/home/YOUR_UBUNTU_USER/liquid-agent-data`
- macOS placeholder: `/Users/YOUR_MACOS_USER/liquid-agent-data`
- Windows 11 placeholder: `C:\Users\YOUR_WINDOWS_USER\liquid-agent-data`
- optional OS-specific overrides:
  - `LIQUID_BIOPSY_DATA_ROOT_UBUNTU`
  - `LIQUID_BIOPSY_DATA_ROOT_MACOS`
  - `LIQUID_BIOPSY_DATA_ROOT_WINDOWS`
- current top-level dataset folders under root: `GSE243474`, `TCGA-BRCA`

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
