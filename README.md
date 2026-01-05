# LiquidBiopsy Agent (BED-only MVP)

Agentised, cache-aware cfDNA pipeline that ingests BED(.bed.gz) fragment coordinates and produces QC, CNV, fragmentomics, methylation-proxy features, cohort summaries, and an HTML report. Runs on CPU-only workstations; LLM integration is optional and degrades gracefully. The pipeline is orchestrated with LangGraph for extensible agent-style execution without LangChain.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart
```bash
python -m liquidbiopsy_agent run --input /path/to/bed_dir --output output --config configs/default.yaml
python -m liquidbiopsy_agent run --input /path/to/bed_dir --output output --instruction "开启所有图"
python -m liquidbiopsy_agent status --run-dir output/run_YYYYMMDD_HHMMSS
python -m liquidbiopsy_agent resume --run-dir output/run_YYYYMMDD_HHMMSS --instruction "no cnv"
```
Supports tar bundles too: `--input GSE243474_RAW.tar`.

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
export LIQUIDBIOPSY_LLM_MODEL_PATH=/path/to/your/model.gguf
```

## Agent interpretation outputs
The agent writes intermediate, interpretable outputs to:
- `analysis/agent/parameter_tuning.json`
- `analysis/agent/rerun_plan.json`
- `analysis/agent/explanation.txt`
These are intended to support biomedical review and parameter refinement between runs.

## Config knobs
See `configs/default.yaml` for bin sizes, chromosome lists, QC thresholds, retry counts, and paths.

## Adding real panels
Drop BED files into `ref/region_sets/` and rerun. The methylation proxy node will auto-detect them.

## Caching and resume
Each node fingerprints inputs and config. On cache hit, the node is marked SKIPPED. `resume` reruns failed nodes only; `clean-cache` clears node records.
