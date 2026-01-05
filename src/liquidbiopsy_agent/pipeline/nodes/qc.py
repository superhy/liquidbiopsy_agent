from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import write_parquet
from liquidbiopsy_agent.utils.plotting import save_boxplot, save_histogram, save_barplot, save_heatmap


def compute_qc(sample_id: str, path: Path, bin_size: int, chromosomes: list[str]) -> Dict[str, Any]:
    lengths = []
    chrom_counts: Dict[str, int] = {}
    bin_counts: Dict[int, int] = {}
    for chunk in pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end"], chunksize=200_000):
        chunk["length"] = chunk["end"] - chunk["start"]
        lengths.extend(chunk["length"].tolist())
        for c, count in chunk["chrom"].value_counts().items():
            chrom_counts[c] = chrom_counts.get(c, 0) + int(count)
        mid = (chunk["start"] + chunk["end"]) // 2
        bins = (mid // bin_size).astype(int)
        for b, count in bins.value_counts().items():
            bin_counts[int(b)] = bin_counts.get(int(b), 0) + int(count)
    if not lengths:
        return {
            "sample_id": sample_id,
            "n_fragments": 0,
            "len_mean": None,
            "len_median": None,
            "len_iqr": None,
            "short_ratio_100_150": None,
            "chrY_fraction": None,
            "chrM_fraction": None,
            "coverage_uniformity_cv": None,
        }
    arr = np.array(lengths)
    coverage_cv = float(np.std(list(bin_counts.values())) / (np.mean(list(bin_counts.values())) + 1e-6)) if bin_counts else None
    total = len(arr)
    return {
        "sample_id": sample_id,
        "n_fragments": int(total),
        "len_mean": float(arr.mean()),
        "len_median": float(np.median(arr)),
        "len_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "short_ratio_100_150": float(((arr >= 100) & (arr <= 150)).sum() / total),
        "chrY_fraction": float(chrom_counts.get("chrY", 0) / total),
        "chrM_fraction": float(chrom_counts.get("chrM", 0) / total),
        "coverage_uniformity_cv": coverage_cv,
    }


def make_task(run_dir: Path, config: Dict[str, Any], decisions) -> Task:
    processed_dir = run_dir / "data" / "processed_bed"
    qc_table = run_dir / "qc" / "sample_qc.parquet"
    plots_dir = run_dir / "qc" / "plots"

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        samples = list(processed_dir.glob("*.bed.gz"))
        bin_size = int(config_section.get("bin_size", 1_000_000))
        chromosomes = config_section.get("chromosomes", [])
        qc_entries = []
        for path in tqdm(samples, desc="qc"):
            sid = path.name.replace(".bed.gz", "")
            qc_entries.append(compute_qc(sid, path, bin_size, chromosomes))
        df = pd.DataFrame(qc_entries)
        thresholds = decisions.qc_thresholds(df.to_dict(orient="records"))
        df["qc_pass"] = (df["n_fragments"] >= thresholds.get("min_fragments", 0)) & (
            df["chrY_fraction"] <= thresholds.get("max_chrY_frac", 1.0)
        )
        write_parquet(df, qc_table)

        chrom_fraction_rows = []
        plot_policy = decisions.plot_policy("qc", len(df), config_section)
        bin_heatmap_rows = []
        for _, row in df.iterrows():
            sid = row["sample_id"]
            lengths = []
            for chunk in pd.read_csv(processed_dir / f"{sid}.bed.gz", sep="\t", header=None, names=["chrom", "start", "end"], chunksize=200_000):
                lengths.extend((chunk["end"] - chunk["start"]).tolist())
            if lengths:
                save_histogram(pd.Series(lengths), f"{sid} length", "Length", plots_dir / f"{sid}_length.png", bins=50)
            chrom_counts = {}
            for chunk in pd.read_csv(processed_dir / f"{sid}.bed.gz", sep="\t", header=None, names=["chrom", "start", "end"], chunksize=200_000):
                for c, count in chunk["chrom"].value_counts().items():
                    chrom_counts[c] = chrom_counts.get(c, 0) + int(count)
            save_barplot(pd.Series(chrom_counts), f"{sid} chrom fractions", plots_dir / f"{sid}_chr_fraction.png")
            total = sum(chrom_counts.values()) or 1
            chrom_fraction_rows.append(
                {"sample_id": sid, **{chrom: chrom_counts.get(chrom, 0) / total for chrom in chromosomes}}
            )
            if plot_policy.get("enable") and plot_policy.get("qc_bin_heatmap"):
                if len(bin_heatmap_rows) < plot_policy.get("max_samples", 20):
                    bin_counts = {}
                    for chunk in pd.read_csv(processed_dir / f"{sid}.bed.gz", sep="\t", header=None, names=["chrom", "start", "end"], chunksize=200_000):
                        mid = (chunk["start"] + chunk["end"]) // 2
                        bins = (mid // bin_size).astype(int)
                        for b, count in bins.value_counts().items():
                            bin_counts[int(b)] = bin_counts.get(int(b), 0) + int(count)
                    max_bins = plot_policy.get("max_bins", 200)
                    row_bins = {f"bin_{k}": bin_counts.get(k, 0) for k in range(max_bins)}
                    bin_heatmap_rows.append({"sample_id": sid, **row_bins})
        if not df.empty:
            save_boxplot(df, "len_median", plots_dir / "cohort_length_box.png")
        if chrom_fraction_rows:
            chrom_df = pd.DataFrame(chrom_fraction_rows).set_index("sample_id")
            save_heatmap(chrom_df, "Coverage uniformity (chromosome fractions)", plots_dir / "coverage_uniformity_heatmap.png")
        if bin_heatmap_rows and plot_policy.get("enable") and plot_policy.get("qc_bin_heatmap"):
            bin_df = pd.DataFrame(bin_heatmap_rows).set_index("sample_id")
            save_heatmap(bin_df, "Coverage uniformity (bin-level)", plots_dir / "coverage_bin_heatmap.png")
        return {"samples": len(df), "qc_table": str(qc_table)}

    return Task(
        name="qc",
        inputs={"processed_dir": processed_dir},
        outputs=[qc_table, plots_dir],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
