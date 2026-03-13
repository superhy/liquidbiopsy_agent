from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict
from tqdm import tqdm

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import write_parquet
from liquidbiopsy_agent.utils.plotting import save_lineplot, save_cnv_chrom_plot


def bin_counts(path: Path, bin_size: int) -> pd.DataFrame:
    records = []
    for chunk in pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end"], chunksize=200_000):
        # Robust BED parsing: tolerate track/header rows and malformed coordinates.
        chunk["start"] = pd.to_numeric(chunk["start"], errors="coerce")
        chunk["end"] = pd.to_numeric(chunk["end"], errors="coerce")
        chunk = chunk.dropna(subset=["chrom", "start", "end"])
        if chunk.empty:
            continue
        chunk["start"] = chunk["start"].astype(np.int64)
        chunk["end"] = chunk["end"].astype(np.int64)
        chunk = chunk[chunk["end"] > chunk["start"]]
        if chunk.empty:
            continue
        mid = (chunk["start"] + chunk["end"]) // 2
        bins = mid // bin_size
        records.append(pd.DataFrame({"bin": bins, "chrom": chunk["chrom"]}))
    if not records:
        return pd.DataFrame(columns=["bin", "chrom", "count"])
    df = pd.concat(records)
    grouped = df.groupby(["chrom", "bin"]).size().reset_index(name="count")
    grouped["start"] = grouped["bin"] * bin_size
    grouped["end"] = grouped["start"] + bin_size
    return grouped[["chrom", "start", "end", "count"]]


def summarise_counts(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"cnv_mad": None, "cnv_var": None, "amp_burden": None, "del_burden": None, "seg_count": 0, "aneuploidy_score_proxy": None}
    counts = df["count"].values
    median = np.median(counts)
    mad = np.median(np.abs(counts - median))
    var = float(np.var(counts))
    z = (counts - median) / (mad + 1e-6)
    amp = float((z > 3).sum() / len(z))
    dele = float((z < -3).sum() / len(z))
    smoothed = pd.Series(counts).rolling(window=3, min_periods=1, center=True).mean()
    seg = int(((np.diff(np.sign(np.diff(smoothed))) != 0).sum()))
    return {
        "cnv_mad": float(mad),
        "cnv_var": var,
        "amp_burden": amp,
        "del_burden": dele,
        "seg_count": seg,
        "aneuploidy_score_proxy": float(np.std(smoothed) / (np.mean(smoothed) + 1e-6)),
    }


def make_task(run_dir: Path, config: Dict[str, Any], decisions) -> Task:
    processed_dir = run_dir / "data" / "processed_bed"
    cnv_dir = run_dir / "features" / "cnv"
    plots_dir = cnv_dir / "plots"

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        samples = list(processed_dir.glob("*.bed.gz"))
        summaries = []
        plot_policy = decisions.plot_policy("cnv", len(samples), config_section)
        for path in tqdm(samples, desc="cnv"):
            sid = path.name.replace(".bed.gz", "")
            decision = decisions.cnv_policy(n_fragments=0, uniformity=0.0)
            bin_size = int(decision.get("bin_size", config_section.get("bin_size", 1_000_000)))
            counts_df = bin_counts(path, bin_size)
            summary = summarise_counts(counts_df)
            summary["sample_id"] = sid
            bin_path = cnv_dir / f"{sid}_bin_counts.parquet"
            write_parquet(counts_df, bin_path)
            if not counts_df.empty:
                counts_df = counts_df.sort_values(["chrom", "start"])
                x = pd.Series(range(len(counts_df)))
                y = counts_df["count"] / (counts_df["count"].mean() + 1e-6)
                smoothed = y.rolling(window=5, min_periods=1, center=True).mean()
                save_lineplot(x, smoothed, f"{sid} CNV profile (smoothed)", "Bin index", "Normalised count", plots_dir / f"{sid}_cnv_profile.png")
                if plot_policy.get("enable") and plot_policy.get("cnv_chrom_plot"):
                    save_cnv_chrom_plot(counts_df, plots_dir / f"{sid}_cnv_chrom_plot.png", f"{sid} CNV by chromosome")
            summaries.append(summary)
            write_parquet(pd.DataFrame([summary]), cnv_dir / f"{sid}_summary.parquet")
        summary_df = pd.DataFrame(summaries)
        if not summary_df.empty:
            write_parquet(summary_df, cnv_dir / "cnv_summary.parquet")
        return {"samples": len(summaries), "bin_size": config_section.get("bin_size", 1_000_000)}

    return Task(
        name="cnv",
        inputs={"processed_dir": processed_dir},
        outputs=[cnv_dir],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
