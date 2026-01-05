from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict
from tqdm import tqdm

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import write_parquet
from liquidbiopsy_agent.utils.plotting import save_multi_hist, save_violinplot


def length_histogram(path: Path, bins: list[int]) -> pd.DataFrame:
    lengths = []
    for chunk in pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end"], chunksize=200_000):
        lengths.extend((chunk["end"] - chunk["start"]).astype(int).tolist())
    if not lengths:
        return pd.DataFrame(columns=["bin", "count"])
    hist, edges = np.histogram(lengths, bins=bins)
    return pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "count": hist})


def summarise_lengths(df: pd.DataFrame) -> Dict[str, Any]:
    lengths = []
    for _, row in df.iterrows():
        bin_mid = (row["bin_start"] + row["bin_end"]) / 2
        lengths.extend([bin_mid] * int(row["count"]))
    if not lengths:
        return {}
    arr = np.array(lengths)
    mode_idx = np.argmax(np.bincount(arr.astype(int)))
    total = len(arr)
    return {
        "len_median": float(np.median(arr)),
        "len_mode": float(mode_idx),
        "len_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "short_100_150_ratio": float(((arr >= 100) & (arr <= 150)).sum() / total),
        "mono_peak_strength_proxy": float(((arr >= 150) & (arr <= 190)).sum() / total),
        "di_ratio_proxy": float(((arr >= 300) & (arr <= 340)).sum() / total),
        "frag_entropy": float(-np.sum((np.histogram(arr, bins=50)[0] / total) * np.log2((np.histogram(arr, bins=50)[0] + 1e-9) / total))),
    }


def make_task(run_dir: Path, config: Dict[str, Any], decisions) -> Task:
    processed_dir = run_dir / "data" / "processed_bed"
    frag_dir = run_dir / "features" / "frag"
    plots_dir = frag_dir / "plots"

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        samples = list(processed_dir.glob("*.bed.gz"))
        decision = decisions.frag_feature_policy(len(samples))
        bins = decision.get("bins") or list(range(50, 505, 5))
        summaries = []
        plot_series = []
        plot_labels = []
        violin_rows = []
        plot_policy = decisions.plot_policy("frag", len(samples), config_section)
        for path in tqdm(samples, desc="frag"):
            sid = path.name.replace(".bed.gz", "")
            hist_df = length_histogram(path, bins)
            summary = summarise_lengths(hist_df)
            summary["sample_id"] = sid
            write_parquet(hist_df, frag_dir / f"{sid}_length_hist.parquet")
            write_parquet(pd.DataFrame([summary]), frag_dir / f"{sid}_summary.parquet")
            summaries.append(summary)
            if len(plot_series) < 5 and not hist_df.empty:
                mids = (hist_df["bin_start"] + hist_df["bin_end"]) / 2
                probs = hist_df["count"] / (hist_df["count"].sum() + 1e-9)
                sample = np.random.choice(mids, size=5000, replace=True, p=probs)
                plot_series.append(pd.Series(sample))
                plot_labels.append(sid)
            if plot_policy.get("enable") and plot_policy.get("frag_violin"):
                if len(violin_rows) < plot_policy.get("max_samples", 20) and not hist_df.empty:
                    mids = (hist_df["bin_start"] + hist_df["bin_end"]) / 2
                    probs = hist_df["count"] / (hist_df["count"].sum() + 1e-9)
                    sample = np.random.choice(mids, size=3000, replace=True, p=probs)
                    for value in sample:
                        violin_rows.append({"sample_id": sid, "length": float(value)})
        if summaries:
            write_parquet(pd.DataFrame(summaries), frag_dir / "frag_summary.parquet")
        if plot_series:
            save_multi_hist(plot_series, plot_labels, "Length peak comparison (subset)", "Length", plots_dir / "length_peak_comparison.png")
        if violin_rows and plot_policy.get("enable") and plot_policy.get("frag_violin"):
            violin_df = pd.DataFrame(violin_rows)
            save_violinplot(violin_df, "sample_id", "length", "Length distribution (violin)", plots_dir / "length_violin.png")
        return {"samples": len(summaries)}

    return Task(
        name="frag",
        inputs={"processed_dir": processed_dir},
        outputs=[frag_dir],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
