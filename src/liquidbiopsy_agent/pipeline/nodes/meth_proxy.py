from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import write_parquet


def load_regions(path: Path) -> Dict[str, List[Tuple[int, int]]]:
    df = pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end"])
    regions: Dict[str, List[Tuple[int, int]]] = {}
    for _, row in df.iterrows():
        regions.setdefault(str(row["chrom"]), []).append((int(row["start"]), int(row["end"])))
    return regions


def count_midpoints(sample_path: Path, regions: Dict[str, List[Tuple[int, int]]]) -> int:
    try:
        import pyranges as pr
    except ImportError:
        return count_midpoints_fallback(sample_path, regions)

    region_rows = []
    for chrom, spans in regions.items():
        for start, end in spans:
            region_rows.append({"Chromosome": chrom, "Start": start, "End": end})
    region_pr = pr.PyRanges(pd.DataFrame(region_rows))
    sample_df = pd.read_csv(sample_path, sep="\t", header=None, names=["Chromosome", "Start", "End"])
    sample_df["Mid"] = (sample_df["Start"] + sample_df["End"]) // 2
    sample_df["Start"] = sample_df["Mid"]
    sample_df["End"] = sample_df["Mid"] + 1
    sample_pr = pr.PyRanges(sample_df[["Chromosome", "Start", "End"]])
    joined = sample_pr.join(region_pr)
    return len(joined)


def count_midpoints_fallback(sample_path: Path, regions: Dict[str, List[Tuple[int, int]]]) -> int:
    total = 0
    for chunk in pd.read_csv(sample_path, sep="\t", header=None, names=["chrom", "start", "end"], chunksize=200_000):
        chunk["mid"] = (chunk["start"] + chunk["end"]) // 2
        for chrom, group in chunk.groupby("chrom"):
            if chrom not in regions:
                continue
            intervals = regions[chrom]
            for _, row in group.iterrows():
                mid = int(row["mid"])
                for start, end in intervals:
                    if start <= mid < end:
                        total += 1
                        break
    return total


def gini(values: List[int]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    cum = 0
    for i, v in enumerate(arr, 1):
        cum += i * v
    return (2 * cum) / (n * sum(arr)) - (n + 1) / n


def make_task(run_dir: Path, config: Dict[str, Any], decisions) -> Task:
    processed_dir = run_dir / "data" / "processed_bed"
    meth_dir = run_dir / "features" / "meth_proxy"
    project_root = Path(__file__).resolve().parents[4]
    panels_dir = project_root / "ref" / "region_sets"

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        panel_files = list(panels_dir.glob("*.bed"))
        if not panel_files:
            return {"skipped": True, "reason": "no panels"}
        decision = decisions.meth_panel_policy([p.name for p in panel_files])
        selected = [p for p in panel_files if p.name in decision.get("panels", [])]
        summaries = []
        samples = list(processed_dir.glob("*.bed.gz"))
        for sample_path in tqdm(samples, desc="meth_proxy"):
            sid = sample_path.name.replace(".bed.gz", "")
            sample_summary = {"sample_id": sid}
            for panel in selected:
                regions = load_regions(panel)
                count = count_midpoints(sample_path, regions)
                vals = [count]
                sample_summary[f"{panel.stem}_mean_count"] = float(count)
                sample_summary[f"{panel.stem}_gini"] = float(gini(vals))
            summaries.append(sample_summary)
            write_parquet(pd.DataFrame([sample_summary]), meth_dir / f"{sid}_summary.parquet")
        if summaries:
            write_parquet(pd.DataFrame(summaries), meth_dir / "meth_proxy_summary.parquet")
        return {"samples": len(summaries), "panels": [p.name for p in selected]}

    return Task(
        name="meth_proxy",
        inputs={"processed_dir": processed_dir},
        outputs=[meth_dir],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
