from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import write_json


BED_COLS = ["chrom", "start", "end"]


def normalise_chrom(chrom: str) -> str:
    if chrom.startswith("chr"):
        return chrom
    return f"chr{chrom}"


def process_file(path: Path, out_path: Path, prefer_cols: List[int]) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    lengths: List[int] = []
    chrom_counts: Dict[str, int] = {}
    chunksize = 200_000
    with gzip.open(out_path, "wt") as out_f:
        for chunk in pd.read_csv(path, sep="\t", header=None, chunksize=chunksize, comment="#", dtype={0: str}):
            cols = chunk.columns
            if len(cols) < 3:
                continue
            selected = chunk.iloc[:, prefer_cols[:3]] if len(cols) > max(prefer_cols) else chunk.iloc[:, :3]
            selected.columns = BED_COLS
            selected["chrom"] = selected["chrom"].astype(str).map(normalise_chrom)
            selected["start"] = selected["start"].astype(int)
            selected["end"] = selected["end"].astype(int)
            selected = selected[selected["end"] > selected["start"]]
            selected.to_csv(out_f, sep="\t", header=False, index=False)
            total_rows += len(selected)
            lens = (selected["end"] - selected["start"]).astype(int)
            lengths.extend(lens.tolist())
            for c, count in selected["chrom"].value_counts().items():
                chrom_counts[c] = chrom_counts.get(c, 0) + int(count)
    return {"rows": total_rows, "length_stats": {"min": min(lengths) if lengths else None, "max": max(lengths) if lengths else None}, "chrom_counts": chrom_counts}


def make_task(run_dir: Path, config: Dict[str, Any], decisions) -> Task:
    manifest_path = run_dir / "data" / "manifest" / "manifest.parquet"
    processed_dir = run_dir / "data" / "processed_bed"
    audit_path = run_dir / "data" / "schema_audit.jsonl"

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        import pandas as pd

        manifest = pd.read_parquet(manifest_path)
        audit_entries = []
        prefer_cols = decisions.schema_policy([]).get("prefer_columns", [0, 1, 2])
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="schema"):
            src = Path(row["file_path"])
            dest = processed_dir / f"{row['sample_id']}.bed.gz"
            stats = process_file(src, dest, prefer_cols)
            audit_entries.append({"sample_id": row["sample_id"], **stats})
        with open(audit_path, "w", encoding="utf-8") as f:
            for entry in audit_entries:
                f.write(json.dumps(entry) + "\n")
        return {"processed": len(audit_entries), "audit": str(audit_path)}

    return Task(
        name="schema",
        inputs={"manifest": manifest_path},
        outputs=[processed_dir, audit_path],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
