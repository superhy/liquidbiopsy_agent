from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import write_csv, write_parquet, write_json


def make_task(run_dir: Path, config: Dict[str, Any]) -> Task:
    qc_path = run_dir / "qc" / "sample_qc.parquet"
    cnv_path = run_dir / "features" / "cnv" / "cnv_summary.parquet"
    frag_path = run_dir / "features" / "frag" / "frag_summary.parquet"
    meth_path = run_dir / "features" / "meth_proxy" / "meth_proxy_summary.parquet"
    out_dir = run_dir / "features" / "merged"
    features_parquet = out_dir / "features_table.parquet"
    features_csv = out_dir / "features_table.csv"

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        dfs = []
        if qc_path.exists():
            dfs.append(pd.read_parquet(qc_path))
        if cnv_path.exists():
            dfs.append(pd.read_parquet(cnv_path))
        if frag_path.exists():
            dfs.append(pd.read_parquet(frag_path))
        if meth_path.exists():
            dfs.append(pd.read_parquet(meth_path))
        if not dfs:
            return {"merged": 0}
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="sample_id", how="left")
        write_parquet(merged, features_parquet)
        write_csv(merged, features_csv)
        schema = {col: str(dtype) for col, dtype in zip(merged.columns, merged.dtypes)}
        write_json(out_dir / "feature_schema.json", schema)
        return {"merged": len(merged), "features_path": str(features_parquet)}

    return Task(
        name="merge",
        inputs={"qc": qc_path, "cnv": cnv_path, "frag": frag_path, "meth": meth_path},
        outputs=[features_parquet, features_csv],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
