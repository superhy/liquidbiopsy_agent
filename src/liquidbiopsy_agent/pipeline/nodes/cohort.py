from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import write_json
from liquidbiopsy_agent.utils.plotting import pca_plot


def robust_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = np.median(np.abs(series - med)) + 1e-6
    return (series - med) / mad


def make_task(run_dir: Path, config: Dict[str, Any]) -> Task:
    features_path = run_dir / "features" / "merged" / "features_table.parquet"
    analysis_dir = run_dir / "analysis"

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        if not features_path.exists():
            return {"skipped": True}
        df = pd.read_parquet(features_path)
        summary = {}
        drift = {}
        numeric_cols = [c for c in df.columns if c != "sample_id" and str(df[c].dtype) != "object"]
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            summary[col] = {
                "median": float(series.median()),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            }
            z = robust_z(series)
            outliers = series[abs(z) > 5]
            if not outliers.empty:
                drift[col] = {sid: float(val) for sid, val in outliers.items()}
        write_json(analysis_dir / "cohort_summary.json", summary)
        write_json(analysis_dir / "drift_flags.json", drift)
        pca_plot(df, "sample_id", analysis_dir / "plots" / "pca.png")
        return {"summary_metrics": len(summary), "drift_flags": len(drift)}

    return Task(
        name="cohort",
        inputs={"features": features_path},
        outputs=[analysis_dir / "cohort_summary.json"],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
