from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import write_json


def make_task(run_dir: Path, config: Dict[str, Any], decisions) -> Task:
    qc_path = run_dir / "qc" / "sample_qc.parquet"
    cnv_path = run_dir / "features" / "cnv" / "cnv_summary.parquet"
    drift_path = run_dir / "analysis" / "drift_flags.json"
    agent_dir = run_dir / "analysis" / "agent"
    summary_path = agent_dir / "agent_summary.json"
    explanation_path = agent_dir / "explanation.txt"
    tuning_path = agent_dir / "parameter_tuning.json"
    rerun_path = agent_dir / "rerun_plan.json"

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        qc = pd.read_parquet(qc_path).to_dict(orient="records") if qc_path.exists() else []
        cnv = pd.read_parquet(cnv_path).to_dict(orient="records") if cnv_path.exists() else []
        drift = {}
        if drift_path.exists():
            with open(drift_path, "r", encoding="utf-8") as f:
                drift = json.load(f)
        payload = {"qc": qc, "cnv": cnv, "drift": drift}
        result = decisions.agent_summary(payload)
        agent_dir.mkdir(parents=True, exist_ok=True)
        write_json(summary_path, result)
        write_json(tuning_path, result.get("parameter_tuning", []))
        write_json(rerun_path, result.get("rerun_samples", []))
        with open(explanation_path, "w", encoding="utf-8") as f:
            f.write(result.get("explanation", ""))
        return {"agent_summary": str(summary_path)}

    return Task(
        name="agent_review",
        inputs={"qc": qc_path, "cnv": cnv_path, "drift": drift_path},
        outputs=[summary_path, explanation_path, tuning_path, rerun_path],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
