from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List

from liquidbiopsy_agent.agent.dag import DAGExecutor
from liquidbiopsy_agent.agent.decisions import DecisionEngine
from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.agent.state import TaskStatus
from liquidbiopsy_agent.config import Config
from liquidbiopsy_agent.pipeline.nodes import ingest, manifest, schema, qc, cnv, frag, meth_proxy, merge, cohort, report, agent_review


def build_pipeline(input_path: Path, output_dir: Path, config: Config, instruction: str = "") -> DAGExecutor:
    run_dir = output_dir / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    enable_llm = bool(config.get("llm.enable", False))
    decisions = DecisionEngine(run_dir, enable_llm=enable_llm, instruction=instruction)

    tasks: Dict[str, Task] = {}
    tasks["ingest"] = ingest.make_task(input_path, run_dir, config.get("ingest", {}))
    tasks["manifest"] = manifest.make_task(run_dir, config.get("manifest", {}), decisions)
    tasks["schema"] = schema.make_task(run_dir, config.get("schema", {}), decisions)
    tasks["qc"] = qc.make_task(run_dir, config.get("qc", {}), decisions)
    tasks["cnv"] = cnv.make_task(run_dir, config.get("cnv", {}), decisions)
    tasks["frag"] = frag.make_task(run_dir, config.get("frag", {}), decisions)
    tasks["meth_proxy"] = meth_proxy.make_task(run_dir, config.get("meth_proxy", {}), decisions)
    tasks["merge"] = merge.make_task(run_dir, config.get("merge", {}))
    tasks["cohort"] = cohort.make_task(run_dir, config.get("cohort", {}))
    tasks["agent_review"] = agent_review.make_task(run_dir, config.get("agent_review", {}), decisions)
    tasks["report"] = report.make_task(run_dir, config.get("report", {}))

    edges: Dict[str, List[str]] = {
        "ingest": ["manifest"],
        "manifest": ["schema"],
        "schema": ["qc", "cnv", "frag", "meth_proxy"],
        "qc": ["merge", "cohort"],
        "cnv": ["merge"],
        "frag": ["merge"],
        "meth_proxy": ["merge"],
        "merge": ["cohort"],
        "cohort": ["agent_review"],
        "agent_review": ["report"],
    }

    config_hash = config.hashable()
    return DAGExecutor(tasks, edges, run_dir, config_hash, decisions=decisions)


def resume_pipeline(run_dir: Path, config: Config, instruction: str = "") -> DAGExecutor:
    enable_llm = bool(config.get("llm.enable", False))
    decisions = DecisionEngine(run_dir, enable_llm=enable_llm, instruction=instruction)
    tasks: Dict[str, Task] = {}
    input_path = Path("unused")
    tasks["ingest"] = ingest.make_task(input_path, run_dir, config.get("ingest", {}))
    tasks["manifest"] = manifest.make_task(run_dir, config.get("manifest", {}), decisions)
    tasks["schema"] = schema.make_task(run_dir, config.get("schema", {}), decisions)
    tasks["qc"] = qc.make_task(run_dir, config.get("qc", {}), decisions)
    tasks["cnv"] = cnv.make_task(run_dir, config.get("cnv", {}), decisions)
    tasks["frag"] = frag.make_task(run_dir, config.get("frag", {}), decisions)
    tasks["meth_proxy"] = meth_proxy.make_task(run_dir, config.get("meth_proxy", {}), decisions)
    tasks["merge"] = merge.make_task(run_dir, config.get("merge", {}))
    tasks["cohort"] = cohort.make_task(run_dir, config.get("cohort", {}))
    tasks["agent_review"] = agent_review.make_task(run_dir, config.get("agent_review", {}), decisions)
    tasks["report"] = report.make_task(run_dir, config.get("report", {}))

    edges: Dict[str, List[str]] = {
        "ingest": ["manifest"],
        "manifest": ["schema"],
        "schema": ["qc", "cnv", "frag", "meth_proxy"],
        "qc": ["merge", "cohort"],
        "cnv": ["merge"],
        "frag": ["merge"],
        "meth_proxy": ["merge"],
        "merge": ["cohort"],
        "cohort": ["agent_review"],
        "agent_review": ["report"],
    }
    return DAGExecutor(tasks, edges, run_dir, config.hashable(), decisions=decisions)
