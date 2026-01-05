from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.hashing import file_sha256
from liquidbiopsy_agent.utils.io import write_parquet


def infer_assay(filename: str, rules: List[str], default: str) -> str:
    for rule in rules:
        if re.search(rule, filename, re.IGNORECASE):
            return rule
    return default


def make_task(run_dir: Path, config: Dict[str, Any], decisions) -> Task:
    raw_dir = run_dir / "data" / "raw"
    extracted_dir = run_dir / "data" / "raw_extracted"
    manifest_path = run_dir / "data" / "manifest" / "manifest.parquet"

    outputs = [manifest_path]

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        files = list(raw_dir.glob("*.bed.gz")) + list(extracted_dir.glob("**/*.bed.gz"))
        filenames = [f.name for f in files]
        decision = decisions.assay_type_rules(filenames)
        data = []
        sha_flag = config_section.get("compute_sha", False)
        for f in tqdm(files, desc="manifest"):
            sample_id = f.name.replace(".bed.gz", "")
            assay_type = infer_assay(f.name, decision.get("rules", []), decision.get("default", "unknown"))
            entry = {
                "sample_id": sample_id,
                "file_path": str(f.resolve()),
                "file_size": f.stat().st_size,
                "sha256": file_sha256(f) if sha_flag else None,
                "assay_type": assay_type,
                "group_label": None,
                "genome_build": config_section.get("genome_build", "unknown"),
            }
            data.append(entry)
        df = pd.DataFrame(data)
        write_parquet(df, manifest_path)
        return {"n_samples": len(df), "manifest": str(manifest_path)}

    return Task(
        name="manifest",
        inputs={"raw_dir": raw_dir},
        outputs=outputs,
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
