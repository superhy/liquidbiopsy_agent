from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Dict, Any

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import list_tar_members, write_json


def make_task(input_path: Path, run_dir: Path, config: Dict[str, Any]) -> Task:
    raw_dir = run_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    outputs = [raw_dir]

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        in_path = Path(inputs["input_path"])
        summary: Dict[str, Any] = {"source": str(in_path)}
        if tarfile.is_tarfile(in_path):
            dest_tar = raw_dir / in_path.name
            shutil.copy2(in_path, dest_tar)
            members = list_tar_members(dest_tar)
            summary["tar_members"] = members[:20]
            summary["tar_total"] = len(members)
            write_json(raw_dir / "tar_members.json", {"members": members})
            extract = [m for m in members if m.endswith(".bed.gz")]
            extract_dir = run_dir / "data" / "raw_extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(dest_tar, "r") as tar:
                for mem in tar.getmembers():
                    if mem.name in extract:
                        tar.extract(mem, extract_dir)
            summary["extracted_bed"] = len(extract)
        else:
            for file in in_path.glob("**/*"):
                if file.is_file():
                    dest = raw_dir / file.name
                    shutil.copy2(file, dest)
            summary["copied_files"] = len(list(raw_dir.glob("*")))
        return summary

    return Task(
        name="ingest",
        inputs={"input_path": input_path},
        outputs=[raw_dir],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
