from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .state import TaskStatus
from .cache import compute_fingerprint
from liquidbiopsy_agent.utils.io import write_json


@dataclass
class TaskRecord:
    name: str
    status: TaskStatus
    started: Optional[str]
    ended: Optional[str]
    inputs: Dict[str, Any]
    outputs: List[str]
    fingerprint: str
    config_hash: str
    summary: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class Task:
    name: str
    inputs: Dict[str, Any]
    outputs: List[Path]
    config_section: Dict[str, Any]
    run_fn: Callable[[Dict[str, Any], Dict[str, Any], Path], Dict[str, Any]]
    retries: int = 0

    def should_skip(self, fingerprint: str, record_path: Path) -> bool:
        if not record_path.exists():
            return False
        try:
            with open(record_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            return prev.get("fingerprint") == fingerprint and all(Path(p).exists() for p in prev.get("outputs", []))
        except Exception:
            return False

    def run(self, run_dir: Path, config_hash: str) -> TaskRecord:
        logs_dir = run_dir / "logs" / "nodes"
        logs_dir.mkdir(parents=True, exist_ok=True)
        record_path = logs_dir / f"{self.name}.json"
        fingerprint = compute_fingerprint(self.inputs, self.config_section)

        if self.should_skip(fingerprint, record_path):
            return TaskRecord(
                name=self.name,
                status=TaskStatus.SKIPPED,
                started=None,
                ended=None,
                inputs=self.inputs,
                outputs=[str(p) for p in self.outputs],
                fingerprint=fingerprint,
                config_hash=config_hash,
                summary={"cached": True},
                error=None,
            )

        attempts = 0
        error_msg = None
        summary: Dict[str, Any] = {}
        while attempts <= self.retries:
            attempts += 1
            started = dt.datetime.utcnow().isoformat()
            try:
                summary = self.run_fn(self.inputs, self.config_section, run_dir)
                ended = dt.datetime.utcnow().isoformat()
                record = TaskRecord(
                    name=self.name,
                    status=TaskStatus.SUCCESS,
                    started=started,
                    ended=ended,
                    inputs=self.inputs,
                    outputs=[str(p) for p in self.outputs],
                    fingerprint=fingerprint,
                    config_hash=config_hash,
                    summary=summary,
                )
                write_json(record_path, record.__dict__)
                return record
            except Exception as e:  # pragma: no cover - defensive
                error_msg = str(e)
                if attempts > self.retries:
                    ended = dt.datetime.utcnow().isoformat()
                    record = TaskRecord(
                        name=self.name,
                        status=TaskStatus.FAILED,
                        started=started,
                        ended=ended,
                        inputs=self.inputs,
                        outputs=[str(p) for p in self.outputs],
                        fingerprint=fingerprint,
                        config_hash=config_hash,
                        summary=summary,
                        error=error_msg,
                    )
                    write_json(record_path, record.__dict__)
                    return record

        raise RuntimeError(f"Task {self.name} failed unexpectedly")
