from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: Optional[Path]) -> "Config":
        if path is None:
            raise ValueError("Config path must be provided")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Config(raw=data)

    def get(self, dotted: str, default: Any = None) -> Any:
        parts = dotted.split(".")
        cur: Any = self.raw
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur

    def to_json(self) -> str:
        return json.dumps(self.raw, sort_keys=True)

    def hashable(self) -> str:
        return self.to_json()
