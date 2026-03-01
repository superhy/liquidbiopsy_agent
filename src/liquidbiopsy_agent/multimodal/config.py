from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class MultiModalConfig:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: Optional[Path]) -> "MultiModalConfig":
        if path is None:
            raise ValueError("Config path must be provided")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Config file must decode to a mapping")
        return MultiModalConfig(raw=data)

    def get(self, dotted: str, default: Any = None) -> Any:
        parts = dotted.split(".")
        cur: Any = self.raw
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur
