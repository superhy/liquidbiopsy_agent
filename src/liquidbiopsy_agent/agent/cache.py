from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from liquidbiopsy_agent.utils.hashing import combine_hash, fingerprint_bytes, fingerprint_paths


def compute_fingerprint(inputs: Dict[str, Any], config_section: Dict[str, Any]) -> str:
    path_inputs: Iterable[Path] = []
    byte_parts = []
    for v in inputs.values():
        if isinstance(v, (str, Path)):
            p = Path(v)
            if p.exists():
                path_inputs = list(path_inputs) + [p]
            else:
                byte_parts.append(str(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, (str, Path)):
                    p = Path(item)
                    if p.exists():
                        path_inputs = list(path_inputs) + [p]
                    else:
                        byte_parts.append(str(item))
        else:
            byte_parts.append(json.dumps(v, sort_keys=True))
    paths_hash = fingerprint_paths(path_inputs)
    config_hash = fingerprint_bytes(json.dumps(config_section, sort_keys=True).encode())
    return combine_hash([paths_hash, config_hash, *byte_parts])
