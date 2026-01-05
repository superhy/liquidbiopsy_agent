from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


def fingerprint_paths(paths: Iterable[Path]) -> str:
    sha = hashlib.sha256()
    for p in sorted(paths):
        sha.update(str(p).encode())
        if p.exists():
            stat = p.stat()
            sha.update(str(stat.st_mtime_ns).encode())
            sha.update(str(stat.st_size).encode())
        else:
            sha.update(b"missing")
    return sha.hexdigest()


def fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def combine_hash(parts: Iterable[str]) -> str:
    sha = hashlib.sha256()
    for part in parts:
        sha.update(part.encode())
    return sha.hexdigest()
