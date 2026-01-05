from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def list_tar_members(tar_path: Path) -> List[str]:
    with tarfile.open(tar_path, "r") as tar:
        return tar.getnames()


def extract_tar_members(tar_path: Path, members: Iterable[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        selected = [m for m in tar.getmembers() if m.name in members]
        tar.extractall(path=dest, members=selected)
