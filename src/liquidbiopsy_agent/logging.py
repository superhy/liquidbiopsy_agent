import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setFormatter(formatter)
    root.addHandler(fh)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
