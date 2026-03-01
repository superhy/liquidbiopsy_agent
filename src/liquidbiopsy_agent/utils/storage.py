from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "LIQUID_BIOPSY_DATA_ROOT"
DEFAULT_DATA_ROOT = Path("/liquid-biopsy-data")


def get_data_root() -> Path:
    """Return the canonical server data root directory."""
    raw = os.getenv(DATA_ROOT_ENV, str(DEFAULT_DATA_ROOT))
    return Path(raw).expanduser().resolve()


def ensure_within_data_root(path: Path, *, path_kind: str = "path") -> Path:
    """Validate that a path is inside the configured data root."""
    root = get_data_root()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"{path_kind} must be inside data root '{root}', got '{resolved}'. "
            f"Set {DATA_ROOT_ENV} if the server root is different."
        ) from e
    return resolved


def resolve_data_path(
    path_value: str | Path,
    *,
    path_kind: str = "path",
    must_exist: bool = False,
) -> Path:
    """Resolve path against data root and enforce root containment.

    Relative paths are resolved under the configured data root.
    Absolute paths are accepted only if they are inside the data root.
    """
    if path_value is None:
        raise ValueError(f"{path_kind} cannot be None")

    root = get_data_root()
    path = Path(path_value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (root / path).resolve()

    resolved = ensure_within_data_root(resolved, path_kind=path_kind)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{path_kind} does not exist: {resolved}")
    return resolved
