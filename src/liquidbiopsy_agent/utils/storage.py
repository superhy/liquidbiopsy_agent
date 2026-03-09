from __future__ import annotations

import os
import platform
from pathlib import Path


DATA_ROOT_ENV = "LIQUID_BIOPSY_DATA_ROOT"
# Optional OS-specific env vars (fallbacks when LIQUID_BIOPSY_DATA_ROOT is not set).
DATA_ROOT_ENV_UBUNTU = "LIQUID_BIOPSY_DATA_ROOT_UBUNTU"
DATA_ROOT_ENV_MACOS = "LIQUID_BIOPSY_DATA_ROOT_MACOS"
DATA_ROOT_ENV_WINDOWS = "LIQUID_BIOPSY_DATA_ROOT_WINDOWS"

# OS-specific placeholders. Update these once per machine if you do not use env vars.
# TODO(you): set your Ubuntu root prefix, then keep "liquid-agent-data" as the data folder name.
DEFAULT_DATA_ROOT_UBUNTU = Path("/home/YOUR_UBUNTU_ROOT_PREFIX") / "liquid-agent-data"
# macOS default root prefix is /Volumes/US202, so data root resolves to /Volumes/US202/liquid-agent-data.
DEFAULT_DATA_ROOT_MACOS = Path("/Volumes/US202") / "liquid-agent-data"
# Windows default root prefix is F:\, so data root resolves to F:\liquid-agent-data.
DEFAULT_DATA_ROOT_WINDOWS = Path(r"F:\\") / "liquid-agent-data"

# Standard subdirectories under data root.
MODELS_DIR_NAME = "models"

# Current known top-level dataset folders under data root.
KNOWN_DATASET_DIRS = ("GSE243474", "TCGA-BRCA")


def _select_os_default_data_root() -> Path:
    system = platform.system()
    if system == "Windows":
        return DEFAULT_DATA_ROOT_WINDOWS
    if system == "Darwin":
        return DEFAULT_DATA_ROOT_MACOS
    return DEFAULT_DATA_ROOT_UBUNTU


def _select_os_data_root_env() -> str | None:
    system = platform.system()
    if system == "Windows":
        return DATA_ROOT_ENV_WINDOWS
    if system == "Darwin":
        return DATA_ROOT_ENV_MACOS
    if system == "Linux":
        return DATA_ROOT_ENV_UBUNTU
    return None


def get_data_root() -> Path:
    """Return the canonical data root directory for the current OS."""
    raw_global = os.getenv(DATA_ROOT_ENV)
    if raw_global:
        return Path(raw_global).expanduser().resolve()

    os_env_name = _select_os_data_root_env()
    if os_env_name:
        raw_os = os.getenv(os_env_name)
        if raw_os:
            return Path(raw_os).expanduser().resolve()

    return _select_os_default_data_root().expanduser().resolve()


def ensure_within_data_root(path: Path, *, path_kind: str = "path") -> Path:
    """Validate that a path is inside the configured data root."""
    root = get_data_root()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"{path_kind} must be inside data root '{root}', got '{resolved}'. "
            f"Set {DATA_ROOT_ENV} (or OS-specific env vars: "
            f"{DATA_ROOT_ENV_UBUNTU}/{DATA_ROOT_ENV_MACOS}/{DATA_ROOT_ENV_WINDOWS}) if the root is different. "
            f"Current expected dataset dirs under root: {', '.join(KNOWN_DATASET_DIRS)}."
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


def get_models_root(*, must_exist: bool = False) -> Path:
    """Return the canonical models root under the configured data root."""
    return resolve_data_path(MODELS_DIR_NAME, path_kind="models root", must_exist=must_exist)
