"""
Canonical path resolution for the NYC Nightscape project.

All scripts MUST import paths from here — no relative '../' paths allowed.
Detects root via .project-root (primary) and fallback markers.
"""

from pathlib import Path
from typing import Optional

ROOT_MARKERS = [".project-root", "pyproject.toml", ".git"]


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Find the project root by searching upward for marker files."""
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current = start_path
    while current != current.parent:
        for marker in ROOT_MARKERS:
            if (current / marker).exists():
                return current
        current = current.parent

    for marker in ROOT_MARKERS:
        if (current / marker).exists():
            return current

    raise FileNotFoundError(
        f"Could not find project root. Searched for markers {ROOT_MARKERS} "
        f"starting from {start_path}"
    )


# Canonical paths (resolved at import time)
PROJECT_ROOT = find_project_root()

# Config
CONFIG_DIR = PROJECT_ROOT / "configs"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"

# Processed subdirectories
REPORTS_DIR = PROCESSED_DIR / "reports"
GEO_DIR = PROCESSED_DIR / "geo"
EQUITY_DIR = PROCESSED_DIR / "equity"
VALIDATION_DIR = PROCESSED_DIR / "validation"
METADATA_DIR = PROCESSED_DIR / "metadata"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
INTERACTIVE_DIR = OUTPUTS_DIR / "interactive"

# Source and scripts
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Tests
TESTS_DIR = PROJECT_ROOT / "tests"

# Docs
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_dirs_exist() -> None:
    """Create all canonical directories if they don't exist."""
    dirs = [
        CONFIG_DIR,
        RAW_DIR, INTERMEDIATE_DIR,
        REPORTS_DIR, GEO_DIR, EQUITY_DIR, VALIDATION_DIR, METADATA_DIR,
        LOGS_DIR,
        FIGURES_DIR, TABLES_DIR, INTERACTIVE_DIR,
        DOCS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RAW_DIR:      {RAW_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"CONFIG_DIR:   {CONFIG_DIR}")
    print(f"LOGS_DIR:     {LOGS_DIR}")
