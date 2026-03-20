"""
Structured JSONL logging utilities.

Every script emits JSONL logs with standard keys:
script_name, run_id, config_digest, inputs, outputs, row_counts, etc.
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from nightscape.paths import LOGS_DIR


def generate_run_id() -> str:
    """Generate a unique run ID for this execution."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


def get_versions() -> dict[str, str]:
    """Get versions of key libraries for reproducibility logging."""
    versions = {"python": sys.version.split()[0]}
    for lib_name in ["geopandas", "pandas", "numpy", "pyproj", "shapely"]:
        try:
            mod = __import__(lib_name)
            versions[lib_name] = mod.__version__
        except ImportError:
            pass
    return versions


class JSONLLogger:
    """Structured JSONL logger for pipeline scripts."""

    def __init__(
        self,
        script_name: str,
        run_id: Optional[str] = None,
        log_dir: Optional[Path] = None,
    ):
        self.script_name = script_name
        self.run_id = run_id or generate_run_id()
        self.log_dir = log_dir or LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{script_name}_{self.run_id}.jsonl"
        self._file_handle = open(self.log_file, "a", encoding="utf-8")

        try:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setLevel(logging.INFO)
            self._console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )

            # Use run_id in logger name to avoid handler stacking on reuse
            self._logger = logging.getLogger(f"nightscape.{script_name}.{self.run_id}")
            self._logger.setLevel(logging.DEBUG)
            self._logger.addHandler(self._console_handler)

            self._write_record(
                level="INFO",
                message="Logger initialized",
                extra={
                    "script_name": script_name,
                    "run_id": self.run_id,
                    "log_file": str(self.log_file),
                    "versions": get_versions(),
                },
            )
        except Exception:
            self._file_handle.close()
            raise

    def _write_record(self, level: str, message: str, extra: Optional[dict] = None) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script_name": self.script_name,
            "run_id": self.run_id,
            "level": level,
            "message": message,
        }
        if extra:
            record["extra"] = extra
        self._file_handle.write(json.dumps(record, default=str) + "\n")
        self._file_handle.flush()

    def debug(self, message: str, extra: Optional[dict] = None) -> None:
        self._write_record("DEBUG", message, extra)
        self._logger.debug(message)

    def info(self, message: str, extra: Optional[dict] = None) -> None:
        self._write_record("INFO", message, extra)
        self._logger.info(message)

    def warning(self, message: str, extra: Optional[dict] = None) -> None:
        self._write_record("WARNING", message, extra)
        self._logger.warning(message)

    def error(self, message: str, extra: Optional[dict] = None) -> None:
        self._write_record("ERROR", message, extra)
        self._logger.error(message)

    def log_config(self, config: dict, config_digest: Optional[str] = None) -> None:
        self._write_record("INFO", "Configuration loaded",
                           extra={"config": config, "config_digest": config_digest})

    def log_inputs(self, inputs: dict) -> None:
        self._write_record("INFO", "Inputs registered", extra={"inputs": inputs})

    def log_outputs(self, outputs: dict) -> None:
        self._write_record("INFO", "Outputs registered", extra={"outputs": outputs})

    def log_metrics(self, metrics: dict) -> None:
        self._write_record("INFO", "Metrics recorded", extra={"metrics": metrics})

    def close(self) -> None:
        self._write_record("INFO", "Logger closing", extra={"run_id": self.run_id})
        self._file_handle.close()
        self._logger.removeHandler(self._console_handler)

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            import traceback
            tb_text = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self.error(f"Exception occurred: {exc_type.__name__}: {exc_val}",
                       extra={"traceback": tb_text})
        self.close()


def get_logger(script_name: str, run_id: Optional[str] = None) -> JSONLLogger:
    """Convenience function to get a configured logger."""
    return JSONLLogger(script_name=script_name, run_id=run_id)
