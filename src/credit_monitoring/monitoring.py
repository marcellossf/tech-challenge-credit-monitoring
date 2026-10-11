"""Aggregate event logs, measured process resources and local MLflow tracking."""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


class ResourceSampler:
    """Sample this process while a stage runs; CPU is percent across all cores.

    RSS is sampled rather than an OS high-water mark. Very short spikes between
    samples can be missed. CPU uses psutil's process convention (may exceed 100%).
    """

    def __init__(self, interval: float = 0.1, disk_path: Path | str = "."):
        if interval <= 0:
            raise ValueError("Sampling interval must be positive")
        self.interval = interval
        self.disk_path = Path(disk_path).resolve()
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss: list[float] = []
        self._cpu: list[float] = []
        self._disk_free: list[float] = []

    def _sample(self, include_cpu: bool = True) -> None:
        self._rss.append(self._process.memory_info().rss / 1024**2)
        if include_cpu:
            self._cpu.append(self._process.cpu_percent(interval=None))
        self._disk_free.append(psutil.disk_usage(str(self.disk_path)).free / 1024**3)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> ResourceSampler:
        self._rss.clear()
        self._cpu.clear()
        self._disk_free.clear()
        self._stop.clear()
        self._process.cpu_percent(interval=None)
        self._sample(include_cpu=False)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()

    @property
    def summary(self) -> dict[str, float | int]:
        return {
            "peak_rss_mb": round(max(self._rss, default=0.0), 3),
            "cpu_percent_mean": round(sum(self._cpu) / len(self._cpu), 3) if self._cpu else 0.0,
            "disk_free_gb": round(min(self._disk_free, default=0.0), 3),
            "samples": len(self._rss),
            "sampling_interval_seconds": self.interval,
        }


class JsonLogger:
    """Append aggregate pipeline events. Callers must never include raw rows."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, **event: Any) -> None:
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")


def _numeric_metrics(values: dict, prefix: str = "") -> dict[str, float]:
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(_numeric_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value):
                result[name] = float(value)
    return result


def track_run(
    tracking_uri: str | None,
    experiment: str,
    name: str,
    params: dict,
    metrics: dict,
    artifacts: dict | list | tuple,
    status: str = "FINISHED",
) -> str:
    """Persist one scenario, including expected rejected batches, in MLflow.

    The default SQLite store and artifacts stay local under artifacts/mlflow.
    Uses MlflowClient so active runs and global tracking settings are untouched.
    """
    from mlflow import MlflowClient

    if status not in {"FINISHED", "FAILED", "KILLED"}:
        raise ValueError("MLflow terminal status must be FINISHED, FAILED or KILLED")
    storage = Path("artifacts/mlflow").resolve()
    storage.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri or f"sqlite:///{(storage / 'tracking.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri)
    existing = client.get_experiment_by_name(experiment)
    experiment_id = (
        existing.experiment_id
        if existing is not None
        else client.create_experiment(experiment, artifact_location=storage.as_uri())
    )
    run = client.create_run(experiment_id, tags={"mlflow.runName": name})
    run_id = run.info.run_id
    try:
        for key, value in params.items():
            serialized = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else str(value)
            )
            client.log_param(run_id, str(key), serialized)
        for key, value in _numeric_metrics(metrics).items():
            client.log_metric(run_id, key, value)
        paths = artifacts.values() if isinstance(artifacts, dict) else artifacts
        for path in paths:
            artifact = Path(path)
            if artifact.is_dir():
                client.log_artifacts(run_id, str(artifact))
            elif artifact.is_file():
                client.log_artifact(run_id, str(artifact))
            else:
                raise FileNotFoundError(artifact)
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id, status=status)
    return run_id
