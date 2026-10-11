"""Orchestrate the academic demo and keep expected blocks distinct from failures."""

import hashlib
import importlib.metadata
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from credit_monitoring.contracts import DataQualityError, validate_batch
from credit_monitoring.dashboard import render_dashboard
from credit_monitoring.data import prepare, read_splits, sha256, write_json
from credit_monitoring.drift import report_drift
from credit_monitoring.monitoring import JsonLogger, ResourceSampler, track_run
from credit_monitoring.settings import FEATURES, SEED
from credit_monitoring.simulate import make_scenarios
from credit_monitoring.train import evaluate, predict_validated, train


def load_config(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Config must be a YAML mapping")
    return settings


def age_metrics(batch: pd.DataFrame, probability, threshold: float) -> dict:
    """Aggregate age-group errors; no subgroup claim when sample/class support is low."""
    groups = {}
    for label, mask in {
        "18-29": batch.AGE.between(18, 29),
        "30-49": batch.AGE.between(30, 49),
        "50-100": batch.AGE.between(50, 100),
    }.items():
        part = batch.loc[mask]
        groups[label] = {"rows": len(part)}
        if (
            len(part) >= 200
            and part.target.value_counts().min() >= 30
            and part.target.nunique() == 2
        ):
            groups[label]["metrics"] = evaluate(
                part.target, probability[mask.to_numpy()], threshold
            )
        else:
            groups[label]["reason"] = "Insufficient support: require n>=200 and >=30 per class"
    return groups


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_demo(
    data_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
    config_path: Path,
    tracking_uri: str | None = None,
) -> dict:
    settings = load_config(config_path)
    seed = int(settings.get("seed", SEED))
    report_dir.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex
    logger = JsonLogger(report_dir / "pipeline.jsonl")
    logger.emit(batch_id=batch_id, stage="demo", status="STARTED", seed=seed)
    started = time.perf_counter()
    try:
        logger.emit(batch_id=batch_id, stage="prepare", status="STARTED")
        provenance = prepare(data_dir, report_dir, seed)
        splits = read_splits(data_dir)
        logger.emit(batch_id=batch_id, stage="prepare", status="FINISHED", rows=provenance["rows"])
        baseline = train(splits["reference"], splits["validation"], artifact_dir, report_dir, seed)
        logger.emit(
            batch_id=batch_id,
            stage="train",
            status="FINISHED",
            rows=baseline["train_rows"],
            metrics=baseline["metrics"],
        )
        bundle = joblib.load(artifact_dir / "baseline.joblib")
        drift_settings = settings.get("drift", {})
        drift_config = {
            "psi_threshold": float(drift_settings.get("psi_threshold", 0.2)),
            "alpha": float(drift_settings.get("ks_alpha", 0.01)),
            "ks_min_effect": float(drift_settings.get("ks_effect_threshold", 0.1)),
        }
        baseline_run = track_run(
            tracking_uri,
            "credit-monitoring",
            f"{batch_id}:baseline",
            {
                "seed": seed,
                "threshold": baseline["threshold"],
                "raw_sha256": provenance["raw_sha256"],
            },
            baseline["metrics"],
            [report_dir / "baseline.json", report_dir / "provenance.json"],
        )
        runs = []
        control_metrics = None
        for name, (batch, manifest) in make_scenarios(splits["production"], seed).items():
            run_started = time.perf_counter()
            run = {
                "scenario": name,
                "rows": len(batch),
                "alerts": [],
                "metrics": {},
                "drift": {},
                "artifacts": {},
                "batch_id": batch_id,
            }
            logger.emit(
                batch_id=batch_id,
                stage="validation",
                scenario=name,
                status="STARTED",
                rows=len(batch),
            )
            manifest_path = report_dir / f"{name}_manifest.json"
            write_json(manifest_path, manifest)
            run["artifacts"]["manifest"] = manifest_path.as_posix()
            with ResourceSampler(disk_path=report_dir) as sampler:
                try:
                    validate_batch(batch)
                    logger.emit(
                        batch_id=batch_id,
                        stage="validation",
                        scenario=name,
                        status="FINISHED",
                        rows=len(batch),
                    )
                    probability = predict_validated(bundle, batch)
                    run["metrics"] = evaluate(batch.target, probability, bundle["threshold"])
                    run["subgroups_age"] = age_metrics(batch, probability, bundle["threshold"])
                    logger.emit(
                        batch_id=batch_id,
                        stage="inference",
                        scenario=name,
                        status="FINISHED",
                        rows=len(batch),
                        metrics=run["metrics"],
                    )
                    drift = report_drift(splits["reference"], batch, report_dir, name, drift_config)
                    run["drift"] = drift
                    for label in ("evidently_html", "evidently_json"):
                        run["artifacts"][label] = Path(drift[label]).as_posix()
                    run["artifacts"]["statistical_summary"] = (
                        report_dir / f"{name}_drift.json"
                    ).as_posix()
                    if drift["drift_detected"]:
                        run["alerts"].append("DATA_DRIFT: " + ", ".join(drift["drifted_features"]))
                    if name == "control":
                        control_metrics = run["metrics"]
                    elif control_metrics is not None:
                        policy = settings["performance"]
                        deltas = {
                            "recall_drop": control_metrics["recall"] - run["metrics"]["recall"],
                            "pr_auc_drop": control_metrics["pr_auc"] - run["metrics"]["pr_auc"],
                            "brier_increase": run["metrics"]["brier"] - control_metrics["brier"],
                        }
                        run["performance_changes_vs_control"] = deltas
                        if any(deltas[key] >= policy[f"{key}_threshold"] for key in deltas):
                            run["alerts"].append("PERFORMANCE_DEGRADATION: investigate labels")
                    run["status"] = "ALERT" if run["alerts"] else "HEALTHY"
                    run["inference_executed"] = True
                    run["label_interpretation"] = (
                        "Synthetic labels: controlled P(y|X) intervention"
                        if manifest.get("labels_synthetic")
                        else "Original labels; shifted-X metrics are illustrative counterfactuals"
                    )
                    if name == "concept_drift":
                        run["features_preserved"] = batch[FEATURES].equals(
                            splits["production"][FEATURES]
                        )
                except DataQualityError as error:
                    if not manifest.get("expected_rejection"):
                        raise
                    run["status"] = "BLOCKED"
                    run["inference_executed"] = False
                    run["rejected_rows"] = len(batch)
                    run["quality_errors"] = error.summary
                    run["alerts"].append("DATA_QUALITY: whole batch blocked before inference")
                    error_path = report_dir / "quality_errors.json"
                    write_json(
                        error_path,
                        {"scenario": name, "rejected_rows": len(batch), "failures": error.summary},
                    )
                    run["artifacts"]["quality_errors"] = error_path.as_posix()
                    logger.emit(
                        batch_id=batch_id,
                        stage="validation",
                        scenario=name,
                        status="BLOCKED",
                        rows=len(batch),
                        quality_errors=error.summary,
                    )
            run["resources"] = sampler.summary
            run["duration_seconds"] = round(time.perf_counter() - run_started, 4)
            run_path = report_dir / f"{name}_run.json"
            # Store portable relative links; no user workspace paths in published reports.
            run["artifacts"] = {
                key: Path(value).relative_to(report_dir).as_posix()
                for key, value in run["artifacts"].items()
            }
            if run["drift"]:
                for key in ("evidently_html", "evidently_json"):
                    run["drift"][key] = Path(run["drift"][key]).name
            write_json(run_path, run)
            run["mlflow_run_id"] = track_run(
                tracking_uri,
                "credit-monitoring",
                f"{batch_id}:{name}",
                {
                    "batch_id": batch_id,
                    "scenario": name,
                    "status": run["status"],
                    "seed": seed,
                    "model_sha256": baseline["model_sha256"],
                    "raw_sha256": provenance["raw_sha256"],
                    "config_sha256": sha256(config_path),
                },
                {
                    **run["metrics"],
                    **run["resources"],
                    "rows": len(batch),
                    "duration_seconds": run["duration_seconds"],
                    "alerts": len(run["alerts"]),
                    "rejected_rows": run.get("rejected_rows", 0),
                },
                [run_path, *[report_dir / path for path in run["artifacts"].values()]],
                status="FAILED" if run["status"] == "BLOCKED" else "FINISHED",
            )
            write_json(run_path, run)
            from mlflow import MlflowClient

            uri = tracking_uri or (
                "sqlite:///" + (Path("artifacts/mlflow/tracking.db").resolve()).as_posix()
            )
            MlflowClient(tracking_uri=uri).log_artifact(run["mlflow_run_id"], str(run_path))
            logger.emit(
                batch_id=batch_id,
                stage="scenario",
                scenario=name,
                status=run["status"],
                rows=len(batch),
                duration_seconds=run["duration_seconds"],
                alerts=run["alerts"],
                resources=run["resources"],
                mlflow_run_id=run["mlflow_run_id"],
            )
            runs.append(run)
        summary = {
            "batch_id": batch_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": _commit(),
            "source_sha256": source_fingerprint(),
            "configuration": settings,
            "configuration_sha256": sha256(config_path),
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("pandas", "scikit-learn", "pandera", "evidently", "mlflow")
            },
            "baseline": baseline,
            "baseline_mlflow_run_id": baseline_run,
            "provenance": provenance,
            "runs": runs,
            "duration_seconds": round(time.perf_counter() - started, 4),
        }
        write_json(report_dir / "summary.json", summary)
        render_dashboard(summary, report_dir / "index.html")
        logger.emit(
            batch_id=batch_id,
            stage="demo",
            status="FINISHED",
            duration_seconds=summary["duration_seconds"],
            scenarios=len(runs),
        )
        return summary
    except Exception as error:
        logger.emit(
            batch_id=batch_id, stage="demo", status="FAILED", error_type=type(error).__name__
        )
        raise


def verify(report_dir: Path) -> dict:
    """Verify delivered evidence, not just the presence of files."""
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    runs = {run["scenario"]: run for run in summary["runs"]}
    required = {
        "control",
        "invalid",
        "data_drift_t1",
        "data_drift_t2",
        "data_drift_t3",
        "concept_drift",
    }
    checks = {
        "dataset_ge_5000": summary["provenance"]["rows"] >= 5000,
        "all_scenarios": set(runs) == required,
        "invalid_blocked_before_inference": runs["invalid"]["status"] == "BLOCKED"
        and not runs["invalid"]["inference_executed"],
        "control_no_alerts": not runs["control"]["alerts"],
        "two_features_detected": {"LIMIT_BAL", "PAY_AMT1"}.issubset(
            runs["data_drift_t3"]["drift"]["drifted_features"]
        ),
        "concept_features_preserved": runs["concept_drift"]["features_preserved"],
        "concept_without_input_drift": not runs["concept_drift"]["drift"]["drift_detected"],
        "concept_performance_alert": any(
            alert.startswith("PERFORMANCE") for alert in runs["concept_drift"]["alerts"]
        ),
        "resources_measured": all(run["resources"]["samples"] >= 2 for run in runs.values()),
        "mlflow_tracked": all(run["mlflow_run_id"] for run in runs.values()),
        "artifacts_present": all(
            (report_dir / path).is_file()
            for run in runs.values()
            for path in run["artifacts"].values()
        ),
        "dashboard_present": (report_dir / "index.html").is_file(),
    }
    write_json(report_dir / "verification.json", checks)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Evidence verification failed: " + ", ".join(failed))
    return checks
