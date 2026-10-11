"""Verify aggregate evidence survives failures and HTML stays self-contained."""

import json
import time

import pytest

from credit_monitoring.dashboard import render_dashboard
from credit_monitoring.monitoring import JsonLogger, ResourceSampler, track_run


def test_resource_sampling_stops_even_when_stage_fails():
    sampler = ResourceSampler(interval=0.01)
    with pytest.raises(RuntimeError), sampler:
        time.sleep(0.025)
        raise RuntimeError("failed stage")
    assert sampler.summary["samples"] >= 2
    assert sampler.summary["peak_rss_mb"] > 0
    assert sampler.summary["disk_free_gb"] > 0
    assert sampler._thread is not None and not sampler._thread.is_alive()


def test_json_logger_appends_parseable_aggregate_events(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    logger = JsonLogger(path)
    logger.emit(scenario="invalid", status="BLOCKED", rows=6000, rejected=4)
    logger.emit(scenario="control", status="SUCCESS", alerts=[])
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert events[0]["status"] == "BLOCKED"
    assert events[0]["timestamp"].endswith("+00:00")


def test_dashboard_escapes_data_and_links_local_evidence(tmp_path):
    artifact = tmp_path / "drift.html"
    artifact.write_text("report", encoding="utf-8")
    output = tmp_path / "dashboard.html"
    render_dashboard(
        {
            "baseline": {"recall": 0.6},
            "runs": [
                {
                    "scenario": "<script>alert(1)</script>",
                    "status": "BLOCKED",
                    "rows": 6000,
                    "duration_seconds": 1.2,
                    "alerts": ["Falha <input>"],
                    "metrics": {},
                    "resources": {"peak_rss_mb": 120},
                    "drift": {},
                    "artifacts": {"drift": str(artifact), "bad": "javascript:alert(1)"},
                }
            ],
            "provenance": {"source": "UCI"},
        },
        output,
    )
    html = output.read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "javascript:" not in html
    assert "href='drift.html'" in html
    assert "BLOCKED" in html and "6000" in html
    assert "Indisponíveis" in html
    assert "<script src" not in html


def test_mlflow_records_blocked_scenario_with_real_local_store(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    artifact = tmp_path / "quality.json"
    artifact.write_text('{"invalid_count": 4}', encoding="utf-8")
    run_id = track_run(
        uri,
        "test-monitoring",
        "invalid",
        {"status": "BLOCKED"},
        {"rows": 6000, "resources": {"peak_rss_mb": 12.5}},
        {"quality": artifact},
        status="FAILED",
    )
    client = mlflow.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.info.status == "FAILED"
    assert run.data.params["status"] == "BLOCKED"
    assert run.data.metrics["resources.peak_rss_mb"] == 12.5
    assert client.list_artifacts(run_id)[0].path == "quality.json"
