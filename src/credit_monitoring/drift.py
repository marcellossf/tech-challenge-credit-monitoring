"""Aggregate drift reports with fixed reference bins and multiplicity correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from credit_monitoring.settings import FEATURES

DEFAULT_CONFIG = {"psi_threshold": 0.2, "alpha": 0.01, "ks_min_effect": 0.1, "bins": 10}


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI with reference quantiles, infinite tails and smoothed zero proportions.

    Equal quantiles are merged. A constant reference gets a narrow center bin so
    shifts beyond its support remain observable. Input validation rejects NaN/inf.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if not len(ref) or not len(cur) or not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("PSI requires nonempty finite numeric samples")
    if bins < 2 or epsilon <= 0:
        raise ValueError("PSI requires bins >= 2 and epsilon > 0")
    internal = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    if np.min(ref) == np.max(ref):
        center = float(ref[0])
        internal = np.array([np.nextafter(center, -np.inf), np.nextafter(center, np.inf)])
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    ref_p = np.histogram(ref, edges)[0] / len(ref)
    cur_p = np.histogram(cur, edges)[0] / len(cur)
    ref_p = np.maximum(ref_p, epsilon)
    cur_p = np.maximum(cur_p, epsilon)
    ref_p /= ref_p.sum()
    cur_p /= cur_p.sum()
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm family-wise correction; return adjusted p-values in original order."""
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite and between zero and one")
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    ranked = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    adjusted[order] = ranked
    return adjusted.tolist()


def report_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str | Path,
    name: str,
    config: dict | None = None,
) -> dict:
    """Save Evidently HTML/JSON and our transparent PSI+KS decision summary.

    Evidently receives only numeric model features: identifiers, individual
    predictions and labels are excluded. Its drift preset exports distributions
    and metric scores, with no row-level examples. Our JSON holds aggregate scores.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    settings = {**DEFAULT_CONFIG, **(config or {})}
    if Path(name).name != name or not name or name in {".", ".."}:
        raise ValueError("Report name must be a plain filename stem")
    results = []
    for feature in FEATURES:
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].to_numpy(dtype=float)
        psi = population_stability_index(ref, cur, bins=int(settings["bins"]))
        statistic, pvalue = ks_2samp(ref, cur, method="auto")
        results.append(
            {
                "feature": feature,
                "psi": psi,
                "ks_statistic": float(statistic),
                "ks_pvalue": float(pvalue),
            }
        )
    adjusted = holm_adjust([item["ks_pvalue"] for item in results])
    for result, pvalue in zip(results, adjusted, strict=True):
        result["ks_pvalue_holm"] = pvalue
        result["psi_alert"] = result["psi"] >= settings["psi_threshold"]
        result["ks_alert"] = (
            pvalue < settings["alpha"] and result["ks_statistic"] >= settings["ks_min_effect"]
        )
        result["drift_detected"] = bool(result["psi_alert"] or result["ks_alert"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / f"{name}_evidently.html"
    json_path = out / f"{name}_evidently.json"
    # Explicit numeric PSI prevents automatic method changes with sample size.
    snapshot = Report(
        [DataDriftPreset(columns=FEATURES, method="psi", threshold=settings["psi_threshold"])]
    ).run(current_data=current[FEATURES], reference_data=reference[FEATURES])
    snapshot.save_html(str(html_path))
    json_path.write_text(snapshot.json(), encoding="utf-8")
    summary = {
        "scenario": name,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "config": settings,
        "features": results,
        "drift_detected": any(item["drift_detected"] for item in results),
        "drifted_features": [item["feature"] for item in results if item["drift_detected"]],
        "evidently_html": str(html_path),
        "evidently_json": str(json_path),
        "interpretation": (
            "Distribution shift does not establish concept drift or an economic cause"
        ),
    }
    public_summary = {
        **summary,
        "evidently_html": html_path.name,
        "evidently_json": json_path.name,
    }
    (out / f"{name}_drift.json").write_text(json.dumps(public_summary, indent=2), encoding="utf-8")
    return summary
