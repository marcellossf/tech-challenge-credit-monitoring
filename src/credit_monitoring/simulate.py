"""Reproducible synthetic scenarios; the reserved production sample stays immutable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_monitoring.settings import FEATURES


def make_scenarios(
    production: pd.DataFrame, seed: int = 42
) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Generate independent batches and manifests without modifying the input.

    These are interventions on a held-out sample, not observed economic periods.
    Concept drift changes P(y|X) in one declared segment while preserving every X.
    """
    if len(production) < 3:
        raise ValueError("At least three production rows are required for scenarios")
    missing = set(FEATURES + ["row_id", "target"]) - set(production.columns)
    if missing:
        raise ValueError(f"Missing scenario columns: {sorted(missing)}")
    scenarios: dict[str, tuple[pd.DataFrame, dict]] = {}

    def manifest(kind: str, **details: object) -> dict:
        return {"kind": kind, "seed": seed, "synthetic": kind != "control", **details}

    scenarios["control"] = (
        production.copy(deep=True),
        manifest("control", transformation="none", labels_synthetic=False),
    )
    invalid = production.copy(deep=True)
    invalid.loc[invalid.index[0], "LIMIT_BAL"] = np.nan
    invalid.loc[invalid.index[1], "AGE"] = 150
    invalid = pd.concat([invalid, invalid.iloc[[2]]], ignore_index=True)
    scenarios["invalid"] = (
        invalid,
        manifest(
            "invalid",
            expected_rejection=True,
            violations=["missing LIMIT_BAL", "AGE=150", "duplicate row_id"],
            labels_synthetic=False,
        ),
    )
    for step, (limit_factor, payment_factor) in enumerate(
        [(1.3, 0.7), (1.6, 0.4), (2.0, 0.15)], start=1
    ):
        current = production.copy(deep=True)
        # Round to whole NT$ and clip to the explicit contract, not training maxima.
        current["LIMIT_BAL"] = current["LIMIT_BAL"].mul(limit_factor).round().clip(upper=2_000_000)
        current["PAY_AMT1"] = current["PAY_AMT1"].mul(payment_factor).round()
        scenarios[f"data_drift_t{step}"] = (
            current,
            manifest(
                "data_drift",
                simulated_window=step,
                transformations={
                    "LIMIT_BAL": {"factor": limit_factor, "cap": 2_000_000},
                    "PAY_AMT1": {"factor": payment_factor},
                },
                labels_synthetic=False,
                causal_claim="Known synthetic intervention; no real economic cause inferred",
            ),
        )
    concept = production.copy(deep=True)
    segment = concept["PAY_0"].le(0)
    rng = np.random.default_rng(seed)
    before = concept["target"].copy()
    concept.loc[segment, "target"] = rng.binomial(1, 0.65, int(segment.sum()))
    scenarios["concept_drift"] = (
        concept,
        manifest(
            "concept_drift",
            segment="PAY_0 <= 0",
            mechanism="Replace labels within segment with Bernoulli(default_probability)",
            default_probability=0.65,
            segment_rows=int(segment.sum()),
            labels_changed=int(before.ne(concept["target"]).sum()),
            labels_synthetic=True,
            features_preserved=True,
            interpretation="Synthetic change in P(y|X); marginal feature drift is not expected",
        ),
    )
    return scenarios
