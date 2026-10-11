"""Train once; choose threshold on validation, never on simulated production."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_monitoring.data import sha256, write_json
from credit_monitoring.settings import FEATURES, SEED


def evaluate(target, probability, threshold: float) -> dict:
    prediction = np.asarray(probability) >= threshold
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "pr_auc": float(average_precision_score(target, probability)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "brier": float(brier_score_loss(target, probability)),
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
    }


def train(
    reference: pd.DataFrame,
    validation: pd.DataFrame,
    artifact_dir: Path,
    report_dir: Path,
    seed: int = SEED,
) -> dict:
    from credit_monitoring.contracts import validate_batch

    validate_batch(reference)
    validate_batch(validation)
    if set(reference.row_id) & set(validation.row_id):
        raise ValueError("Training/validation leakage: overlapping keys")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    model.fit(reference[FEATURES], reference.target)
    probabilities = model.predict_proba(validation[FEATURES])[:, 1]
    precision, recall, thresholds = precision_recall_curve(validation.target, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[np.argmax(scores)])
    dummy = DummyClassifier(strategy="prior").fit(reference[FEATURES], reference.target)
    dummy_probability = dummy.predict_proba(validation[FEATURES])[:, 1]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "baseline.joblib"
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES, "seed": seed}, path)
    summary = {
        "model": "StandardScaler + LogisticRegression",
        "seed": seed,
        "features": FEATURES,
        "threshold": threshold,
        "threshold_policy": "maximum validation F1; academic choice, not credit approval policy",
        "train_rows": len(reference),
        "validation_rows": len(validation),
        "metrics": evaluate(validation.target, probabilities, threshold),
        "dummy_metrics": evaluate(validation.target, dummy_probability, threshold),
        "coefficients_standardized": dict(
            zip(FEATURES, model.named_steps["classifier"].coef_[0].tolist(), strict=True)
        ),
        "model_sha256": sha256(path),
    }
    write_json(report_dir / "baseline.json", summary)
    return summary


def predict_validated(bundle: dict, batch: pd.DataFrame) -> np.ndarray:
    """Single inference boundary: contracts must pass before invoking the model."""
    from credit_monitoring.contracts import validate_batch

    validate_batch(batch, require_target=False)
    return bundle["model"].predict_proba(batch[bundle["features"]])[:, 1]
