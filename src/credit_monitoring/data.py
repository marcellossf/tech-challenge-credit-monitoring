"""Download the original UCI workbook, profile and create disjoint splits."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_monitoring.settings import FEATURES, SEED, SOURCE_PAGE, SOURCE_SHA256, SOURCE_URL


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def download(data_dir: Path) -> Path:
    """Cache original bytes locally; use HTTPS and a finite network timeout."""
    raw = data_dir / "raw" / "uci_credit.zip"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        request = urllib.request.Request(
            SOURCE_URL, headers={"User-Agent": "credit-monitoring/0.1"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read()
        # Validate before replacing the cache; interrupted downloads cannot become a valid cache.
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive.testzip() is not None:
                raise ValueError("UCI archive checksum validation failed")
        temporary = raw.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(raw)
    with zipfile.ZipFile(raw) as archive:
        workbooks = [name for name in archive.namelist() if name.lower().endswith(".xls")]
        if len(workbooks) != 1:
            raise ValueError("Expected exactly one original UCI workbook")
    if sha256(raw) != SOURCE_SHA256:
        raise ValueError("Source hash changed: inspect provenance before accepting a new dataset")
    return raw


def load_source(raw: Path) -> pd.DataFrame:
    with zipfile.ZipFile(raw) as archive:
        name = next(name for name in archive.namelist() if name.lower().endswith(".xls"))
        df = pd.read_excel(io.BytesIO(archive.read(name)), header=1, engine="xlrd")
    required = ["ID", *FEATURES, "default payment next month"]
    if any(column not in df for column in required):
        raise ValueError("Unexpected workbook schema; review source before mapping")
    # The source ID is a local split/deduplication key only. It never enters model features/logs.
    selected = df[required].rename(columns={"ID": "row_id", "default payment next month": "target"})
    return selected


def split_data(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    if len(df) < 5000 or df["row_id"].duplicated().any():
        raise ValueError("Need >=5000 real rows with unique keys")
    if set(df["target"].unique()) != {0, 1}:
        raise ValueError("Expected a binary financial target")
    reference, remaining = train_test_split(
        df, test_size=0.4, stratify=df["target"], random_state=seed
    )
    validation, production = train_test_split(
        remaining, test_size=0.5, stratify=remaining["target"], random_state=seed
    )
    return {
        name: frame.sort_values("row_id").reset_index(drop=True)
        for name, frame in {
            "reference": reference,
            "validation": validation,
            "production": production,
        }.items()
    }


def prepare(data_dir: Path, report_dir: Path, seed: int = SEED) -> dict:
    from credit_monitoring.contracts import validate_batch

    raw = download(data_dir)
    df = load_source(raw)
    validate_batch(df)
    splits = split_data(df, seed)
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(processed / f"{name}.csv", index=False)
    provenance = {
        "source": SOURCE_PAGE,
        "download_url": SOURCE_URL,
        "author": "I-Cheng Yeh",
        "doi": "10.24432/C55S3H",
        "license": "CC BY 4.0",
        "raw_sha256": sha256(raw),
        "seed": seed,
        "rows": len(df),
        "rows_rejected": 0,
        "features": FEATURES,
        "split_method": "stratified random 60/20/20; simulated production, not temporal holdout",
        "splits": {
            name: {
                "rows": len(frame),
                "default_rate": float(frame.target.mean()),
                "sha256": sha256(processed / f"{name}.csv"),
            }
            for name, frame in splits.items()
        },
        "profile": {
            col: {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
            for col in FEATURES
        },
        "excluded_predictors": ["row_id", "SEX", "EDUCATION", "MARRIAGE", "target"],
    }
    write_json(report_dir / "provenance.json", provenance)
    return provenance


def read_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(data_dir / "processed" / f"{name}.csv")
        for name in ("reference", "validation", "production")
    }
