"""Strict batch contracts and aggregate-only quality diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

FEATURES = ["AGE", "LIMIT_BAL", "PAY_0", "BILL_AMT1", "PAY_AMT1"]
_COLUMNS = {"row_id", *FEATURES, "target"}
_RULES = {
    "integer_dtype",
    "numeric_dtype",
    "finite",
    "age_range",
    "credit_limit_range",
    "payment_status_range",
    "nonnegative",
    "positive",
    "binary_target",
    "not_nullable",
    "field_uniqueness",
    "column_in_dataframe",
    "column_in_schema",
    "dtype",
    "schema",
}


class DataQualityError(ValueError):
    """A rejected batch; ``summary`` contains no row values or indices."""

    def __init__(self, summary: list[dict[str, str | int]]) -> None:
        self.summary = summary
        super().__init__("Batch rejected by data contract; see aggregate quality summary.")


def _numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
        series.dtype
    )


def _integer(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
        series.dtype
    )


def _finite(series: pd.Series) -> bool:
    return _numeric(series) and bool(np.isfinite(series.to_numpy(dtype=float)).all())


def _range(series: pd.Series, lower: float, upper: float) -> bool | pd.Series:
    if not _numeric(series):
        return False
    return series.between(lower, upper)


def _nonnegative(series: pd.Series) -> bool | pd.Series:
    return series.ge(0) if _numeric(series) else False


def _positive(series: pd.Series) -> bool | pd.Series:
    return series.gt(0) if _numeric(series) else False


def _column(*checks: pa.Check, integer: bool = False, unique: bool = False) -> pa.Column:
    return pa.Column(
        checks=[
            pa.Check(
                _integer if integer else _numeric,
                name="integer_dtype" if integer else "numeric_dtype",
            ),
            pa.Check(_finite, name="finite"),
            *checks,
        ],
        nullable=False,
        unique=unique,
        coerce=False,
    )


def batch_schema(require_target: bool = True) -> pa.DataFrameSchema:
    """Build the evaluation or inference contract; no silent type conversion."""
    return pa.DataFrameSchema(
        {
            "row_id": _column(pa.Check(_positive, name="positive"), integer=True, unique=True),
            "AGE": _column(pa.Check(lambda s: _range(s, 18, 100), name="age_range"), integer=True),
            "LIMIT_BAL": _column(
                pa.Check(
                    lambda s: _range(s, np.nextafter(0.0, 1.0), 2_000_000),
                    name="credit_limit_range",
                )
            ),
            "PAY_0": _column(
                pa.Check(lambda s: _range(s, -2, 9), name="payment_status_range"),
                integer=True,
            ),
            "BILL_AMT1": _column(),
            "PAY_AMT1": _column(pa.Check(_nonnegative, name="nonnegative")),
            "target": pa.Column(
                checks=[
                    pa.Check(_integer, name="integer_dtype"),
                    pa.Check(lambda s: _range(s, 0, 1), name="binary_target"),
                ],
                nullable=False,
                required=require_target,
                coerce=False,
            ),
        },
        strict=True,
        coerce=False,
        unique_column_names=True,
    )


def _sanitized_summary(failures: pd.DataFrame) -> list[dict[str, str | int]]:
    counts: dict[tuple[str, str], int] = {}
    for failure in failures.to_dict(orient="records"):
        column = failure.get("column")
        rule = failure.get("check")
        # Never pass through arbitrary column names, check text, values, or row labels.
        column = column if isinstance(column, str) and column in _COLUMNS else "__schema__"
        rule = rule if isinstance(rule, str) and rule in _RULES else "schema"
        key = (column, rule)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"column": column, "rule": rule, "count": count}
        for (column, rule), count in sorted(counts.items())
    ]


def validate_batch(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Validate all rows before inference; a single violation rejects the whole batch.

    Counts are Pandera failure occurrences per column/rule, not distinct rejected rows.
    With ``require_target=False``, a supplied target is still checked.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataQualityError([{"column": "__schema__", "rule": "schema", "count": 1}])
    if df.empty:
        raise DataQualityError([{"column": "__schema__", "rule": "schema", "count": 1}])
    try:
        return batch_schema(require_target).validate(df, lazy=True)
    except pa.errors.SchemaErrors as error:
        raise DataQualityError(_sanitized_summary(error.failure_cases)) from None
