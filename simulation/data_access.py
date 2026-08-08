"""
Read Phase 1's bronze assumption tables for the simulation.

Reuses the Databricks auth and SQL plumbing already in `databricks/_common.py`
rather than duplicating a client. Values come from the live tables -- nothing
here hardcodes numbers copied out of an earlier report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "databricks"))

from _common import (  # noqa: E402  -- import after sys.path bootstrap
    TBL_ASSUMPTIONS,
    TBL_CORRELATION,
    TBL_CVR_CORRELATION,
    get_client,
    resolve_warehouse_id,
    sql,
)

CORRELATION_TABLES = {
    "cvr": TBL_CVR_CORRELATION,
    "revenue": TBL_CORRELATION,
}

ASSUMPTION_COLUMNS = [
    "channel_id", "channel_name",
    "mean_cpc", "std_cpc",
    "mean_cvr", "std_cvr",
    "mean_revenue_per_conversion", "std_revenue_per_conversion",
    "distribution_type",
]


def open_session():
    w = get_client()
    return w, resolve_warehouse_id(w)


def load_assumptions(w, warehouse_id) -> pd.DataFrame:
    """One row per channel, ordered by channel_id for a stable index."""
    cols, rows = sql(
        w, warehouse_id,
        f"SELECT {', '.join(ASSUMPTION_COLUMNS)} FROM {TBL_ASSUMPTIONS} ORDER BY channel_id",
    )
    df = pd.DataFrame(rows, columns=cols)
    numeric = [c for c in ASSUMPTION_COLUMNS if c.startswith(("mean_", "std_"))]
    df[numeric] = df[numeric].astype(float)
    if df.empty:
        raise RuntimeError(f"{TBL_ASSUMPTIONS} returned no rows")
    if df["channel_id"].duplicated().any():
        raise RuntimeError(f"{TBL_ASSUMPTIONS} has duplicate channel_id rows")
    return df


def load_correlation_matrix(w, warehouse_id, channels: list[str],
                            source: str = "cvr") -> np.ndarray:
    """Dense (k, k) matrix ordered to match `channels` exactly.

    `source` selects which bronze matrix to read -- "cvr" (what the simulation
    should use, since CVR is the only correlated quantity) or "revenue" (the
    original proxy, kept for comparison).

    The long-form bronze table has no inherent ordering, so this pivots and
    then reindexes against the assumption table's channel order. Getting this
    wrong would silently pair the wrong channels' correlations, so it raises
    on any missing cell rather than filling a default.
    """
    try:
        table = CORRELATION_TABLES[source]
    except KeyError:
        raise ValueError(
            f"unknown correlation source {source!r}; expected one of {sorted(CORRELATION_TABLES)}"
        ) from None

    _, rows = sql(
        w, warehouse_id,
        f"SELECT channel_id_a, channel_id_b, correlation_coefficient FROM {table}",
    )
    lookup = {(a, b): float(v) for a, b, v in rows}

    k = len(channels)
    matrix = np.empty((k, k), dtype=float)
    for i, a in enumerate(channels):
        for j, b in enumerate(channels):
            try:
                matrix[i, j] = lookup[(a, b)]
            except KeyError as exc:
                raise RuntimeError(
                    f"{table} is missing the ({a}, {b}) pair -- "
                    f"cannot build a complete {k}x{k} matrix"
                ) from exc
    return matrix
