"""Read exact persisted candidate/result artifacts from a matching Phase 5 run."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def _read_volume_parquet(w, path: str) -> pd.DataFrame:
    response = w.files.download(path)
    if response.contents is None:
        raise RuntimeError(f"Databricks returned no content for {path}")
    return pd.read_parquet(io.BytesIO(response.contents.read()))


def load_persisted_candidates(w, sweep: pd.DataFrame, *, catalog: str,
                              phase5_job_id: int) -> tuple[pd.DataFrame, int]:
    """Return candidates from the newest successful run whose results match gold."""
    runs = []
    for run in w.jobs.list_runs(job_id=phase5_job_id, completed_only=True):
        result = getattr(getattr(run, "state", None), "result_state", None)
        if str(getattr(result, "value", result)).upper() == "SUCCESS":
            runs.append(run)
    runs.sort(key=lambda r: int(getattr(r, "start_time", 0) or 0), reverse=True)
    if not runs:
        raise RuntimeError(f"no successful runs found for Phase 5 job {phase5_job_id}")

    metric_cols = ["mean_revenue", "std_revenue", "var_95", "cvar_95", "expected_roas"]
    gold = sweep.set_index(["allocation_id", "scenario_id"]).sort_index()
    failures = []
    for run in runs:
        run_id = int(run.run_id)
        base = f"/Volumes/{catalog}/bronze/landing/phase5/run_{run_id}"
        try:
            candidates = pd.concat([
                _read_volume_parquet(w, f"{base}/stage1_candidates.parquet"),
                _read_volume_parquet(w, f"{base}/stage2_candidates.parquet"),
            ], ignore_index=True)
            results = pd.concat([
                _read_volume_parquet(w, f"{base}/stage1_results.parquet"),
                _read_volume_parquet(w, f"{base}/stage2_results.parquet"),
            ], ignore_index=True)
        except Exception as exc:
            failures.append(f"run {run_id}: {type(exc).__name__}")
            continue

        if candidates.allocation_id.duplicated().any():
            failures.append(f"run {run_id}: duplicate candidate ids")
            continue
        if set(candidates.allocation_id) != set(sweep.allocation_id):
            failures.append(f"run {run_id}: candidate ids do not match gold")
            continue
        spend_cols = [c for c in candidates.columns if c.startswith("spend_")]
        candidate_totals = candidates[spend_cols].sum(axis=1).to_numpy(float)
        gold_budgets = sweep["total_spend"].dropna().astype(float).unique()
        if len(gold_budgets) != 1 or not np.allclose(
                candidate_totals, gold_budgets[0], rtol=0.0, atol=1e-6):
            failures.append(f"run {run_id}: candidate spend totals do not match gold")
            continue
        persisted = results.set_index(["allocation_id", "scenario_id"]).sort_index()
        if not persisted.index.equals(gold.index):
            failures.append(f"run {run_id}: result keys do not match gold")
            continue
        if any(not np.allclose(persisted[col].to_numpy(float), gold[col].to_numpy(float),
                               rtol=1e-12, atol=1e-8, equal_nan=True)
               for col in metric_cols):
            failures.append(f"run {run_id}: aggregate metrics do not match gold")
            continue
        return candidates, run_id

    raise RuntimeError(
        "no successful Phase 5 candidate artifact matched live gold; "
        + "; ".join(failures[:5])
    )
