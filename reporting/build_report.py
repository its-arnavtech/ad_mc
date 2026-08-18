"""
Phase 6 -- build the budget-allocation report from the gold tables.

EVERY RESULT IN THE OUTPUT COMES FROM A QUERY RUN BY THIS SCRIPT. Nothing is
transcribed from an earlier phase's report, and nothing is re-simulated: the
Monte Carlo already ran, was verified, and lives in
`ad_mc_poc.gold.{allocation_sweep_results, efficient_frontier,
frontier_recommendations}`. This module reads those three tables and renders.

The ONE source outside those tables is stated because it is a real data-model
boundary: gold is allocation-grain and does not store per-channel spend. The
report therefore reads the exact stage-1/stage-2 candidate Parquet artifacts
persisted by the latest successful Phase 5 Workflow run. Before use, their
971 allocation ids and aggregate result rows are checked back against live
gold. No candidate is regenerated, no RNG is called, and no channel revenue or
risk component is analytically recomputed. Channel "return link" and "risk
link" are descriptive correlations between exact spend shares and gold's
stored allocation-level mean/std metrics.

    python reporting/build_report.py
"""

from __future__ import annotations

import html
import io
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "simulation"))
sys.path.insert(0, str(REPO / "databricks"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_access as DA  # noqa: E402
from _common import resolve_warehouse_id, sql  # noqa: E402

GOLD = "ad_mc_poc.gold"
OUT = REPO / "reporting" / "budget_allocation_report.html"
PHASE5_JOB_ID = 94493651519110

SCEN_ORDER = ["seasonal_peak", "normal", "recession", "platform_algo_change"]
SCEN_LABEL = {
    "seasonal_peak": "Seasonal peak",
    "normal": "Normal",
    "recession": "Recession",
    "platform_algo_change": "Platform change",
}
PAIR_LABEL = {
    "mean_revenue vs std_revenue": "Return vs volatility",
    "mean_revenue vs var_95": "Return vs worst-case floor (VaR-95)",
    "mean_revenue vs cvar_95": "Return vs deep-tail average (CVaR-95)",
}


# ---------------------------------------------------------------- data access

def load(w, wid):
    """Everything the report renders, in four queries."""
    def frame(q):
        cols, rows = sql(w, wid, q)
        return pd.DataFrame(rows, columns=cols)

    sweep = frame(f"""
        SELECT allocation_id, scenario_id, stage, family,
               mean_revenue, std_revenue, var_95, cvar_95, expected_roas,
               extrapolation_floor_applied, n_channels_floored
        FROM {GOLD}.allocation_sweep_results
    """)
    front = frame(f"""
        SELECT scenario_id, objective_pair, allocation_id, mean_revenue,
               std_revenue, var_95, cvar_95, rank_by_return,
               extrapolation_floor_applied
        FROM {GOLD}.efficient_frontier
    """)
    recs = frame(f"""
        SELECT scenario_id, objective_pair, recommendation, allocation_id,
               mean_revenue, std_revenue, var_95, cvar_95, expected_roas,
               n_efficient, ordering_unresolved, extrapolation_floor_applied,
               nearest_neighbour_gap
        FROM {GOLD}.frontier_recommendations
    """)
    for df in (sweep, front, recs):
        for c in df.columns:
            if c in ("mean_revenue", "std_revenue", "var_95", "cvar_95",
                     "expected_roas", "nearest_neighbour_gap"):
                df[c] = df[c].astype(float)
            elif c in ("rank_by_return", "n_efficient", "stage", "n_channels_floored"):
                df[c] = df[c].astype("Int64")
            elif c in ("extrapolation_floor_applied", "ordering_unresolved"):
                df[c] = df[c].astype(str).str.lower().eq("true")
    return sweep, front, recs


def _read_volume_parquet(w, path: str) -> pd.DataFrame:
    response = w.files.download(path)
    if response.contents is None:
        raise RuntimeError(f"Databricks returned no content for {path}")
    return pd.read_parquet(io.BytesIO(response.contents.read()))


def load_persisted_candidates(w, sweep: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Exact candidate vectors from the latest successful Phase 5 run.

    The result Parquets are checked against live gold before the candidate
    vectors are trusted. This makes the Workflow artifact a persisted input,
    not a reconstructed approximation whose ids merely happen to line up.
    """
    runs = []
    for run in w.jobs.list_runs(job_id=PHASE5_JOB_ID, completed_only=True):
        result = getattr(getattr(run, "state", None), "result_state", None)
        if str(getattr(result, "value", result)).upper() == "SUCCESS":
            runs.append(run)
    runs.sort(key=lambda r: int(getattr(r, "start_time", 0) or 0), reverse=True)
    if not runs:
        raise RuntimeError(f"no successful runs found for Phase 5 job {PHASE5_JOB_ID}")

    metric_cols = ["mean_revenue", "std_revenue", "var_95", "cvar_95", "expected_roas"]
    gold = sweep.set_index(["allocation_id", "scenario_id"]).sort_index()
    failures = []
    for run in runs:
        run_id = int(run.run_id)
        base = f"/Volumes/ad_mc_poc/bronze/landing/phase5/run_{run_id}"
        try:
            candidates = pd.concat([
                _read_volume_parquet(w, f"{base}/stage1_candidates.parquet"),
                _read_volume_parquet(w, f"{base}/stage2_candidates.parquet"),
            ], ignore_index=True)
            results = pd.concat([
                _read_volume_parquet(w, f"{base}/stage1_results.parquet"),
                _read_volume_parquet(w, f"{base}/stage2_results.parquet"),
            ], ignore_index=True)
        except Exception as exc:  # try the next earlier successful run
            failures.append(f"run {run_id}: {type(exc).__name__}")
            continue

        if candidates.allocation_id.duplicated().any():
            failures.append(f"run {run_id}: duplicate candidate ids")
            continue
        if set(candidates.allocation_id) != set(sweep.allocation_id):
            failures.append(f"run {run_id}: candidate ids do not match gold")
            continue
        persisted = results.set_index(["allocation_id", "scenario_id"]).sort_index()
        if not persisted.index.equals(gold.index):
            failures.append(f"run {run_id}: result keys do not match gold")
            continue
        mismatch = False
        for col in metric_cols:
            if not np.allclose(persisted[col].to_numpy(float), gold[col].to_numpy(float),
                               rtol=1e-12, atol=1e-8, equal_nan=True):
                failures.append(f"run {run_id}: {col} does not match gold")
                mismatch = True
                break
        if mismatch:
            continue
        return candidates, run_id

    raise RuntimeError(
        "no successful Phase 5 candidate artifact matched live gold; "
        + "; ".join(failures[:5])
    )


def channel_breakdown(candidates: pd.DataFrame, sweep: pd.DataFrame,
                      allocation_id: str) -> pd.DataFrame:
    """Exact spend plus descriptive links to gold return and volatility."""
    spend_cols = [c for c in candidates.columns if c.startswith("spend_")]
    channels = [c.removeprefix("spend_") for c in spend_cols]
    row = candidates.set_index("allocation_id").loc[allocation_id]
    spend = row[spend_cols].to_numpy(float)

    normal = (sweep[sweep.scenario_id == "normal"]
              [["allocation_id", "mean_revenue", "std_revenue"]]
              .drop_duplicates("allocation_id"))
    portfolio = candidates[["allocation_id", *spend_cols]].merge(
        normal, on="allocation_id", validate="one_to_one")

    rows = []
    for col, channel, amount in zip(spend_cols, channels, spend):
        shares = portfolio[col].to_numpy(float) / 500_000.0
        rows.append({
            "channel": channel,
            "spend": amount,
            "spend_pct": amount / spend.sum(),
            "return_corr": float(np.corrcoef(
                shares, portfolio["mean_revenue"].to_numpy(float))[0, 1]),
            "risk_corr": float(np.corrcoef(
                shares, portfolio["std_revenue"].to_numpy(float))[0, 1]),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rendering

def main() -> None:
    w, wid = DA.open_session()
    wid = wid or resolve_warehouse_id(w)
    sweep, front, recs = load(w, wid)
    print(f"loaded gold: sweep {len(sweep)}  frontier {len(front)}  recs {len(recs)}")

    n_cand = sweep.allocation_id.nunique()
    candidates, candidate_run_id = load_persisted_candidates(w, sweep)
    print(f"candidate vectors loaded from verified Phase 5 run {candidate_run_id}")
    top = recs[(recs.scenario_id == "normal")
               & (recs.recommendation == "max_return")].iloc[0]
    chan = channel_breakdown(candidates, sweep, top.allocation_id)
    print(f"channel breakdown built for {top.allocation_id}")

    # --- frontier sizes, straight from gold
    sizes = (front.groupby(["objective_pair", "scenario_id"])
                  .size().rename("n").reset_index())

    # --- the three headline picks under 'normal', by risk appetite
    def pick(pair, kind):
        m = recs[(recs.scenario_id == "normal") & (recs.objective_pair == pair)
                 & (recs.recommendation == kind)]
        return m.iloc[0]

    PAIR_MV = "mean_revenue vs std_revenue"
    PAIR_VAR = "mean_revenue vs var_95"
    tiers = [
        ("Conservative", "Lowest volatility on the frontier", pick(PAIR_MV, "min_risk"), "std_revenue"),
        ("Balanced", "The knee — where extra return stops paying for itself", pick(PAIR_MV, "balanced"), "std_revenue"),
        ("Aggressive", "Highest expected revenue on the frontier", pick(PAIR_MV, "max_return"), "std_revenue"),
        ("Floor-protective", "Highest worst-case floor (VaR-95)", pick(PAIR_VAR, "min_risk"), "var_95"),
        ("Balanced, worst-case view", "The knee of the worst-case frontier",
         pick(PAIR_VAR, "balanced"), "var_95"),
    ]
    tier_ids = [t[2].allocation_id for t in tiers]
    sens = sweep[sweep.allocation_id.isin(tier_ids)]

    # A reader cannot act on "put $546,276 here" without knowing the split, so
    # EVERY recommendation gets one, not only the headline pick.
    splits = {}
    for _l, _w, _row, _r in tiers:
        if _row.allocation_id not in splits:
            splits[_row.allocation_id] = channel_breakdown(
                candidates, sweep, _row.allocation_id)
    print(f"channel splits built for {len(splits)} recommended allocations")

    n_ord = int(recs.ordering_unresolved.sum())
    n_floor = int(recs.extrapolation_floor_applied.sum())
    n_both = int((recs.ordering_unresolved & recs.extrapolation_floor_applied).sum())
    floor_front = int(front.extrapolation_floor_applied.sum())
    floor_sweep = int(sweep.extrapolation_floor_applied.sum())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(
        sweep, front, recs, sizes, tiers, sens, chan, top,
        n_cand, n_ord, n_floor, n_both, floor_front, floor_sweep, splits,
        candidate_run_id,
    ), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")


def render(sweep, front, recs, sizes, tiers, sens, chan, top,
           n_cand, n_ord, n_floor, n_both, floor_front, floor_sweep, splits,
           candidate_run_id) -> str:
    from report_template import build  # keeps markup out of the query code
    return build(**locals())


if __name__ == "__main__":
    main()
