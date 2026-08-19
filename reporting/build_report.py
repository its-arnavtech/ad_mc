"""
Phase 6 -- build the budget-allocation report from the gold tables.

EVERY RESULT IN THE OUTPUT COMES FROM A QUERY RUN BY THIS SCRIPT. Nothing is
transcribed from an earlier phase's report, and nothing is re-simulated: the
Monte Carlo already ran, was verified, and lives in
`<catalog>.gold.{allocation_sweep_results, efficient_frontier,
frontier_recommendations, recommendation_channel_contributions}`. This module
reads those four tables and renders. Channel contribution rows were created by
an approved targeted rerun using the original gold seeds/path counts and exact
Phase 5 candidate spends, then reconciled to the authoritative allocation gold.
The report itself performs no candidate generation, RNG, or simulation.

    python reporting/build_report.py
    python reporting/build_report.py --catalog ad_mc_poc
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "simulation"))
sys.path.insert(0, str(REPO / "databricks"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_access as DA  # noqa: E402
from _common import resolve_warehouse_id, sql  # noqa: E402

DEFAULT_CATALOG = "ad_mc_poc"
DEFAULT_OUT = REPO / "reporting" / "budget_allocation_report.html"

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

def load(w, wid, gold: str):
    """Everything the report renders, in four read-only gold queries."""
    def frame(q):
        cols, rows = sql(w, wid, q)
        return pd.DataFrame(rows, columns=cols)

    sweep = frame(f"""
        SELECT allocation_id, scenario_id, stage, family, n_paths, seed,
               total_spend, mean_revenue, std_revenue, var_95, cvar_95, expected_roas,
               extrapolation_floor_applied, n_channels_floored
        FROM {gold}.allocation_sweep_results
    """)
    front = frame(f"""
        SELECT scenario_id, objective_pair, allocation_id, mean_revenue,
               std_revenue, var_95, cvar_95, rank_by_return,
               extrapolation_floor_applied
        FROM {gold}.efficient_frontier
    """)
    recs = frame(f"""
        SELECT scenario_id, objective_pair, recommendation, allocation_id,
               mean_revenue, std_revenue, var_95, cvar_95, expected_roas,
               n_efficient, ordering_unresolved, extrapolation_floor_applied,
               nearest_neighbour_gap
        FROM {gold}.frontier_recommendations
    """)
    contrib = frame(f"""
        SELECT scenario_id, objective_pair, recommendation, allocation_id, channel_id,
               phase5_workflow_run_id, n_paths, seed, total_spend, channel_spend,
               spend_share, mean_revenue_contribution, revenue_share,
               covariance_with_total_revenue, std_revenue_component, std_risk_share,
               mean_reconciliation_factor, std_reconciliation_factor, attribution_method
        FROM {gold}.recommendation_channel_contributions
    """)
    for df in (sweep, front, recs, contrib):
        for c in df.columns:
            if c in ("total_spend", "mean_revenue", "std_revenue", "var_95", "cvar_95",
                     "expected_roas", "nearest_neighbour_gap", "channel_spend",
                     "spend_share", "mean_revenue_contribution", "revenue_share",
                     "covariance_with_total_revenue", "std_revenue_component",
                     "std_risk_share", "mean_reconciliation_factor",
                     "std_reconciliation_factor"):
                df[c] = df[c].astype(float)
            elif c in ("rank_by_return", "n_efficient", "stage", "n_paths",
                       "n_channels_floored", "phase5_workflow_run_id"):
                df[c] = df[c].astype("Int64")
            elif c in ("extrapolation_floor_applied", "ordering_unresolved"):
                df[c] = df[c].astype(str).str.lower().eq("true")
    return sweep, front, recs, contrib


def contribution_breakdown(contrib: pd.DataFrame, recommendation: pd.Series) -> pd.DataFrame:
    """Five persisted channel rows for one exact recommendation key."""
    rows = contrib[
        (contrib.scenario_id == recommendation.scenario_id)
        & (contrib.objective_pair == recommendation.objective_pair)
        & (contrib.recommendation == recommendation.recommendation)
        & (contrib.allocation_id == recommendation.allocation_id)
    ].copy()
    if rows.empty or rows.channel_id.duplicated().any():
        raise RuntimeError(
            f"invalid contribution rows for {recommendation.scenario_id}/"
            f"{recommendation.objective_pair}/{recommendation.recommendation}")
    rows = rows.rename(columns={
        "channel_id": "channel", "channel_spend": "spend",
        "spend_share": "spend_pct", "mean_revenue_contribution": "revenue",
        "revenue_share": "revenue_pct", "std_revenue_component": "risk_component",
        "std_risk_share": "risk_pct",
    })
    return rows.sort_values("channel").reset_index(drop=True)


def validate_contributions(sweep: pd.DataFrame, recs: pd.DataFrame,
                           contrib: pd.DataFrame) -> tuple[int, str]:
    """Reject incomplete, duplicated, or unreconciled attribution gold."""
    keys = ["scenario_id", "objective_pair", "recommendation", "allocation_id"]
    if recs.duplicated(keys).any() or contrib.duplicated(keys + ["channel_id"]).any():
        raise RuntimeError("recommendation contribution keys must be unique")

    grouped = contrib.groupby(keys, as_index=False).agg(
        channels=("channel_id", "nunique"),
        spend_sum=("channel_spend", "sum"),
        total_spend=("total_spend", "max"),
        total_spend_min=("total_spend", "min"),
        spend_share_sum=("spend_share", "sum"),
        mean_sum=("mean_revenue_contribution", "sum"),
        revenue_share_sum=("revenue_share", "sum"),
        std_sum=("std_revenue_component", "sum"),
        risk_share_sum=("std_risk_share", "sum"),
    )
    expected_channels = int(contrib.channel_id.nunique())
    if expected_channels <= 0 or not (grouped.channels == expected_channels).all():
        raise RuntimeError("every recommendation must contain every attributed channel")
    joined = recs[keys + ["mean_revenue", "std_revenue"]].merge(
        grouped, on=keys, how="outer", validate="one_to_one", indicator=True)
    if len(joined) != len(recs) or not (joined._merge == "both").all():
        raise RuntimeError("contribution recommendation keys do not match gold")

    checks = (
        np.isclose(joined.spend_sum, joined.total_spend, rtol=0.0, atol=1e-6),
        np.isclose(joined.total_spend_min, joined.total_spend, rtol=0.0, atol=1e-12),
        np.isclose(joined.spend_share_sum, 1.0, rtol=0.0, atol=1e-12),
        np.isclose(joined.mean_sum, joined.mean_revenue, rtol=1e-12, atol=1e-6),
        np.isclose(joined.revenue_share_sum, 1.0, rtol=0.0, atol=1e-12),
        np.isclose(joined.std_sum, joined.std_revenue, rtol=1e-12, atol=1e-6),
        np.isclose(joined.risk_share_sum, 1.0, rtol=0.0, atol=1e-12),
    )
    if not all(check.all() for check in checks):
        raise RuntimeError("channel contribution totals do not reconcile to gold")

    cell_meta = contrib[["allocation_id", "scenario_id", "n_paths", "seed",
                         "total_spend"]].drop_duplicates()
    if cell_meta.duplicated(["allocation_id", "scenario_id"]).any():
        raise RuntimeError("one allocation/scenario has conflicting attribution lineage")
    sweep_meta = sweep[["allocation_id", "scenario_id", "n_paths", "seed",
                        "total_spend"]].rename(columns={
                            "n_paths": "gold_n_paths", "seed": "gold_seed",
                            "total_spend": "gold_total_spend",
                        })
    lineage = cell_meta.merge(
        sweep_meta, on=["allocation_id", "scenario_id"], how="left", validate="one_to_one")
    if (lineage.gold_n_paths.isna().any()
            or not (lineage.n_paths.astype(int) == lineage.gold_n_paths.astype(int)).all()
            or not (lineage.seed.astype(str) == lineage.gold_seed.astype(str)).all()
            or not np.allclose(lineage.total_spend, lineage.gold_total_spend,
                               rtol=0.0, atol=1e-6)):
        raise RuntimeError("channel contribution lineage does not match sweep gold")

    run_ids = contrib.phase5_workflow_run_id.dropna().astype(int).unique()
    methods = contrib.attribution_method.dropna().astype(str).unique()
    if len(run_ids) != 1 or len(methods) != 1:
        raise RuntimeError("channel contributions must have one Workflow run and method")
    return int(run_ids[0]), methods[0]


# ------------------------------------------------------------------ rendering

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Phase 6 report from verified Databricks outputs.")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG,
                        help=f"Unity Catalog catalog (default: {DEFAULT_CATALOG})")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT,
                        help=f"HTML output path (default: {DEFAULT_OUT})")
    return parser.parse_args(argv)


def report_run_shape(sweep: pd.DataFrame) -> tuple[float, int]:
    """Return the single (budget, paths) configuration represented by gold."""
    budgets = sweep["total_spend"].dropna().astype(float).unique()
    if len(budgets) != 1 or not np.isfinite(budgets[0]) or budgets[0] <= 0.0:
        raise RuntimeError(
            f"gold must contain one positive total_spend; found {budgets.tolist()}")
    path_counts = sweep["n_paths"].dropna().astype(int).unique()
    if len(path_counts) != 1 or path_counts[0] <= 0:
        raise RuntimeError(
            f"gold must contain one positive n_paths; found {path_counts.tolist()}")
    return float(budgets[0]), int(path_counts[0])


def main(argv=None) -> None:
    args = parse_args(argv)
    catalog = args.catalog.strip()
    if not catalog or not catalog.replace("_", "a").isalnum():
        raise ValueError(f"invalid catalog name {args.catalog!r}")
    gold = f"{catalog}.gold"

    w, wid = DA.open_session()
    wid = wid or resolve_warehouse_id(w)
    sweep, front, recs, contrib = load(w, wid, gold)
    print(f"loaded gold: sweep {len(sweep)}  frontier {len(front)}  recs {len(recs)}")
    contribution_run_id, attribution_method = validate_contributions(sweep, recs, contrib)

    total_budget, n_paths = report_run_shape(sweep)

    n_cand = sweep.allocation_id.nunique()
    top = recs[(recs.scenario_id == "normal")
               & (recs.objective_pair == "mean_revenue vs std_revenue")
               & (recs.recommendation == "max_return")].iloc[0]
    chan = contribution_breakdown(contrib, top)
    print(f"channel contributions loaded for {top.allocation_id}")

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
        ("Balanced", "The knee &mdash; where extra return stops paying for itself", pick(PAIR_MV, "balanced"), "std_revenue"),
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
            splits[_row.allocation_id] = contribution_breakdown(contrib, _row)
    print(f"channel splits built for {len(splits)} recommended allocations")

    n_ord = int(recs.ordering_unresolved.sum())
    n_floor = int(recs.extrapolation_floor_applied.sum())
    n_both = int((recs.ordering_unresolved & recs.extrapolation_floor_applied).sum())
    floor_front = int(front.extrapolation_floor_applied.sum())
    floor_sweep = int(sweep.extrapolation_floor_applied.sum())

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(
        sweep, front, recs, sizes, tiers, sens, chan, top,
        n_cand, n_ord, n_floor, n_both, floor_front, floor_sweep, splits,
        contribution_run_id, attribution_method, total_budget, n_paths,
    ), encoding="utf-8")
    print(f"wrote {output}  ({output.stat().st_size:,} bytes)")


def render(sweep, front, recs, sizes, tiers, sens, chan, top,
           n_cand, n_ord, n_floor, n_both, floor_front, floor_sweep, splits,
           contribution_run_id, attribution_method, total_budget, n_paths) -> str:
    from report_template import build  # keeps markup out of the query code
    return build(**locals())


if __name__ == "__main__":
    main()
