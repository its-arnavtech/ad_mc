"""
Phase 6 -- build the budget-allocation report from the gold tables.

EVERY NUMBER IN THE OUTPUT COMES FROM A QUERY RUN BY THIS SCRIPT. Nothing is
transcribed from an earlier phase's report, and nothing is re-simulated: the
Monte Carlo already ran, was verified, and lives in
`ad_mc_poc.gold.{allocation_sweep_results, efficient_frontier,
frontier_recommendations}`. This module reads those three tables and renders.

The ONE exception, stated because it is a real one: the channel-level
decomposition needs each candidate's per-channel spend, and gold is stored at
allocation grain, so the candidate DEFINITIONS are regenerated from
`frontier.generate_stage1/2`. That is a deterministic pure function of the
master seed. Those generators DO draw from a Dirichlet, so this is not "no
RNG" -- it is deterministic given the fixed seed, a weaker and more accurate
claim. No Monte Carlo paths are drawn; nothing is re-simulated. The result is
checked against gold before use. Per-channel expected revenue then comes from
`frontier.analytic_expected_revenue_weights`, the closed form the repo already
validates against the simulation, because gold does not carry a per-channel
breakdown.

    python reporting/build_report.py
"""

from __future__ import annotations

import html
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "simulation"))
sys.path.insert(0, str(REPO / "databricks"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_access as DA  # noqa: E402
import frontier as F  # noqa: E402
import saturation as SAT  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from _common import resolve_warehouse_id, sql  # noqa: E402

GOLD = "ad_mc_poc.gold"
OUT = REPO / "reporting" / "budget_allocation_report.html"

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


def channel_breakdown(w, wid, sweep: pd.DataFrame, allocation_id: str):
    """Per-channel spend, expected revenue and floor status for one allocation."""
    a = DA.load_assumptions(w, wid)
    ch = a["channel_id"].tolist()
    inputs = F.SweepInputs(
        channels=tuple(ch),
        mean_cvr=a["mean_cvr"].to_numpy(float), std_cvr=a["std_cvr"].to_numpy(float),
        mean_cpc=a["mean_cpc"].to_numpy(float), std_cpc=a["std_cpc"].to_numpy(float),
        mean_rpc=a["mean_revenue_per_conversion"].to_numpy(float),
        std_rpc=a["std_revenue_per_conversion"].to_numpy(float),
        correlation=DA.load_correlation_matrix(w, wid, ch, "cvr"),
        theta=SAT.theta_vector(ch, dict(SAT.DEFAULT_THETA)),
        spend_floor=SAT.spend_floor_vector(ch),
    )
    s1 = F.generate_stage1(ch)
    ids1 = {c.allocation_id for c in s1}
    df1 = sweep[sweep.allocation_id.isin(ids1)]
    seeds = [c for c in s1 if c.allocation_id in set(F.pareto_union_ids(df1))]
    s2 = F.generate_stage2(ch, seeds, 500_000.0)
    cands = {c.allocation_id: c for c in list(s1) + list(s2)}

    gold_ids = set(sweep.allocation_id.unique())
    if set(cands) != gold_ids:
        raise RuntimeError(
            f"regenerated candidate ids do not match gold "
            f"({len(set(cands) ^ gold_ids)} differ); refusing to report a "
            f"channel split that may not be the one that was simulated"
        )

    # MATCHING IDS IS NOT ENOUGH, and the gap is real: regenerating at a
    # different master seed yields the SAME 571 ids with different spend
    # vectors, so an id-only check would pass while every channel split shown
    # was wrong. gold already stores a fact derived from the spend vector --
    # how many channels fell below their floor -- so check against that.
    floor_v = SAT.spend_floor_vector(ch)
    gold_floored = (sweep.drop_duplicates("allocation_id")
                         .set_index("allocation_id")["n_channels_floored"])
    mismatched = []
    for aid, cand in cands.items():
        sp = np.array(cand.spend, dtype=float)
        if int(((sp > 0.0) & (sp < floor_v)).sum()) != int(gold_floored[aid]):
            mismatched.append(aid)
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} regenerated spend vectors disagree with gold's "
            f"n_channels_floored (e.g. {mismatched[:3]}); the ids line up but "
            f"the allocations do not, so the channel split would be wrong"
        )

    spend = np.array(cands[allocation_id].spend, dtype=float)
    normal = next(s for s in SCENARIOS if s.scenario_id == "normal")
    per = F.analytic_expected_revenue_weights(
        spend / spend.sum(), inputs, normal, per_channel=True)
    theta = SAT.theta_vector(ch, dict(SAT.DEFAULT_THETA))
    floor = SAT.spend_floor_vector(ch)
    rows = []
    for i, c in enumerate(ch):
        rows.append({
            "channel": c, "spend": spend[i], "spend_pct": spend[i] / spend.sum(),
            "revenue": per[i], "revenue_pct": per[i] / per.sum(),
            "roas": per[i] / spend[i] if spend[i] > 0 else float("nan"),
            "theta": theta[i], "floor": floor[i],
            "floored": bool(0.0 < spend[i] < floor[i]),
        })
    return pd.DataFrame(rows), inputs.correlation, ch


# ------------------------------------------------------------------ rendering

def main() -> None:
    w, wid = DA.open_session()
    wid = wid or resolve_warehouse_id(w)
    sweep, front, recs = load(w, wid)
    print(f"loaded gold: sweep {len(sweep)}  frontier {len(front)}  recs {len(recs)}")

    n_cand = sweep.allocation_id.nunique()
    top = recs[(recs.scenario_id == "normal")
               & (recs.recommendation == "max_return")].iloc[0]
    chan, corr, ch = channel_breakdown(w, wid, sweep, top.allocation_id)
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
                w, wid, sweep, _row.allocation_id)[0]
    print(f"channel splits built for {len(splits)} recommended allocations")

    n_ord = int(recs.ordering_unresolved.sum())
    n_floor = int(recs.extrapolation_floor_applied.sum())
    n_both = int((recs.ordering_unresolved & recs.extrapolation_floor_applied).sum())
    floor_front = int(front.extrapolation_floor_applied.sum())
    floor_sweep = int(sweep.extrapolation_floor_applied.sum())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(
        sweep, front, recs, sizes, tiers, sens, chan, corr, ch, top,
        n_cand, n_ord, n_floor, n_both, floor_front, floor_sweep, splits,
    ), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")


def render(sweep, front, recs, sizes, tiers, sens, chan, corr, ch, top,
           n_cand, n_ord, n_floor, n_both, floor_front, floor_sweep, splits) -> str:
    from report_template import build  # keeps markup out of the query code
    return build(**locals())


if __name__ == "__main__":
    main()
