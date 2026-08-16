"""Turn a finished sweep into the three gold frames. No I/O, no simulation.

WHY THIS MODULE EXISTS
----------------------
Phase 4 built the gold rows inline in `load_phase4_gold.main()`, which was fine
while the sweep ran locally and wrote over the SQL API. Phase 5 runs the same
sweep as a Databricks Workflow task, in a different process, on serverless job
compute -- and it has to produce BITWISE the same gold. Two copies of this
transformation would be two things to keep in step, and the first divergence
would look exactly like a real modelling change.

So the transformation moved here VERBATIM and both callers use it:

    simulation/load_phase4_gold.py                 (local, SQL warehouse)
    simulation/phase5_tasks.py::run_gold_aggregate (Workflow task, Spark)

NOTHING HERE IS NEW LOGIC. The Pareto computation, the recommendation rule, the
flag derivation and the column lists are the Phase 4 ones, moved. The only
addition is `canonical_cell_order`, which exists because the distributed sweep
returns rows in whatever order the Spark tasks finished in, and the frontier
helpers break exact ties by row position -- so the rows are put back into the
order `frontier.run_sweep` would have produced before anything reads them. That
removes scheduling order as an input to the answer; it does not change the
answer for any non-tied input.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

try:  # package-style import (simulation.gold_assembly / ad_mc_sim.gold_assembly)
    from . import frontier as F
    from . import saturation as SAT
    from .frontier import Candidate
    from .scenarios import SCENARIOS, Scenario
except ImportError:  # flat import, which is how the rest of this repo runs
    import frontier as F
    import saturation as SAT
    from frontier import Candidate
    from scenarios import SCENARIOS, Scenario


# Column lists, in the order the gold tables declare them. Kept here so the two
# writers cannot disagree about column ORDER either -- an INSERT ... VALUES is
# positional and a swapped pair of DOUBLEs would be silent.
SWEEP_TABLE_COLUMNS: list[str] = [
    "allocation_id", "scenario_id", "stage", "family", "n_paths", "seed",
    "total_spend", "mean_revenue", "se_mean_revenue", "std_revenue",
    "median_revenue", "min_revenue", "max_revenue", "var_95", "cvar_95",
    "expected_roas", "extrapolation_floor_applied", "n_channels_floored",
]

FRONTIER_TABLE_COLUMNS: list[str] = [
    "scenario_id", "objective_pair", "allocation_id", "mean_revenue",
    "std_revenue", "var_95", "cvar_95", "expected_roas", "rank_by_return",
    "extrapolation_floor_applied",
]

REC_TABLE_COLUMNS: list[str] = [
    "scenario_id", "objective_pair", "recommendation", "allocation_id",
    "mean_revenue", "std_revenue", "var_95", "cvar_95", "expected_roas",
    "n_efficient", "balanced_is_degenerate", "nearest_neighbour_gap",
    "ordering_unresolved", "extrapolation_floor_applied",
]


def canonical_cell_order(
    candidates: Sequence[Candidate], scenarios: Sequence[Scenario] = SCENARIOS
) -> list[tuple[str, str]]:
    """The (scenario_id, allocation_id) order `frontier.run_sweep` emits.

    `run_sweep` loops scenario-major over the candidate list, and
    `load_phase4_gold` then concatenates stage 1's frame before stage 2's. So the
    canonical order is scenario-major WITHIN each stage, stages in order. Pass
    the full candidate list (stage 1 then stage 2) and that is what comes back.

    This matters only for exact ties: `sort_values`, `np.lexsort` and the knee's
    1e-12 tolerance all resolve ties by position. Ties are vanishingly unlikely
    on continuous Monte Carlo output, but "unlikely" is not a reproducibility
    guarantee, and re-imposing the order costs nothing.
    """
    stages = sorted({int(c.stage) for c in candidates})
    order: list[tuple[str, str]] = []
    for stage in stages:
        in_stage = [c for c in candidates if int(c.stage) == stage]
        for scenario in scenarios:
            for cand in in_stage:
                order.append((scenario.scenario_id, cand.allocation_id))
    return order


def reorder_cells(
    df: pd.DataFrame, candidates: Sequence[Candidate],
    scenarios: Sequence[Scenario] = SCENARIOS,
) -> pd.DataFrame:
    """Reindex `df` into `canonical_cell_order`. Raises on a missing/extra cell."""
    want = canonical_cell_order(candidates, scenarios)
    have = list(zip(df["scenario_id"].astype(str), df["allocation_id"].astype(str)))
    if len(have) != len(set(have)):
        raise ValueError("duplicate (scenario_id, allocation_id) cells in the sweep result")
    missing = set(want) - set(have)
    extra = set(have) - set(want)
    if missing or extra:
        raise ValueError(
            f"sweep result does not match the candidate set: {len(missing)} missing "
            f"(e.g. {sorted(missing)[:3]}), {len(extra)} unexpected "
            f"(e.g. {sorted(extra)[:3]})"
        )
    pos = {key: i for i, key in enumerate(want)}
    out = df.copy()
    out["_ord"] = [pos[k] for k in have]
    out = out.sort_values("_ord", kind="stable").drop(columns=["_ord"])
    return out.reset_index(drop=True)


def annotate_sweep(
    df: pd.DataFrame, candidates: Sequence[Candidate], spend_floor: np.ndarray | None
) -> pd.DataFrame:
    """Add stage / family / floor columns and stringify the seed.

    The extrapolation flag is a pure function of (allocation, floor), so it is
    recomputed here from the candidate set rather than trusted to have survived
    the sweep's fixed output schema. That also means a cached sweep from before
    the flag existed still loads correctly.
    """
    meta = {c.allocation_id: c for c in candidates}
    out = df.copy()
    out["stage"] = out["allocation_id"].map(lambda a: meta[a].stage)
    out["family"] = out["allocation_id"].map(lambda a: meta[a].family)

    floored_n = {}
    for cand in candidates:
        _, mask = SAT.apply_spend_floor(np.array(cand.spend, dtype=float), spend_floor)
        floored_n[cand.allocation_id] = int(mask.sum())
    out["n_channels_floored"] = out["allocation_id"].map(floored_n)
    out["extrapolation_floor_applied"] = out["n_channels_floored"] > 0
    out["seed"] = out["seed"].map(str)   # unsigned 64-bit does not fit a BIGINT
    return out


def build_frontier_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The non-dominated subset per (objective pair, scenario), ranked by return."""
    rows = []
    for ret, risk, higher_better in F.OBJECTIVE_PAIRS:
        pair = f"{ret} vs {risk}"
        for scenario_id, sub in df.groupby("scenario_id", sort=True):
            eff = F.frontier_for_pair(sub, ret, risk, higher_better)
            eff = eff[eff["is_efficient"]].sort_values(ret, ascending=False)
            for rank, (_, row) in enumerate(eff.iterrows(), start=1):
                rows.append({
                    "scenario_id": scenario_id, "objective_pair": pair,
                    "allocation_id": row["allocation_id"],
                    "mean_revenue": row["mean_revenue"], "std_revenue": row["std_revenue"],
                    "var_95": row["var_95"], "cvar_95": row["cvar_95"],
                    "expected_roas": row["expected_roas"], "rank_by_return": rank,
                    "extrapolation_floor_applied": bool(row["extrapolation_floor_applied"]),
                })
    return pd.DataFrame(rows, columns=FRONTIER_TABLE_COLUMNS)


def build_recommendation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Three named picks per (scenario, objective pair), under `frontier`'s rule."""
    recs = pd.concat(
        [F.recommend_from_frontier(df, ret, risk, hb) for ret, risk, hb in F.OBJECTIVE_PAIRS],
        ignore_index=True,
    )
    return recs[REC_TABLE_COLUMNS]


def build_gold_frames(
    df: pd.DataFrame,
    candidates: Sequence[Candidate],
    spend_floor: np.ndarray | None,
    scenarios: Sequence[Scenario] = SCENARIOS,
    reorder: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(sweep, frontier, recommendations), ready to write.

    `reorder=True` puts the cells back into `canonical_cell_order` first; the
    local runner leaves it False because `run_sweep` already emits that order,
    and the Workflow sets it True because Spark does not.
    """
    annotated = annotate_sweep(df, candidates, spend_floor)
    if reorder:
        annotated = reorder_cells(annotated, candidates, scenarios)
    sweep = annotated[SWEEP_TABLE_COLUMNS]
    return sweep, build_frontier_frame(annotated), build_recommendation_frame(annotated)
