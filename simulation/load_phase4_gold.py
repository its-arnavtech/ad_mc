"""
Phase 4 -- run the two-stage sweep and load the gold frontier tables.

    python simulation/load_phase4_gold.py            # full sweep, then load
    python simulation/load_phase4_gold.py --load-only  # reuse a cached sweep

WHAT THIS DOES
--------------
1. Builds stage-1 candidates (broad simplex coverage), evaluates every
   (candidate, scenario) cell, and takes the Pareto union.
2. Builds stage-2 candidates by refining around that union, and evaluates those.
3. Computes the efficient frontier for three objective pairs and the three
   fixed recommendations per frontier.
4. Writes all of it to gold.

SEEDING -- COMMON RANDOM NUMBERS, and why that reverses Phase 3
--------------------------------------------------------------
Phase 3 gave every (allocation, scenario) cell its own independent stream, so
each cell's estimate stood alone. A FRONTIER is not made of standalone
estimates: it is made of PAIRWISE DOMINANCE comparisons, and with independent
streams each comparison carries the full standard error of a difference.

MEASURED, at 300 replicates per arm, 10,000 paths per simulation. The gain is
not one number: CRN cancels COMMON noise, so it grows as two allocations get
more similar -- which is exactly the regime a frontier lives in, since adjacent
efficient points are by construction close.

    allocation pair          SD of difference        variance
                             CRN      independent    reduction   95% CI
    distant (even vs tilt)   $647     $2,638          16.6x      [13.2, 20.9]
    near-frontier            $561     $2,756          24.1x      [19.2, 30.3]
    nearly identical          $45     $2,974        4369.9x      [3482, 5484]

The independent arm sits at $2,600-$3,000 regardless of the pair: it gets no
benefit from similarity. No detectable bias in any pair (max |t| = 0.68 on the
difference of the two arms' means).

WHAT THAT BUYS, AND WHAT IT DOES NOT. Realized adjacent gaps along the `normal`
mean-VaR frontier are $9,961 / $2,714 / $6,777 / $5,971 / $323. At the
independent SE most of those would be noise. Under CRN most are resolved -- but
not all: that last pair was measured directly and its own CRN difference SD is
$322, so a $323 gap is 1.0x the noise and those two points are NOT ordered by
this sweep. `recommend_from_frontier` flags exactly that case rather than
quietly picking a winner.

Every allocation within a scenario therefore shares one seed (`crn_seed`).
Scenarios keep DIFFERENT seeds, checked by `assert_scenarios_distinct`, so a
normal-vs-recession contrast is not silently paired as well.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "databricks"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_access as DA  # noqa: E402
import frontier as F  # noqa: E402
import gold_assembly as GA  # noqa: E402
import saturation as SAT  # noqa: E402
from config import TOTAL_BUDGET  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from _common import get_client, resolve_warehouse_id, sql  # noqa: E402

TBL_SWEEP = "ad_mc_poc.gold.allocation_sweep_results"
TBL_FRONTIER = "ad_mc_poc.gold.efficient_frontier"
TBL_RECS = "ad_mc_poc.gold.frontier_recommendations"

CACHE = REPO_ROOT / "data" / "processed" / "phase4_sweep.pkl"
BATCH = 250

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def _lit(v) -> str:
    """SQL literal. NULL for NaN/None so a missing value never becomes 0.0."""
    if v is None:
        return "NULL"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "NULL" if not np.isfinite(v) else repr(float(v))
    s = str(v).replace("'", "''")
    return f"'{s}'"


def insert_rows(w, wid, table: str, columns: list[str], rows: list[dict]) -> None:
    """INSERT OVERWRITE in batches; the first batch replaces, the rest append.

    OVERWRITE not APPEND for the same reason Phase 3 used it: these tables have
    no run_id, so a second run would otherwise stack a duplicate sweep that any
    aggregate would silently average.
    """
    if not rows:
        raise ValueError(f"refusing to write zero rows to {table}")
    collist = ", ".join(columns)
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        values = ",\n".join(
            "(" + ", ".join(_lit(r.get(c)) for c in columns) + ")" for r in chunk
        )
        verb = "INSERT OVERWRITE" if i == 0 else "INSERT INTO"
        sql(w, wid, f"{verb} {table} ({collist}) VALUES\n{values}")
        print(f"    {min(i + BATCH, len(rows)):>5} / {len(rows)} rows", flush=True)


def build_inputs(w, wid) -> tuple[F.SweepInputs, list[str]]:
    a = DA.load_assumptions(w, wid)
    ch = a["channel_id"].tolist()
    return F.SweepInputs(
        channels=tuple(ch),
        mean_cvr=a["mean_cvr"].to_numpy(), std_cvr=a["std_cvr"].to_numpy(),
        mean_cpc=a["mean_cpc"].to_numpy(), std_cpc=a["std_cpc"].to_numpy(),
        mean_rpc=a["mean_revenue_per_conversion"].to_numpy(),
        std_rpc=a["std_revenue_per_conversion"].to_numpy(),
        correlation=DA.load_correlation_matrix(w, wid, ch, "cvr"),
        theta=SAT.theta_vector(ch, dict(SAT.DEFAULT_THETA)),
        spend_floor=SAT.spend_floor_vector(ch),
    ), ch


def assert_floor_snapshot_matches_live(w, wid) -> dict[str, float]:
    """Re-derive the spend floors from live bronze and check the frozen copy.

    The floors are frozen in `saturation.py` so they are available without a
    warehouse round trip, which means they can silently go stale if bronze is
    ever reloaded. This is the call that stops that -- it exists because the
    guard shipped uncalled the first time, which made the comment claiming it
    're-validates' the snapshot untrue of any code path.
    """
    q = float(SAT.BRONZE_SPEND_FLOOR_PERCENTILE)
    _c, rows = sql(w, wid, f"""
        SELECT channel_id, PERCENTILE(spend, {q}) AS p
        FROM ad_mc_poc.bronze.channel_performance_history
        GROUP BY channel_id
    """)
    live = {r[0]: float(r[1]) for r in rows}
    diffs = SAT.check_spend_floor_snapshot(live)
    print(f"  spend-floor snapshot vs live bronze: max rel {max(diffs.values()):.2e} "
          f"(frozen values are the live p5 rounded to the dollar)")
    return diffs


def run_two_stage(inputs: F.SweepInputs, ch: list[str]):
    def tick(done, total):
        if done % 500 == 0 or done == total:
            print(f"    {done}/{total} cells", flush=True)

    s1 = F.generate_stage1(ch)
    F.assert_candidate_set_valid(s1, ch, TOTAL_BUDGET)
    print(f"stage 1: {len(s1)} candidates")
    t0 = time.time()
    df1 = F.run_sweep(s1, inputs, SCENARIOS, progress=tick)
    print(f"  {len(df1)} cells in {time.time() - t0:.1f}s")

    seed_ids = set(F.pareto_union_ids(df1))
    seeds = [c for c in s1 if c.allocation_id in seed_ids]
    print(f"  stage-1 Pareto union: {len(seeds)} points")

    s2 = F.generate_stage2(ch, seeds, TOTAL_BUDGET)
    F.assert_candidate_set_valid(s2, ch, TOTAL_BUDGET)
    print(f"stage 2: {len(s2)} candidates")
    t0 = time.time()
    df2 = F.run_sweep(s2, inputs, SCENARIOS, progress=tick)
    print(f"  {len(df2)} cells in {time.time() - t0:.1f}s")

    return list(s1) + list(s2), pd.concat([df1, df2], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-only", action="store_true",
                    help="reuse the cached sweep instead of re-running it")
    args = ap.parse_args()

    print("=" * 72)
    print("PHASE 4 -- OPTIMIZATION SWEEP -> GOLD")
    print("=" * 72)
    w = get_client()
    wid = resolve_warehouse_id(w)
    inputs, ch = build_inputs(w, wid)
    assert_floor_snapshot_matches_live(w, wid)

    if args.load_only and CACHE.exists():
        with CACHE.open("rb") as fh:
            cached = pickle.load(fh)
        candidates, df = cached["candidates"], cached["df"]
        print(f"loaded cached sweep: {len(candidates)} candidates, {len(df)} cells")
    else:
        candidates, df = run_two_stage(inputs, ch)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("wb") as fh:
            pickle.dump({"candidates": candidates, "df": df, "channels": ch}, fh)

    # The three frames are built by `gold_assembly`, which the Phase 5 Workflow
    # task calls too. Same code, same column order, so the local path and the
    # Databricks path cannot drift. `reorder=False`: `run_sweep` already emits
    # the canonical scenario-major order here.
    sweep, frontier_df, recs = GA.build_gold_frames(
        df, candidates, inputs.spend_floor, reorder=False
    )

    print(f"\ntotal: {len(candidates)} candidates, {len(df)} cells, "
          f"{len(df) * inputs.n_paths:,} paths")

    # ---- gold 1: every candidate -------------------------------------------
    print(f"\nwriting {TBL_SWEEP}")
    insert_rows(w, wid, TBL_SWEEP, GA.SWEEP_TABLE_COLUMNS, sweep.to_dict("records"))

    # ---- gold 2: the frontier ----------------------------------------------
    print(f"\nwriting {TBL_FRONTIER} ({len(frontier_df)} rows)")
    insert_rows(w, wid, TBL_FRONTIER, GA.FRONTIER_TABLE_COLUMNS,
                frontier_df.to_dict("records"))

    # ---- gold 3: recommendations -------------------------------------------
    print(f"\nwriting {TBL_RECS} ({len(recs)} rows)")
    insert_rows(w, wid, TBL_RECS, GA.REC_TABLE_COLUMNS, recs.to_dict("records"))

    # ---- verify -------------------------------------------------------------
    print("\n== verification against the live tables ==")
    _c, r = sql(w, wid, f"SELECT COUNT(*), COUNT(DISTINCT allocation_id), "
                        f"COUNT(DISTINCT scenario_id) FROM {TBL_SWEEP}")
    check("sweep rows == candidates x scenarios",
          int(r[0][0]) == len(candidates) * len(SCENARIOS),
          f"{r[0][0]} rows, {r[0][1]} allocations, {r[0][2]} scenarios")
    _c, r = sql(w, wid, f"SELECT COUNT(*) FROM {TBL_SWEEP} WHERE "
                        f"mean_revenue IS NULL OR var_95 IS NULL OR cvar_95 IS NULL")
    check("no null metrics", int(r[0][0]) == 0, f"{r[0][0]} null rows")
    _c, r = sql(w, wid, f"SELECT COUNT(*) FROM {TBL_SWEEP} WHERE cvar_95 > var_95")
    check("CVaR-95 <= VaR-95 everywhere (the tail mean cannot exceed its own cutoff)",
          int(r[0][0]) == 0, f"{r[0][0]} violations")
    _c, r = sql(w, wid, f"SELECT COUNT(*) FROM {TBL_SWEEP} WHERE ABS(total_spend - 500000) > 1e-6")
    check("every candidate spends the full budget", int(r[0][0]) == 0, f"{r[0][0]} violations")
    _c, r = sql(w, wid, f"SELECT COUNT(*) FROM {TBL_FRONTIER} f LEFT JOIN {TBL_SWEEP} s "
                        f"ON f.allocation_id = s.allocation_id AND f.scenario_id = s.scenario_id "
                        f"WHERE s.allocation_id IS NULL")
    check("every frontier row references a real sweep row", int(r[0][0]) == 0, f"{r[0][0]} orphans")
    _c, r = sql(w, wid, f"SELECT COUNT(*) FROM {TBL_SWEEP} WHERE "
                        f"extrapolation_floor_applied IS NULL OR n_channels_floored IS NULL")
    check("the extrapolation flag is populated on every sweep row",
          int(r[0][0]) == 0, f"{r[0][0]} null rows")
    _c, r = sql(w, wid, f"SELECT COUNT(*) FROM {TBL_SWEEP} WHERE "
                        f"(n_channels_floored > 0) <> extrapolation_floor_applied")
    check("the flag agrees with the channel count it summarises",
          int(r[0][0]) == 0, f"{r[0][0]} inconsistent rows")
    _c, r = sql(w, wid, f"SELECT objective_pair, COUNT(*) FROM {TBL_RECS} GROUP BY objective_pair")
    check("3 recommendations x 4 scenarios for each of 3 objective pairs",
          all(int(x[1]) == 12 for x in r) and len(r) == 3, str([(x[0], x[1]) for x in r]))

    print("\n== frontier size per scenario and objective pair ==")
    cols, rows = sql(w, wid, f"""
        SELECT objective_pair, scenario_id, COUNT(*) AS n_efficient
        FROM {TBL_FRONTIER} GROUP BY objective_pair, scenario_id
        ORDER BY objective_pair, scenario_id
    """)
    for row in rows:
        print(f"  {row[0]:<32} {row[1]:<22} {row[2]:>4}")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"GOLD LOAD FAILED -- {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("GOLD LOAD PASSED.")


if __name__ == "__main__":
    main()
