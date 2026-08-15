"""
Phase 4 -- create the gold tables the optimization sweep populates.

WHY GOLD, AND WHAT GRAIN
------------------------
Silver holds PATH-level simulation output: 840,000 rows for the Phase 3 grid,
one per (allocation, scenario, path). That grain is what a risk metric is
computed FROM, but it is not what a decision is made from.

Gold is the decision layer, so its grain is one row per (allocation, scenario)
-- the aggregate already reduced to mean / std / VaR / CVaR / ROAS. The Phase 4
sweep evaluates ~971 candidates x 4 scenarios; at path level that would be
~38.8 MILLION rows with no consumer, since nothing downstream re-derives a
percentile from raw paths. The aggregation happens inside the simulation UDF,
where the paths already live in memory, and only the summary crosses the wire.

THREE TABLES, THREE DIFFERENT QUESTIONS
---------------------------------------
`allocation_sweep_results`  -- every candidate's risk/return. The full search
                               space, so a later phase can re-cut the frontier
                               under a different risk measure without re-running
                               38.8M paths.
`efficient_frontier`        -- the non-dominated subset, per (scenario,
                               objective pair). Derived from the table above and
                               deliberately materialised: the Pareto computation
                               is O(n^2) and the answer is what gets read.
`frontier_recommendations`  -- three named picks per (scenario, objective pair)
                               under a FIXED rule (max return / min risk / knee),
                               so the recommendation is reproducible rather than
                               a judgement call made once in a notebook.

NOT PARTITIONED, same reasoning as silver: a few thousand rows of short strings
and doubles is a single Parquet file, and Hive partitioning would only create
many tiny ones.

    python databricks/08_create_gold_frontier.py
"""

from __future__ import annotations

import sys

from _common import (
    CATALOG,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)

GOLD = f"{CATALOG}.gold"
TBL_SWEEP = f"{GOLD}.allocation_sweep_results"
TBL_FRONTIER = f"{GOLD}.efficient_frontier"
TBL_RECS = f"{GOLD}.frontier_recommendations"

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


# Applied on every run, so metadata converges even on tables whose columns were
# added by ALTER after the original CREATE.
COLUMN_COMMENTS: list[tuple[str, str, str]] = [
    (TBL_SWEEP, "extrapolation_floor_applied", "TRUE when at least one funded channel in this allocation fell below its per-channel evidence floor (p5 of that channel's observed daily spend), so its CPC was priced AT the floor rather than extrapolated below anything bronze has observed. Respect this the same way as ordering_unresolved: the number is defensible, but it sits at the edge of what the data can support."),
    (TBL_SWEEP, "n_channels_floored",
     "How many channels were clipped to their evidence floor; 0 means the whole "
     "allocation sits inside the observed spend range."),
    (TBL_FRONTIER, "extrapolation_floor_applied", "TRUE when at least one funded channel in this allocation fell below its per-channel evidence floor (p5 of that channel's observed daily spend), so its CPC was priced AT the floor rather than extrapolated below anything bronze has observed. Respect this the same way as ordering_unresolved: the number is defensible, but it sits at the edge of what the data can support."),
    (TBL_RECS, "extrapolation_floor_applied", "TRUE when at least one funded channel in this allocation fell below its per-channel evidence floor (p5 of that channel's observed daily spend), so its CPC was priced AT the floor rather than extrapolated below anything bronze has observed. Respect this the same way as ordering_unresolved: the number is defensible, but it sits at the edge of what the data can support."),
]

DDL = [
    (TBL_SWEEP, f"""
        CREATE TABLE IF NOT EXISTS {TBL_SWEEP} (
            allocation_id     STRING  NOT NULL COMMENT 'Candidate id; stable and derived from its generating family',
            scenario_id       STRING  NOT NULL COMMENT 'FK to bronze.scenario_definitions',
            stage             INT     COMMENT '1 = broad coverage, 2 = local refinement around the stage-1 Pareto set',
            family            STRING  COMMENT 'How the candidate was generated: phase3, lattice3/4, dirichlet_*, refine, blend',
            n_paths           INT     COMMENT 'Monte Carlo paths behind this row',
            seed              STRING  COMMENT 'Decimal string: the seed is an unsigned 64-bit value that does not fit a BIGINT',
            total_spend       DOUBLE  COMMENT 'Always the full budget; the sweep is over the mix, not the level',
            mean_revenue      DOUBLE  COMMENT 'Expected total revenue across paths',
            se_mean_revenue   DOUBLE  COMMENT 'Monte Carlo standard error OF THE MEAN for this cell alone. Under common random numbers, differences BETWEEN allocations are far more precise than this suggests',
            std_revenue       DOUBLE  COMMENT 'Path-level standard deviation -- the mean-variance risk axis',
            median_revenue    DOUBLE,
            min_revenue       DOUBLE,
            max_revenue       DOUBLE,
            var_95            DOUBLE  COMMENT 'VaR-95: 5th percentile of revenue. HIGHER is better',
            cvar_95           DOUBLE  COMMENT 'CVaR-95: mean of the worst 5% of paths. HIGHER is better',
            expected_roas     DOUBLE  COMMENT 'mean_revenue / total_spend',
            extrapolation_floor_applied BOOLEAN COMMENT 'TRUE when at least one funded channel in this allocation fell below its per-channel evidence floor, so its CPC was priced AT the floor rather than extrapolated below the observed data. Respect this the same way as ordering_unresolved: the number is defensible, but it sits at the edge of what bronze can support.',
            n_channels_floored INT    COMMENT 'How many channels were clipped; 0 means the allocation is entirely inside the observed range'
        )
        USING DELTA
        COMMENT 'Phase 4 optimization sweep: risk and return for every candidate allocation under every scenario. One row per (allocation, scenario) -- path level stays in silver.'
    """),
    (TBL_FRONTIER, f"""
        CREATE TABLE IF NOT EXISTS {TBL_FRONTIER} (
            scenario_id       STRING  NOT NULL,
            objective_pair    STRING  NOT NULL COMMENT 'e.g. "mean_revenue vs var_95" -- which two objectives this frontier trades off',
            allocation_id     STRING  NOT NULL,
            mean_revenue      DOUBLE,
            std_revenue       DOUBLE,
            var_95            DOUBLE,
            cvar_95           DOUBLE,
            expected_roas     DOUBLE,
            rank_by_return    INT     COMMENT 'Position along the frontier, 1 = highest expected revenue',
            extrapolation_floor_applied BOOLEAN COMMENT 'TRUE when at least one funded channel in this allocation fell below its per-channel evidence floor, so its CPC was priced AT the floor rather than extrapolated below the observed data. Respect this the same way as ordering_unresolved: the number is defensible, but it sits at the edge of what bronze can support.'
        )
        USING DELTA
        COMMENT 'The non-dominated (Pareto-efficient) allocations per scenario and objective pair. Derived from allocation_sweep_results.'
    """),
    (TBL_RECS, f"""
        CREATE TABLE IF NOT EXISTS {TBL_RECS} (
            scenario_id            STRING  NOT NULL,
            objective_pair         STRING  NOT NULL,
            recommendation         STRING  NOT NULL COMMENT 'max_return | min_risk | balanced (the knee of the frontier)',
            allocation_id          STRING  NOT NULL,
            mean_revenue           DOUBLE,
            std_revenue            DOUBLE,
            var_95                 DOUBLE,
            cvar_95                DOUBLE,
            expected_roas          DOUBLE,
            n_efficient            INT     COMMENT 'How many points the frontier had -- context for whether "balanced" was meaningful',
            balanced_is_degenerate BOOLEAN COMMENT 'TRUE when the frontier had fewer than 3 points, so the knee is an endpoint rather than a considered interior choice',
            nearest_neighbour_gap  DOUBLE  COMMENT 'Distance along the return axis from this pick to the closest other efficient point',
            ordering_unresolved    BOOLEAN COMMENT 'TRUE when that gap is under 2x the measured CRN noise, i.e. this pick is not distinguishable from its neighbour by this sweep. A frontier can be non-degenerate and still not ordered.',
            extrapolation_floor_applied BOOLEAN COMMENT 'TRUE when at least one funded channel in this allocation fell below its per-channel evidence floor, so its CPC was priced AT the floor rather than extrapolated below the observed data. Respect this the same way as ordering_unresolved: the number is defensible, but it sits at the edge of what bronze can support.'
        )
        USING DELTA
        COMMENT 'Three named picks per scenario and objective pair under a fixed, stated rule. Reproducible rather than hand-chosen.'
    """),
]


def main() -> None:
    print("=" * 72)
    print("PHASE 4 -- GOLD FRONTIER TABLES")
    print("=" * 72)
    w = get_client()
    wid = resolve_warehouse_id(w)

    sql(w, wid, f"CREATE SCHEMA IF NOT EXISTS {GOLD}")
    print(f"\nschema {GOLD} ready")

    for name, ddl in DDL:
        sql(w, wid, ddl)
        print(f"  created/verified {name}")

    # CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a column
    # added later by ALTER TABLE carries no comment and the DDL above cannot
    # supply one. That is exactly what happened to the spend-floor columns.
    # Re-applying every comment on each run makes the metadata converge instead
    # of depending on whether a table happened to be created before or after a
    # column existed.
    print("\n-- re-applying column comments (idempotent) --")
    applied = 0
    for table, column, comment in COLUMN_COMMENTS:
        try:
            sql(w, wid,
                f"ALTER TABLE {table} ALTER COLUMN {column} "
                f"COMMENT '{comment.replace(chr(39), chr(39) * 2)}'")
            applied += 1
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            print(f"  WARN {table}.{column}: {str(exc)[:100]}")
    print(f"  applied {applied} of {len(COLUMN_COMMENTS)} column comments")

    print("\n-- schemas as they now exist --")
    expected = {
        TBL_SWEEP: 18,
        TBL_FRONTIER: 10,
        TBL_RECS: 14,
    }
    for name, n_expected in expected.items():
        cols, rows = sql(w, wid, f"DESCRIBE TABLE {name}")
        real = [r for r in rows if r[0] and not str(r[0]).startswith("#") and str(r[0]).strip()]
        print(f"\n{name}")
        print_table(cols, real, max_rows=20)
        check(f"{name} has {n_expected} columns", len(real) == n_expected, f"got {len(real)}")
        _c, cnt = sql(w, wid, f"SELECT COUNT(*) FROM {name}")
        print(f"  rows: {cnt[0][0]}")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"GOLD DDL FAILED -- {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("GOLD DDL PASSED -- three tables ready for the sweep to populate.")


if __name__ == "__main__":
    sys.exit(main())
