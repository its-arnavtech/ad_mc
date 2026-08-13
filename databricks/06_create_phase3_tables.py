"""
Phase 3, step 2 -- create the three Delta tables the distributed simulation
reads and writes.

    ad_mc_poc.bronze.scenario_definitions           macro scenario overlays
    ad_mc_poc.silver.allocation_candidates          the budget splits to test
    ad_mc_poc.silver.simulated_allocation_outcomes  one row per simulated path

WHY THE DDL LIVES HERE AND NOT INSIDE THE SPARK JOB
---------------------------------------------------
The Phase 3 job writes ~800k rows through `applyInPandas`. If it created its
own tables with `saveAsTable`, the schema would be whatever the UDF happened to
emit on that run, and a bug in the UDF would surface as silent schema drift
instead of a failed write. Declaring the schema up front makes the table the
contract and the job's output checked against it. It also keeps this step on
the SQL warehouse, so the tables exist before the cluster is even up.

CREATE TABLE IF NOT EXISTS, NOT CREATE OR REPLACE -- deliberate
---------------------------------------------------------------
Phase 1's scripts (02/03) use CREATE OR REPLACE because those tables are
*derived*: re-running recomputes them from `channel_performance_history` in
seconds, so replacing them loses nothing. These three are not derived.
`simulated_allocation_outcomes` costs a cluster and a full Monte Carlo run to
repopulate; the other two are hand-authored reference data owned by the
simulation modules. So re-running this script must never destroy data:

  * every table is CREATE TABLE IF NOT EXISTS
  * table and column COMMENTs are re-applied on every run, so metadata still
    converges on the spec below even when the table already existed
  * the script then DESCRIBEs each table and FAILS LOUDLY if the live schema
    has drifted from the spec, rather than pretending everything is fine
  * `--replace` is the explicit, opt-in way to rebuild after a drift. It drops
    the tables, so it will not happen by accident.

PARTITIONING / LIQUID CLUSTERING -- deliberately NEITHER
--------------------------------------------------------
`simulated_allocation_outcomes` is the only table big enough to raise the
question: ~20 allocations x 4 scenarios x 10,000 paths ~= 800k rows of two
short strings and four doubles. That is on the order of 10 MB as Parquet --
well under Delta's ~128 MB target file size, i.e. the whole table is one file.

  * Hive-style PARTITIONED BY (allocation_id, scenario_id) would cut that into
    80 directories of ~10k rows each. That is textbook over-partitioning: 80
    tiny files, worse compression and dictionary encoding, file-listing
    overhead on every read, and a slower write. It would make Phase 4 slower,
    not faster.
  * Liquid clustering (CLUSTER BY) avoids the directory explosion, but it earns
    its keep through data skipping, and data skipping only pays on *selective*
    reads. Phase 4's actual query is a full-table aggregate --
    GROUP BY (allocation_id, scenario_id) over every row -- so there is nothing
    to skip. Adding CLUSTER BY here would be cargo cult.

So: no partitioning, no clustering. Revisit if the path count per cell grows by
~2 orders of magnitude (say 1M paths -> 80M rows, multi-GB), at which point
CLUSTER BY (allocation_id, scenario_id) becomes the right tool -- still not
Hive partitioning, whose 80 fixed directories would stay unbalanced.

What IS worth setting at this size is `delta.autoOptimize.optimizeWrite`. The
real small-file risk here is not layout, it is the writer: an applyInPandas
over ~80 groups emits at least one file per task, so a naive append lands ~80
files of ~100 KB. Optimized writes coalesce those at write time. autoCompact is
deliberately NOT set -- it re-compacts after each commit, which buys nothing
for a single bulk append and just adds a job.

CONTRACT FOR THE WRITER (the applyInPandas job)
------------------------------------------------
  * `path_number` is INT, and numpy defaults to int64 on Linux executors. The
    UDF's declared output schema should say `path_number int` and the returned
    pandas column should be int32, so nothing depends on an implicit narrowing
    cast at insert time.
  * The four key columns are NOT NULL, so a null allocation_id / scenario_id /
    channel_id / path_number fails the write instead of landing unjoinable rows.
  * Append, do not overwrite, unless you mean to discard earlier runs -- there
    is no run_id column, so re-running the same cell appends duplicates rather
    than replacing them. That is a real gap if Phase 3 ends up re-running
    individual cells; flagged rather than silently designed around.

    python databricks/06_create_phase3_tables.py
    python databricks/06_create_phase3_tables.py --replace   # destructive
"""

from __future__ import annotations

import argparse
import sys

from _common import (
    CATALOG,
    TBL_ALLOCATION_CANDIDATES,
    TBL_SCENARIOS,
    TBL_SIMULATED_OUTCOMES,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)

# --- Schema spec -------------------------------------------------------------
# (column_name, sql_type, not_null, comment)
#
# NOT NULL is applied to key columns only. A null allocation_id / scenario_id /
# channel_id / path_number is never a legitimate value -- it is a bug in the
# writer -- and Delta enforces NOT NULL, so the write fails at the source
# instead of leaving unjoinable rows for a data-quality check to find later.
# Measure columns stay nullable so a genuinely undefined measure (e.g. ROAS on
# zero spend) can be recorded as NULL rather than as a misleading 0.0.

SCENARIO_COLUMNS = [
    ("scenario_id", "STRING", True,
     "Stable key for the macro scenario, e.g. 'normal'. Join target for "
     "silver.simulated_allocation_outcomes.scenario_id."),
    ("scenario_name", "STRING", False,
     "Human-readable display name for reporting."),
    ("description", "STRING", False,
     "One line describing the macro conditions this scenario represents."),
    ("cvr_multiplier", "DOUBLE", False,
     "Multiplicative shock applied to each channel's mean conversion rate "
     "before sampling. 1.0 = unshocked."),
    ("cpc_multiplier", "DOUBLE", False,
     "Multiplicative shock applied to each channel's mean cost per click "
     "before sampling. Above 1.0 means clicks get more expensive."),
    ("revenue_multiplier", "DOUBLE", False,
     "Multiplicative shock applied to each channel's mean revenue per "
     "conversion before sampling. 1.0 = unshocked."),
    ("rationale", "STRING", False,
     "Why these multipliers were chosen. Kept next to the numbers on purpose, "
     "so a reader does not have to go find the modelling notebook."),
]

SCENARIO_COMMENT = (
    "Bronze: macro scenario overlays for the Phase 3 distributed simulation. "
    "One row per scenario; the three multipliers shock the Phase 1 per-channel "
    "assumptions (mean CVR / mean CPC / mean revenue per conversion) before "
    "sampling. Hand-authored reference data owned by simulation/scenarios.py -- "
    "unlike the other bronze tables it is NOT derived from "
    "channel_performance_history."
)

ALLOCATION_COLUMNS = [
    ("allocation_id", "STRING", True,
     "Stable key for one candidate budget split, e.g. 'even_split'. Grain of "
     "this table is (allocation_id, channel_id)."),
    ("channel_id", "STRING", True,
     "Channel receiving the spend. Joins to bronze.channel_assumptions.channel_id."),
    ("spend_pct", "DOUBLE", False,
     "Share of total_budget assigned to this channel, as a FRACTION in [0, 1] "
     "-- not 0-100. Sums to 1.0 across the rows of one allocation_id."),
    ("spend_dollars", "DOUBLE", False,
     "spend_pct * total_budget. Stored rather than recomputed so the Spark job "
     "reads dollars directly and there is exactly one definition of the split."),
    ("total_budget", "DOUBLE", False,
     "Total budget this allocation divides up, repeated on every row of the "
     "allocation. Denormalised on purpose: it is constant within an "
     "allocation_id and it keeps the simulation's read a single flat scan."),
]

ALLOCATION_COMMENT = (
    "Silver: the candidate budget allocations Phase 3 simulates. Long form -- "
    "one row per (allocation_id, channel_id), so the number of channels is not "
    "baked into the schema. Produced by simulation/allocations.py."
)

OUTCOME_COLUMNS = [
    ("allocation_id", "STRING", True,
     "Allocation that was simulated. Joins to silver.allocation_candidates.allocation_id."),
    ("scenario_id", "STRING", True,
     "Scenario that was simulated. Joins to bronze.scenario_definitions.scenario_id."),
    ("path_number", "INT", True,
     "0-based index of the Monte Carlo path within this (allocation_id, "
     "scenario_id) cell. INT, not BIGINT: the domain is bounded by the path "
     "count (10,000), four orders of magnitude below the INT limit."),
    ("total_simulated_revenue", "DOUBLE", False,
     "Revenue on this path, summed across every channel in the allocation. "
     "Per-channel detail is deliberately NOT stored -- this table is "
     "portfolio-level, which is the grain Phase 4's risk metrics need."),
    ("total_spend", "DOUBLE", False,
     "Total dollars spent on this path. Under the current model spend is "
     "deterministic and equals the allocation's total_budget; it is stored per "
     "path anyway so a later phase can make spend stochastic without a schema "
     "change."),
    ("roas", "DOUBLE", False,
     "total_simulated_revenue / total_spend for this path. Stored rather than "
     "derived so Phase 4 can aggregate it without repeating the division, and "
     "so the definition of ROAS lives in one place."),
]

OUTCOME_COMMENT = (
    "Silver: one row per simulated Monte Carlo path. Grain is (allocation_id, "
    "scenario_id, path_number) -- roughly 20 allocations x 4 scenarios x 10,000 "
    "paths = ~800k rows. Deliberately NOT partitioned and NOT liquid-clustered; "
    "the reasoning is in databricks/06_create_phase3_tables.py. Written by the "
    "Phase 3 applyInPandas job, read by Phase 4's efficient-frontier sweep."
)

# Optimized writes: the applyInPandas job emits at least one file per group
# (~80 groups), so without this a bulk append lands ~80 files of ~100 KB.
# autoCompact is intentionally omitted -- see the module docstring.
OUTCOME_TBLPROPERTIES = {"delta.autoOptimize.optimizeWrite": "true"}

TABLES = [
    {
        "name": TBL_SCENARIOS,
        "columns": SCENARIO_COLUMNS,
        "comment": SCENARIO_COMMENT,
        "tblproperties": None,
    },
    {
        "name": TBL_ALLOCATION_CANDIDATES,
        "columns": ALLOCATION_COLUMNS,
        "comment": ALLOCATION_COMMENT,
        "tblproperties": None,
    },
    {
        "name": TBL_SIMULATED_OUTCOMES,
        "columns": OUTCOME_COLUMNS,
        "comment": OUTCOME_COMMENT,
        "tblproperties": OUTCOME_TBLPROPERTIES,
    },
]

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}{(' -- ' + detail) if detail else ''}")


def _esc(text: str) -> str:
    """Escape a SQL string literal. Comments are prose, so apostrophes happen."""
    return text.replace("'", "''")


def create_sql(spec: dict) -> str:
    cols = ",\n".join(
        f"    {name} {dtype}{' NOT NULL' if not_null else ''} COMMENT '{_esc(comment)}'"
        for name, dtype, not_null, comment in spec["columns"]
    )
    props = ""
    if spec["tblproperties"]:
        rendered = ",\n        ".join(
            f"'{k}' = '{v}'" for k, v in spec["tblproperties"].items()
        )
        props = f"\nTBLPROPERTIES (\n        {rendered}\n)"
    return (
        f"CREATE TABLE IF NOT EXISTS {spec['name']} (\n{cols}\n)\n"
        f"USING DELTA\n"
        f"COMMENT '{_esc(spec['comment'])}'{props}"
    )


def reapply_comments(w, wid, spec: dict) -> None:
    """
    Re-state the table and column comments on every run.

    CREATE TABLE IF NOT EXISTS is a no-op on an existing table, which means
    edits to the comments above would otherwise never reach a table that
    already exists. Types and nullability cannot be reconciled this way -- that
    is what the drift check below is for.
    """
    sql(w, wid, f"COMMENT ON TABLE {spec['name']} IS '{_esc(spec['comment'])}'")
    for name, _dtype, _not_null, comment in spec["columns"]:
        sql(
            w, wid,
            f"ALTER TABLE {spec['name']} ALTER COLUMN {name} COMMENT '{_esc(comment)}'",
        )


def describe(w, wid, table: str):
    """DESCRIBE TABLE, minus the trailing metadata block."""
    cols, rows = sql(w, wid, f"DESCRIBE TABLE {table}")
    real = []
    for r in rows:
        name = (r[0] or "").strip()
        if not name or name.startswith("#"):
            break
        real.append(r)
    return cols, real


def check_schema(w, wid, spec: dict) -> None:
    """Fail loudly if the live schema does not match the spec in this file."""
    table = spec["name"]
    _cols, rows = describe(w, wid, table)
    live = [((r[0] or "").strip(), (r[1] or "").strip().lower()) for r in rows]
    want = [(name, dtype.lower()) for name, dtype, _nn, _c in spec["columns"]]

    if live == want:
        check(f"{table} schema matches spec", True, f"{len(want)} columns")
    else:
        detail = f"expected {want} but found {live}"
        check(f"{table} schema matches spec", False, detail)
        print("      -> fix with: python databricks/06_create_phase3_tables.py --replace")
        return

    # DESCRIBE does not show nullability, so read it from information_schema.
    _c, nullrows = sql(
        w, wid,
        f"""
        SELECT column_name, is_nullable
        FROM {CATALOG}.information_schema.columns
        WHERE table_schema = '{table.split('.')[1]}'
          AND table_name   = '{table.split('.')[2]}'
        """,
    )
    live_nn = {name for name, is_nullable in nullrows if str(is_nullable).upper() == "NO"}
    want_nn = {name for name, _d, not_null, _c in spec["columns"] if not_null}
    check(
        f"{table} NOT NULL columns match spec",
        live_nn == want_nn,
        f"expected {sorted(want_nn)}, found {sorted(live_nn)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="DROP and recreate the three tables. Destroys any simulation "
             "output already written. Off by default.",
    )
    args = parser.parse_args()

    w = get_client()
    wid = resolve_warehouse_id(w)

    # Pre-flight: the schemas must already exist (01_setup_catalog.py makes them).
    _c, schema_rows = sql(w, wid, f"SHOW SCHEMAS IN {CATALOG}")
    have = {(r[0] or "").strip() for r in schema_rows}
    missing = {"bronze", "silver"} - have
    if missing:
        print(
            f"ERROR: schema(s) {sorted(missing)} do not exist in `{CATALOG}`.\n"
            "Run `python databricks/01_setup_catalog.py` first.",
            file=sys.stderr,
        )
        raise SystemExit(4)

    if args.replace:
        print("== --replace: DROPPING the three Phase 3 tables ==")
        print("   any simulation output already written will be lost.")
        for spec in TABLES:
            sql(w, wid, f"DROP TABLE IF EXISTS {spec['name']}")
            print(f"  dropped (if present): {spec['name']}")

    print("\n== Creating tables ==")
    for spec in TABLES:
        statement = create_sql(spec)
        print(f"\n--- {spec['name']} ---")
        print("\n".join("  " + line for line in statement.splitlines()))
        sql(w, wid, statement)
        reapply_comments(w, wid, spec)
        print("  ok (created if absent; comments re-applied)")

    print("\n== Verifying against the spec in this file ==")
    for spec in TABLES:
        check_schema(w, wid, spec)

    print("\n== Schemas as they now exist ==")
    for spec in TABLES:
        print(f"\n--- DESCRIBE TABLE {spec['name']} ---")
        cols, rows = describe(w, wid, spec["name"])
        print_table(cols, rows)

    print("\n== Table properties (simulated_allocation_outcomes) ==")
    cols, rows = sql(w, wid, f"SHOW TBLPROPERTIES {TBL_SIMULATED_OUTCOMES}")
    interesting = [
        r for r in rows
        if str(r[0]).startswith(("delta.autoOptimize", "delta.minReaderVersion",
                                 "delta.minWriterVersion", "clusteringColumns"))
    ]
    print_table(cols, interesting or rows, max_rows=40)

    print("\n== Partitioning / clustering (expected: none) ==")
    for spec in TABLES:
        _c, rows = sql(w, wid, f"DESCRIBE TABLE EXTENDED {spec['name']}")
        flags = [
            r for r in rows
            if "Partition" in str(r[0]) or "Clustering" in str(r[0]) or "Clustering" in str(r[1])
        ]
        check(
            f"{spec['name']} is unpartitioned and unclustered",
            not flags,
            "; ".join(f"{r[0]}={r[1]}" for r in flags) if flags else "",
        )

    print("\n== Row counts ==")
    for spec in TABLES:
        n = sql(w, wid, f"SELECT COUNT(*) FROM {spec['name']}")[1][0][0]
        print(f"  {spec['name']}: {n} rows")

    print()
    if FAILURES:
        print(f"FAILED -- {len(FAILURES)} check(s) did not pass:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("Phase 3 tables ready.")


if __name__ == "__main__":
    main()
