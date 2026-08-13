"""
Phase 3, step 3 -- load the two REFERENCE tables the distributed simulation
reads:

    ad_mc_poc.bronze.scenario_definitions    <- simulation/scenarios.py
    ad_mc_poc.silver.allocation_candidates   <- simulation/allocations.py

Both are small, hand-authored, fully deterministic reference data. Neither is
derived from another table, so there is nothing to recompute -- the Python
modules ARE the source of truth and this script's only job is to make the Delta
tables agree with them.

WHY SQL AND NOT SPARK
---------------------
This is 4 + 105 rows. It runs through the Statement Execution API on a SQL
warehouse, exactly like Phase 1, so it has no dependency on a cluster, on
databricks-connect, or on the Python-UDF sandbox (which does not work in this
workspace -- see simulation/run_phase3_distributed.py). Keeping the reference
load on the warehouse means the tables are populated and verifiable before the
simulation job is ever submitted.

IDEMPOTENCY -- MERGE, NOT INSERT, AND NOT DELETE+INSERT
-------------------------------------------------------
Re-running must leave the tables in exactly the state the Python modules
describe: no duplicates, no stale rows, no partial state. Three options were
considered:

  * plain INSERT            -- duplicates on every run. Rejected.
  * DELETE then INSERT      -- correct end state, but it is TWO Delta commits.
    Between them the table is empty, so a concurrent Phase 4 reader sees no
    rows rather than the old rows. Rejected.
  * MERGE with WHEN NOT MATCHED BY SOURCE THEN DELETE  -- one atomic commit
    that inserts new keys, updates changed rows, and removes rows that no
    longer exist in the Python module. This is what runs.

The third clause is the one that matters and is easy to forget: without it, an
allocation deleted from allocations.py would linger in silver forever and would
silently be simulated by a later phase.

NOTE ON SCENARIO COLUMNS -- `net_revenue_factor` IS NOT LOADED
---------------------------------------------------------------
`scenarios.scenario_rows()` returns EIGHT keys; the Delta table declared in
06_create_phase3_tables.py has SEVEN columns. The extra key is
`net_revenue_factor`, which is exactly `cvr_multiplier * revenue_multiplier /
cpc_multiplier` -- a pure function of three columns that ARE stored. It is
dropped here rather than added to the table:

  * storing a derivable column invites the two copies to disagree, and
  * widening a table whose schema 06 verifies would make 06's drift check fail.

If a future phase wants it in SQL it should be a view or a generated column,
not a hand-loaded value. This is called out loudly (the script prints it)
rather than being a silent projection.

    python databricks/07_load_phase3_reference.py
    python databricks/07_load_phase3_reference.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "simulation"))

from _common import (  # noqa: E402
    TBL_ALLOCATION_CANDIDATES,
    TBL_SCENARIOS,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)

import allocations as allocations_mod  # noqa: E402
import scenarios as scenarios_mod  # noqa: E402
from config import TOTAL_BUDGET  # noqa: E402
from data_access import load_assumptions  # noqa: E402

# Columns actually present in each Delta table, in table order. Anything a
# module emits that is not in this list is dropped, loudly.
SCENARIO_TABLE_COLUMNS = [
    "scenario_id", "scenario_name", "description",
    "cvr_multiplier", "cpc_multiplier", "revenue_multiplier", "rationale",
]
SCENARIO_KEYS = ["scenario_id"]
SCENARIO_STRING_COLUMNS = {"scenario_id", "scenario_name", "description", "rationale"}

ALLOCATION_TABLE_COLUMNS = [
    "allocation_id", "channel_id", "spend_pct", "spend_dollars", "total_budget",
]
ALLOCATION_KEYS = ["allocation_id", "channel_id"]
ALLOCATION_STRING_COLUMNS = {"allocation_id", "channel_id"}

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}{(' -- ' + detail) if detail else ''}")


def _str_lit(text: str) -> str:
    """SQL string literal. Comments and rationales are prose, so quotes happen."""
    return "'" + str(text).replace("\\", "\\\\").replace("'", "''") + "'"


def _dbl_lit(value: float) -> str:
    """Exact float64 literal.

    `repr` on a Python float is the shortest string that round-trips, and Spark
    parses a numeric literal to DECIMAL (38 significant digits) before the CAST
    correctly rounds it back to DOUBLE. So the bit pattern that lands in Delta
    is the bit pattern Python computed -- which matters for spend_pct, where the
    whole-dollar rounding in allocations.py produces values like 0.299998 that
    must reproduce spend_dollars / total_budget exactly.
    """
    return f"CAST({float(value)!r} AS DOUBLE)"


def _row_literal(row: dict, columns: list[str], string_columns: set[str]) -> str:
    parts = [
        _str_lit(row[c]) if c in string_columns else _dbl_lit(row[c])
        for c in columns
    ]
    return "(" + ", ".join(parts) + ")"


def merge_statement(table: str, rows: list[dict], columns: list[str],
                    keys: list[str], string_columns: set[str]) -> str:
    """One atomic MERGE that makes `table` exactly equal to `rows`."""
    values = ",\n        ".join(
        _row_literal(r, columns, string_columns) for r in rows
    )
    col_list = ", ".join(columns)
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    return f"""
MERGE INTO {table} AS t
USING (
    SELECT * FROM VALUES
        {values}
    AS s({col_list})
) AS s
ON {on}
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
WHEN NOT MATCHED BY SOURCE THEN DELETE
""".strip()


def project(rows: list[dict], columns: list[str], label: str) -> list[dict]:
    """Keep only the table's columns; say out loud what is being dropped."""
    if not rows:
        raise RuntimeError(f"{label}: source module produced no rows")
    extra = [k for k in rows[0] if k not in columns]
    missing = [c for c in columns if c not in rows[0]]
    if missing:
        raise RuntimeError(
            f"{label}: source module does not supply required column(s) {missing}"
        )
    if extra:
        print(f"  NOTE: dropping module-only column(s) not in the table: {extra}")
    return [{c: r[c] for c in columns} for r in rows]


def verify(w, wid, table: str, expected: list[dict], columns: list[str],
           keys: list[str], label: str) -> None:
    """Read the table back and compare it, value by value, to the Python rows."""
    order = ", ".join(keys)
    cols, rows = sql(w, wid, f"SELECT {', '.join(columns)} FROM {table} ORDER BY {order}")
    live = [dict(zip(cols, r)) for r in rows]

    check(f"{label}: row count", len(live) == len(expected),
          f"expected {len(expected)}, found {len(live)}")
    if len(live) != len(expected):
        return

    want = sorted(expected, key=lambda r: tuple(str(r[k]) for k in keys))
    got = sorted(live, key=lambda r: tuple(str(r[k]) for k in keys))

    mismatches = []
    for wrow, grow in zip(want, got):
        for c in columns:
            wv, gv = wrow[c], grow[c]
            if isinstance(wv, str):
                same = str(gv) == wv
            else:
                # exact float64 equality on purpose -- these values are written
                # from repr(), so "close enough" would hide a real precision bug
                same = gv is not None and float(gv) == float(wv)
            if not same:
                mismatches.append(
                    f"{tuple(wrow[k] for k in keys)}.{c}: expected {wv!r}, found {gv!r}"
                )
    check(f"{label}: every value round-trips exactly", not mismatches,
          "; ".join(mismatches[:3]) + (f" (+{len(mismatches) - 3} more)" if len(mismatches) > 3 else ""))

    dup_cols = ", ".join(keys)
    _c, dups = sql(
        w, wid,
        f"SELECT {dup_cols}, COUNT(*) AS n FROM {table} GROUP BY {dup_cols} HAVING COUNT(*) > 1",
    )
    check(f"{label}: no duplicate keys", not dups, f"{len(dups)} duplicated key(s)")

    null_expr = " OR ".join(f"{c} IS NULL" for c in columns)
    _c, nulls = sql(w, wid, f"SELECT COUNT(*) FROM {table} WHERE {null_expr}")
    check(f"{label}: no nulls in any column", int(nulls[0][0]) == 0,
          f"{nulls[0][0]} row(s) with a null")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the MERGE statements and the rows, write nothing.")
    args = parser.parse_args()

    w = get_client()
    wid = resolve_warehouse_id(w)

    # Channels come from bronze, not from a literal, so the allocation grid
    # follows the assumptions table if a channel is ever added.
    assumptions = load_assumptions(w, wid)
    channels = assumptions["channel_id"].tolist()
    print(f"\nchannels from {len(channels)}-row bronze.channel_assumptions: {channels}")
    print(f"total budget: ${TOTAL_BUDGET:,.0f}")

    scenario_rows = project(scenarios_mod.scenario_rows(), SCENARIO_TABLE_COLUMNS,
                            "scenario_definitions")
    allocation_rows = project(allocations_mod.allocation_rows(channels, TOTAL_BUDGET),
                              ALLOCATION_TABLE_COLUMNS, "allocation_candidates")

    n_alloc = len({r["allocation_id"] for r in allocation_rows})
    print(f"\nrows to load: {len(scenario_rows)} scenarios, "
          f"{len(allocation_rows)} allocation rows ({n_alloc} allocations x {len(channels)} channels)")

    # --- local invariants, before anything is written ---
    print("\n== Pre-flight checks on the source rows ==")
    check("scenario_definitions has the expected 4 scenarios", len(scenario_rows) == 4,
          f"found {len(scenario_rows)}")
    check("allocation_candidates is a complete rectangle",
          len(allocation_rows) == n_alloc * len(channels),
          f"{len(allocation_rows)} rows vs {n_alloc} x {len(channels)}")

    by_alloc: dict[str, list[dict]] = {}
    for r in allocation_rows:
        by_alloc.setdefault(r["allocation_id"], []).append(r)
    bad_sum = [
        a for a, rs in by_alloc.items()
        if math.fsum(r["spend_dollars"] for r in rs) != TOTAL_BUDGET
    ]
    check("every allocation's spend_dollars sums to the budget exactly", not bad_sum,
          f"offenders: {bad_sum[:3]}")
    bad_pct = [
        a for a, rs in by_alloc.items()
        if any(r["spend_dollars"] != r["spend_pct"] * r["total_budget"] for r in rs)
    ]
    check("spend_pct * total_budget reproduces spend_dollars exactly", not bad_pct,
          f"offenders: {bad_pct[:3]}")

    if args.dry_run:
        print("\n== --dry-run: statements that WOULD run ==")
        print(merge_statement(TBL_SCENARIOS, scenario_rows, SCENARIO_TABLE_COLUMNS,
                              SCENARIO_KEYS, SCENARIO_STRING_COLUMNS)[:2000])
        print("\n...\n")
        print(merge_statement(TBL_ALLOCATION_CANDIDATES, allocation_rows,
                              ALLOCATION_TABLE_COLUMNS, ALLOCATION_KEYS,
                              ALLOCATION_STRING_COLUMNS)[:2000])
        return

    print("\n== Loading (idempotent MERGE) ==")
    for table, rows, columns, keys, strings in (
        (TBL_SCENARIOS, scenario_rows, SCENARIO_TABLE_COLUMNS, SCENARIO_KEYS,
         SCENARIO_STRING_COLUMNS),
        (TBL_ALLOCATION_CANDIDATES, allocation_rows, ALLOCATION_TABLE_COLUMNS,
         ALLOCATION_KEYS, ALLOCATION_STRING_COLUMNS),
    ):
        before = int(sql(w, wid, f"SELECT COUNT(*) FROM {table}")[1][0][0])
        cols, res = sql(w, wid, merge_statement(table, rows, columns, keys, strings))
        after = int(sql(w, wid, f"SELECT COUNT(*) FROM {table}")[1][0][0])
        stats = dict(zip(cols, res[0])) if cols and res else {}
        print(f"  {table}: {before} -> {after} rows   {stats}")

    print("\n== Verifying against the Python modules ==")
    verify(w, wid, TBL_SCENARIOS, scenario_rows, SCENARIO_TABLE_COLUMNS,
           SCENARIO_KEYS, "scenario_definitions")
    verify(w, wid, TBL_ALLOCATION_CANDIDATES, allocation_rows,
           ALLOCATION_TABLE_COLUMNS, ALLOCATION_KEYS, "allocation_candidates")

    print("\n== bronze.scenario_definitions (as loaded) ==")
    cols, rows = sql(
        w, wid,
        f"""SELECT scenario_id, scenario_name, cvr_multiplier, cpc_multiplier,
                   revenue_multiplier,
                   ROUND(cvr_multiplier * revenue_multiplier / cpc_multiplier, 4)
                     AS net_revenue_factor_derived
            FROM {TBL_SCENARIOS} ORDER BY net_revenue_factor_derived DESC""",
    )
    print_table(cols, rows)

    print("\n== silver.allocation_candidates -- one row per allocation ==")
    cols, rows = sql(
        w, wid,
        f"""SELECT allocation_id,
                   COUNT(*)                        AS channels,
                   ROUND(SUM(spend_pct), 10)       AS pct_sum,
                   SUM(spend_dollars)              AS dollars,
                   ROUND(MAX(spend_pct), 6)        AS max_pct,
                   ROUND(SUM(POW(spend_pct, 2)), 4) AS hhi
            FROM {TBL_ALLOCATION_CANDIDATES}
            GROUP BY allocation_id
            ORDER BY hhi, allocation_id""",
    )
    print_table(cols, rows, max_rows=30)

    print()
    if FAILURES:
        print(f"FAILED -- {len(FAILURES)} check(s) did not pass:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("Phase 3 reference data loaded and verified.")


if __name__ == "__main__":
    main()
