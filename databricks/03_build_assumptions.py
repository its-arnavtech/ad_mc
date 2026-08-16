"""
Step 5 -- derive the assumption tables FROM the bronze history table.

  ad_mc_poc.bronze.channel_assumptions
  ad_mc_poc.bronze.channel_correlation_matrix       (revenue-based, diagnostic)
  ad_mc_poc.bronze.channel_cvr_correlation_matrix   (CVR-based, drives the sim)

Both are recomputed from `channel_performance_history`, which stays the single
source of truth. Re-running this script is safe (CREATE OR REPLACE).

Assumptions baked in here -- change them in one place if you disagree:

  * Correlation window = the FULL history (all 730 days). No rolling window,
    no recency weighting. Worth revisiting in a later phase if you want the
    frontier to react to recent regime changes.
  * TWO correlation matrices are built, both Pearson, both pairing channels on
    `date`, both storing the full symmetric matrix including the 1.0 diagonal
    (the form a Cholesky decomposition wants):
      - `channel_correlation_matrix` on DAILY REVENUE. Retained as a
        diagnostic. Revenue = clicks x CVR x revenue-per-conversion, so this
        bundles CTR, CVR and revenue-per-conversion co-movement together and
        is NOT a clean read on any one of them.
      - `channel_cvr_correlation_matrix` on DAILY CVR. This is the one the
        simulation uses, because CVR is the only quantity correlated across
        channels in the Cholesky step.
  * Days with zero clicks leave CVR undefined. Those channel-days are EXCLUDED
    pairwise (`WHERE a.cvr IS NOT NULL AND b.cvr IS NOT NULL`) rather than
    imputed -- imputing a rate would invent co-movement that isn't in the data.
    The script prints the observation count per pair so any exclusion is
    visible. Note that pairwise deletion can in principle yield a non-positive
    -definite matrix; 04_validate_bronze.py checks for that explicitly.
  * `channel_name` is derived by title-casing `channel_id` ("paid_search" ->
    "Paid Search"). The CSV carries no display-name column; if you have real
    names, this is the line to change.
  * `distribution_type` describes the REVENUE-PER-CONVERSION marginal. It is
    chosen from observed skewness rather than hardcoded: skew > 0.5 ->
    'lognormal', else 'normal'. NOTE: as of Phase 2 the row carries three
    distributions (CPC, CVR, revenue-per-conversion) but only this one
    type column. Phase 2's families are specified by the simulation spec,
    not read from here, so the column stays descriptive-only. If a later
    phase wants to drive families from data, this needs splitting into
    per-quantity type columns.
  * `mean_cpc` / `std_cpc` are derived from daily `spend / clicks`. CPC is
    what the forward simulation needs: it maps a dollar allocation to clicks.
    CTR maps impressions -> clicks, which is the historical generator's
    direction, and is retained as a diagnostic only.
  * Daily ratios are computed per day and then averaged across days
    (mean-of-ratios, not ratio-of-totals), so each day is weighted equally.
    Rows with a zero denominator are excluded from the relevant statistic.
  * std_* use the SAMPLE standard deviation (stddev_samp).

    python databricks/03_build_assumptions.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "simulation"))

from _common import (  # noqa: E402
    TBL_ASSUMPTIONS,
    TBL_CORRELATION,
    TBL_CVR_CORRELATION,
    TBL_HISTORY,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)

# The three statements below moved to simulation/bronze_sql.py VERBATIM so the
# Phase 5 `bronze_refresh` Workflow task can run the same text through spark.sql
# instead of keeping a second copy. See that module's docstring.
from bronze_sql import (  # noqa: E402
    SKEW_THRESHOLD,
    assumptions_statement,
    cvr_correlation_statement,
    revenue_correlation_statement,
)


def main() -> None:
    w = get_client()
    warehouse_id = resolve_warehouse_id(w)

    # ---- channel_assumptions ------------------------------------------------
    print(f"== Building {TBL_ASSUMPTIONS} ==")
    sql(w, warehouse_id, assumptions_statement(TBL_ASSUMPTIONS, TBL_HISTORY, SKEW_THRESHOLD))
    cols, rows = sql(w, warehouse_id, f"SELECT * FROM {TBL_ASSUMPTIONS} ORDER BY channel_id")
    print_table(cols, rows)

    # ---- channel_correlation_matrix ----------------------------------------
    print(f"\n== Building {TBL_CORRELATION} ==")
    sql(w, warehouse_id, revenue_correlation_statement(TBL_CORRELATION, TBL_HISTORY))
    cols, rows = sql(
        w,
        warehouse_id,
        f"SELECT * FROM {TBL_CORRELATION} ORDER BY channel_id_a, channel_id_b",
    )
    print_table(cols, rows, max_rows=30)

    # ---- channel_cvr_correlation_matrix ------------------------------------
    print(f"\n== Building {TBL_CVR_CORRELATION} ==")

    undefined = sql(w, warehouse_id, f"""
        SELECT COUNT(*) FROM {TBL_HISTORY} WHERE clicks IS NULL OR clicks = 0
    """)[1][0][0]
    print(f"  channel-days with undefined CVR (zero/null clicks): {undefined}")
    print("  -> excluded pairwise from the correlation; not imputed")

    sql(w, warehouse_id, cvr_correlation_statement(TBL_CVR_CORRELATION, TBL_HISTORY))

    cols, rows = sql(
        w,
        warehouse_id,
        f"SELECT * FROM {TBL_CVR_CORRELATION} ORDER BY channel_id_a, channel_id_b",
    )
    print_table(cols, rows, max_rows=30)

    print("\n  observation counts per pair (should equal the number of shared days):")
    cols, rows = sql(w, warehouse_id, f"""
        WITH daily_cvr AS (
            SELECT date, channel_id,
                   CASE WHEN clicks > 0 THEN conversions / clicks END AS cvr
            FROM {TBL_HISTORY}
        )
        SELECT MIN(n) AS min_obs, MAX(n) AS max_obs FROM (
            SELECT a.channel_id, b.channel_id, COUNT(*) AS n
            FROM daily_cvr a JOIN daily_cvr b ON a.date = b.date
            WHERE a.cvr IS NOT NULL AND b.cvr IS NOT NULL
            GROUP BY a.channel_id, b.channel_id)
    """)
    print_table(cols, rows)

    print("\n  side-by-side vs the revenue-based matrix (off-diagonal pairs):")
    cols, rows = sql(w, warehouse_id, f"""
        SELECT c.channel_id_a, c.channel_id_b,
               ROUND(r.correlation_coefficient, 4) AS revenue_based,
               ROUND(c.correlation_coefficient, 4) AS cvr_based,
               ROUND(c.correlation_coefficient - r.correlation_coefficient, 4) AS delta
        FROM {TBL_CVR_CORRELATION} c
        JOIN {TBL_CORRELATION} r
          ON c.channel_id_a = r.channel_id_a AND c.channel_id_b = r.channel_id_b
        WHERE c.channel_id_a < c.channel_id_b
        ORDER BY c.channel_id_a, c.channel_id_b
    """)
    print_table(cols, rows)

    print("\nAssumption tables complete.")


if __name__ == "__main__":
    main()
