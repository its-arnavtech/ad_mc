"""
Step 5 -- derive the two assumption tables FROM the bronze history table.

  ad_mc_poc.bronze.channel_assumptions
  ad_mc_poc.bronze.channel_correlation_matrix

Both are recomputed from `channel_performance_history`, which stays the single
source of truth. Re-running this script is safe (CREATE OR REPLACE).

Assumptions baked in here -- change them in one place if you disagree:

  * Correlation window = the FULL history (all 730 days). No rolling window,
    no recency weighting. Worth revisiting in a later phase if you want the
    frontier to react to recent regime changes.
  * Correlation is Pearson on DAILY REVENUE, pairing channels on `date`.
    Produces the full symmetric matrix including the 1.0 diagonal, which is
    the form a Cholesky decomposition wants later.
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

from _common import (
    TBL_ASSUMPTIONS,
    TBL_CORRELATION,
    TBL_HISTORY,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)

SKEW_THRESHOLD = 0.5


def main() -> None:
    w = get_client()
    warehouse_id = resolve_warehouse_id(w)

    # ---- channel_assumptions ------------------------------------------------
    print(f"== Building {TBL_ASSUMPTIONS} ==")
    sql(
        w,
        warehouse_id,
        f"""
        CREATE OR REPLACE TABLE {TBL_ASSUMPTIONS}
        COMMENT 'Bronze: per-channel distribution parameters derived from channel_performance_history.'
        AS
        WITH daily AS (
            SELECT
                channel_id,
                CASE WHEN impressions > 0 THEN clicks      / impressions END AS ctr,
                CASE WHEN clicks      > 0 THEN spend       / clicks      END AS cpc,
                CASE WHEN clicks      > 0 THEN conversions / clicks      END AS cvr,
                CASE WHEN conversions > 0 THEN revenue     / conversions END AS revenue_per_conversion
            FROM {TBL_HISTORY}
        )
        SELECT
            channel_id,
            INITCAP(REPLACE(channel_id, '_', ' '))          AS channel_name,
            AVG(ctr)                                        AS mean_ctr,
            STDDEV_SAMP(ctr)                                AS std_ctr,
            AVG(cpc)                                        AS mean_cpc,
            STDDEV_SAMP(cpc)                                AS std_cpc,
            AVG(cvr)                                        AS mean_cvr,
            STDDEV_SAMP(cvr)                                AS std_cvr,
            AVG(revenue_per_conversion)                     AS mean_revenue_per_conversion,
            STDDEV_SAMP(revenue_per_conversion)             AS std_revenue_per_conversion,
            CASE
                WHEN SKEWNESS(revenue_per_conversion) > {SKEW_THRESHOLD} THEN 'lognormal'
                ELSE 'normal'
            END                                             AS distribution_type
        FROM daily
        GROUP BY channel_id
        ORDER BY channel_id
        """,
    )
    cols, rows = sql(w, warehouse_id, f"SELECT * FROM {TBL_ASSUMPTIONS} ORDER BY channel_id")
    print_table(cols, rows)

    # ---- channel_correlation_matrix ----------------------------------------
    print(f"\n== Building {TBL_CORRELATION} ==")
    sql(
        w,
        warehouse_id,
        f"""
        CREATE OR REPLACE TABLE {TBL_CORRELATION}
        COMMENT 'Bronze: Pearson correlation of daily revenue between every channel pair, full history window.'
        AS
        SELECT
            a.channel_id            AS channel_id_a,
            b.channel_id            AS channel_id_b,
            CORR(a.revenue, b.revenue) AS correlation_coefficient
        FROM {TBL_HISTORY} a
        JOIN {TBL_HISTORY} b
          ON a.date = b.date
        GROUP BY a.channel_id, b.channel_id
        ORDER BY channel_id_a, channel_id_b
        """,
    )
    cols, rows = sql(
        w,
        warehouse_id,
        f"SELECT * FROM {TBL_CORRELATION} ORDER BY channel_id_a, channel_id_b",
    )
    print_table(cols, rows, max_rows=30)

    print("\nAssumption tables complete.")


if __name__ == "__main__":
    main()
