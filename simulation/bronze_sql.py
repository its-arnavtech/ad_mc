"""The bronze rebuild statements, in one place, for two callers.

WHY THIS LIVES IN `simulation/` AND NOT `databricks/`
-----------------------------------------------------
The statements themselves are Phase 1's -- they were inline f-strings inside
`databricks/03_build_assumptions.py`, which runs locally against a SQL warehouse.
Phase 5's `bronze_refresh` Workflow task has to run the SAME statements ON
Databricks through `spark.sql`, and the only code-distribution mechanism this
project has for the job compute is the `ad_mc_sim` wheel, which is built from
`simulation/`. Duplicating ~60 lines of aggregation SQL into a notebook would
mean two definitions of what `mean_cpc` is; the first time they drifted, every
downstream number would move and nothing would say why.

So the text moved here VERBATIM and both callers import it:

    databricks/03_build_assumptions.py            (local, SQL warehouse)
    simulation/phase5_tasks.py::run_bronze_refresh (Workflow task, spark.sql)

Nothing here is new SQL. The assumptions this SQL bakes in -- full-history
correlation window, pairwise deletion of undefined CVR days, mean-of-ratios,
sample standard deviations, skewness-chosen distribution_type -- are documented
at length in `03_build_assumptions.py` and are unchanged.

CREATE OR REPLACE, so re-running is idempotent on unchanged source data. Note
what that does and does not promise: it promises the same DEFINITION, not
bit-identical floats, because a distributed AVG/STDDEV sums in partition order
and two different compute shapes can differ in the last ulp. Phase 5 measures
that rather than assuming it away.
"""

from __future__ import annotations

# Skewness above which revenue-per-conversion is labelled 'lognormal'.
SKEW_THRESHOLD = 0.5


def assumptions_statement(table: str, history: str,
                          skew_threshold: float = SKEW_THRESHOLD) -> str:
    return f"""
        CREATE OR REPLACE TABLE {table}
        COMMENT 'Bronze: per-channel distribution parameters derived from channel_performance_history.'
        AS
        WITH daily AS (
            SELECT
                channel_id,
                CASE WHEN impressions > 0 THEN clicks      / impressions END AS ctr,
                CASE WHEN clicks      > 0 THEN spend       / clicks      END AS cpc,
                CASE WHEN clicks      > 0 THEN conversions / clicks      END AS cvr,
                CASE WHEN conversions > 0 THEN revenue     / conversions END AS revenue_per_conversion
            FROM {history}
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
                WHEN SKEWNESS(revenue_per_conversion) > {skew_threshold} THEN 'lognormal'
                ELSE 'normal'
            END                                             AS distribution_type
        FROM daily
        GROUP BY channel_id
        ORDER BY channel_id
        """


def revenue_correlation_statement(table: str, history: str) -> str:
    return f"""
        CREATE OR REPLACE TABLE {table}
        COMMENT 'Bronze: Pearson correlation of daily revenue between every channel pair, full history window.'
        AS
        SELECT
            a.channel_id            AS channel_id_a,
            b.channel_id            AS channel_id_b,
            CORR(a.revenue, b.revenue) AS correlation_coefficient
        FROM {history} a
        JOIN {history} b
          ON a.date = b.date
        GROUP BY a.channel_id, b.channel_id
        ORDER BY channel_id_a, channel_id_b
        """


def cvr_correlation_statement(table: str, history: str) -> str:
    return f"""
        CREATE OR REPLACE TABLE {table}
        COMMENT 'Bronze: Pearson correlation of daily CVR (conversions/clicks) between every channel pair, full history window. Days with zero clicks are excluded pairwise. This is the matrix the Monte Carlo Cholesky step uses.'
        AS
        WITH daily_cvr AS (
            SELECT
                date,
                channel_id,
                CASE WHEN clicks > 0 THEN conversions / clicks END AS cvr
            FROM {history}
        )
        SELECT
            a.channel_id              AS channel_id_a,
            b.channel_id              AS channel_id_b,
            CORR(a.cvr, b.cvr)        AS correlation_coefficient
        FROM daily_cvr a
        JOIN daily_cvr b
          ON a.date = b.date
        WHERE a.cvr IS NOT NULL
          AND b.cvr IS NOT NULL
        GROUP BY a.channel_id, b.channel_id
        ORDER BY channel_id_a, channel_id_b
        """
