"""
Step 4 -- load data/raw/channel_performance_history.csv into the managed Delta
table `ad_mc_poc.bronze.channel_performance_history`.

Two hops:
  1. upload the CSV into the UC volume `ad_mc_poc.bronze.landing` (Files API)
  2. CREATE OR REPLACE TABLE ... AS SELECT ... FROM read_files(...)

The schema is declared explicitly rather than inferred, so the Delta table's
types are deterministic and don't drift if the CSV is regenerated.

    python databricks/02_load_bronze.py
"""

from _common import (
    LANDING_CSV,
    LOCAL_CSV,
    TBL_HISTORY,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)

# Declared once here; read_files() applies it and the CTAS inherits it.
CSV_SCHEMA = (
    "date DATE, "
    "channel_id STRING, "
    "impressions BIGINT, "
    "clicks BIGINT, "
    "conversions BIGINT, "
    "spend DOUBLE, "
    "revenue DOUBLE"
)


def main() -> None:
    if not LOCAL_CSV.exists():
        raise SystemExit(
            f"ERROR: {LOCAL_CSV} not found.\n"
            "Generate it first:\n"
            "  python data_generation/generate_channel_data.py "
            "--start-date 2024-01-01 --days 730 --seed 42"
        )

    w = get_client()
    warehouse_id = resolve_warehouse_id(w)

    size_kb = LOCAL_CSV.stat().st_size / 1024
    print(f"== Uploading {LOCAL_CSV.name} ({size_kb:,.0f} KB) -> {LANDING_CSV} ==")
    with open(LOCAL_CSV, "rb") as fh:
        w.files.upload(LANDING_CSV, fh, overwrite=True)
    print("  upload ok")

    print(f"\n== Building {TBL_HISTORY} ==")
    sql(
        w,
        warehouse_id,
        f"""
        CREATE OR REPLACE TABLE {TBL_HISTORY}
        COMMENT 'Bronze: daily synthetic per-channel ad performance. Source of truth for Phase 1.'
        AS
        SELECT
            date,
            channel_id,
            impressions,
            clicks,
            conversions,
            spend,
            revenue
        FROM read_files(
            '{LANDING_CSV}',
            format      => 'csv',
            header      => true,
            schema      => '{CSV_SCHEMA}',
            mode        => 'FAILFAST'
        )
        """,
    )
    print("  table created")

    print("\n== Result ==")
    cols, rows = sql(
        w,
        warehouse_id,
        f"""
        SELECT
            COUNT(*)                     AS row_count,
            COUNT(DISTINCT channel_id)   AS channels,
            MIN(date)                    AS first_date,
            MAX(date)                    AS last_date
        FROM {TBL_HISTORY}
        """,
    )
    print_table(cols, rows)

    cols, rows = sql(w, warehouse_id, f"DESCRIBE TABLE {TBL_HISTORY}")
    print("\n== Schema ==")
    print_table(cols, rows)

    print("\nBronze history load complete.")


if __name__ == "__main__":
    main()
