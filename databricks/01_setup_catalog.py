"""
Step 3 -- create the `ad_mc_poc` catalog and its bronze / silver / gold schemas.

Idempotent (CREATE IF NOT EXISTS throughout). Also creates the `landing`
volume that step 4 uploads the raw CSV into.

    python databricks/01_setup_catalog.py
"""

from _common import (
    CATALOG,
    LANDING_VOLUME,
    SCHEMAS,
    detect_metastore,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)


def main() -> None:
    w = get_client()

    metastore = detect_metastore(w)
    print(f"== Metastore: {metastore} ==")
    if metastore != "unity-catalog":
        print(
            "ABORTING: this workspace is on the legacy Hive metastore.\n"
            "`CREATE CATALOG ad_mc_poc` is a Unity Catalog operation and will fail here.\n"
            "The Phase 1 spec asks for ad_mc_poc.bronze.<table>, which needs UC.\n"
            "Options: (a) enable UC on this workspace, or (b) fall back to\n"
            "hive_metastore databases named ad_mc_poc_bronze / _silver / _gold.\n"
            "That is a naming change to the spec, so it is your call -- not mine."
        )
        raise SystemExit(4)

    warehouse_id = resolve_warehouse_id(w)

    print(f"\n== Creating catalog `{CATALOG}` ==")
    sql(w, warehouse_id, f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    sql(
        w,
        warehouse_id,
        f"COMMENT ON CATALOG {CATALOG} IS "
        "'Ad Revenue Monte Carlo POC -- portfolio-style VaR/efficient-frontier "
        "simulation over advertising channels.'",
    )
    print(f"  ok: {CATALOG}")

    print("\n== Creating schemas ==")
    for schema in SCHEMAS:
        sql(w, warehouse_id, f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
        print(f"  ok: {CATALOG}.{schema}")

    print(f"\n== Creating landing volume `{LANDING_VOLUME}` ==")
    sql(w, warehouse_id, f"CREATE VOLUME IF NOT EXISTS {LANDING_VOLUME}")
    print(f"  ok: {LANDING_VOLUME}")

    print("\n== Verifying ==")
    cols, rows = sql(w, warehouse_id, f"SHOW SCHEMAS IN {CATALOG}")
    print_table(cols, rows)

    print("\nCatalog setup complete.")


if __name__ == "__main__":
    main()
