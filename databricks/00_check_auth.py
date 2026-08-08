"""
Step 2 -- verify Databricks auth and report what kind of workspace this is.

Run this FIRST. It is read-only: it creates nothing and changes nothing.

    python databricks/00_check_auth.py
"""

from _common import detect_metastore, get_client, resolve_warehouse_id, sql


def main() -> None:
    print("== Databricks auth check ==")
    w = get_client()

    print(f"  host          : {w.config.host}")
    print(f"  auth type     : {w.config.auth_type}")

    try:
        me = w.current_user.me()
        print(f"  authenticated : {me.user_name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: host reachable but the token was rejected: {exc}")
        raise SystemExit(2) from exc

    metastore = detect_metastore(w)
    print(f"  metastore     : {metastore}")
    if metastore == "unity-catalog":
        current = w.metastores.current()
        print(f"  metastore id  : {current.metastore_id}")
        print("  -> three-level naming (ad_mc_poc.bronze.<table>) is supported.")
    else:
        print(
            "  -> LEGACY HIVE METASTORE. CREATE CATALOG is not supported here, so\n"
            "     `ad_mc_poc.bronze.<table>` cannot be created as specified.\n"
            "     Stop and decide on a naming fallback before continuing."
        )

    print("\n== SQL warehouses ==")
    warehouses = list(w.warehouses.list())
    if not warehouses:
        print("  none found -- see resolve_warehouse_id() for what this implies.")
    for wh in warehouses:
        state = getattr(wh.state, "value", wh.state)
        serverless = getattr(wh, "enable_serverless_compute", None)
        print(f"  - {wh.name} (id={wh.id}, state={state}, serverless={serverless})")

    if warehouses:
        print("\n== Connectivity smoke test ==")
        warehouse_id = resolve_warehouse_id(w)
        _, rows = sql(w, warehouse_id, "SELECT current_catalog(), current_user(), version()")
        print(f"  catalog={rows[0][0]}  user={rows[0][1]}  dbr/sql version={rows[0][2]}")

    print("\nAuth check complete.")


if __name__ == "__main__":
    main()
