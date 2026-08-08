"""
Shared helpers for the Phase 1 (bronze) Databricks scripts.

Design notes
------------
* Auth comes ONLY from the Databricks unified auth chain -- environment
  variables (DATABRICKS_HOST / DATABRICKS_TOKEN) or ~/.databrickscfg.
  Nothing in this repo ever holds a literal token.
* SQL runs through the Statement Execution API against a SQL warehouse.
  That keeps Phase 1 free of databricks-connect and free of any dependency
  on a specific cluster / DBR version.
* Every script here is manually runnable. Nothing is scheduled.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

# --- Project constants ------------------------------------------------------
CATALOG = "ad_mc_poc"
SCHEMAS = ("bronze", "silver", "gold")
BRONZE = f"{CATALOG}.bronze"

TBL_HISTORY = f"{BRONZE}.channel_performance_history"
TBL_ASSUMPTIONS = f"{BRONZE}.channel_assumptions"
TBL_CORRELATION = f"{BRONZE}.channel_correlation_matrix"

# UC volume used purely as a landing zone for the raw CSV
LANDING_VOLUME = f"{CATALOG}.bronze.landing"
LANDING_DIR = f"/Volumes/{CATALOG}/bronze/landing"
LANDING_CSV = f"{LANDING_DIR}/channel_performance_history.csv"

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CSV = REPO_ROOT / "data" / "raw" / "channel_performance_history.csv"

AUTH_HELP = """
Databricks credentials are not configured.

Set them for this shell (they are read by the SDK's unified auth chain and are
never written into the repo):

    PowerShell:
        $env:DATABRICKS_HOST  = "https://<your-workspace>.cloud.databricks.com"
        $env:DATABRICKS_TOKEN = "<personal-access-token>"

    bash:
        export DATABRICKS_HOST="https://<your-workspace>.cloud.databricks.com"
        export DATABRICKS_TOKEN="<personal-access-token>"

Optionally pin the SQL warehouse used to run statements:
        DATABRICKS_WAREHOUSE_ID=<warehouse-id>
""".strip()


def get_client() -> WorkspaceClient:
    """WorkspaceClient, or a clear message about how to configure auth."""
    try:
        w = WorkspaceClient()
        # Force a credential resolution now rather than at first API call.
        _ = w.config.host
        if not w.config.host:
            raise ValueError("no host resolved")
        return w
    except Exception as exc:  # noqa: BLE001 - we want any auth failure here
        print(f"ERROR: could not authenticate to Databricks ({type(exc).__name__}).", file=sys.stderr)
        print(f"\n{AUTH_HELP}\n", file=sys.stderr)
        raise SystemExit(2) from exc


def detect_metastore(w: WorkspaceClient) -> str:
    """
    Return "unity-catalog" or "hive-metastore".

    This matters: on Unity Catalog we can create a real `ad_mc_poc` catalog.
    On a legacy Hive metastore there is only the single `hive_metastore`
    catalog and CREATE CATALOG is unsupported, so the three-level naming in
    this project would have to change. We detect rather than assume.
    """
    try:
        current = w.metastores.current()
        if current and current.metastore_id:
            return "unity-catalog"
    except Exception:  # noqa: BLE001
        pass
    return "hive-metastore"


def resolve_warehouse_id(w: WorkspaceClient) -> str:
    """
    Pick the SQL warehouse to execute statements against.

    Order: DATABRICKS_WAREHOUSE_ID env var -> a running warehouse -> the first
    warehouse available. Fails loudly if the workspace has none, since that
    changes the approach (we'd need an all-purpose cluster instead).
    """
    explicit = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if explicit:
        return explicit

    warehouses = list(w.warehouses.list())
    if not warehouses:
        print(
            "ERROR: no SQL warehouse found in this workspace.\n"
            "These scripts execute SQL through the Statement Execution API, which\n"
            "needs a SQL warehouse. Either create one (Serverless Starter is fine)\n"
            "or tell me and I'll port these steps to a notebook that runs on an\n"
            "all-purpose cluster instead.",
            file=sys.stderr,
        )
        raise SystemExit(3)

    running = [wh for wh in warehouses if str(getattr(wh.state, "value", wh.state)).upper() == "RUNNING"]
    chosen = (running or warehouses)[0]
    print(f"  using SQL warehouse: {chosen.name} ({chosen.id})")
    return chosen.id


def sql(w: WorkspaceClient, warehouse_id: str, statement: str, timeout_s: int = 600):
    """
    Execute one SQL statement and return (columns, rows).

    Polls to a terminal state so long-running DDL/CTAS doesn't silently
    return a PENDING handle.
    """
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        wait_timeout="30s",
        catalog=None,
        schema=None,
    )

    deadline = time.time() + timeout_s
    while resp.status and resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() > deadline:
            raise TimeoutError(f"statement did not finish within {timeout_s}s:\n{statement[:300]}")
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)

    state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        err = getattr(resp.status, "error", None)
        msg = getattr(err, "message", None) or str(err)
        raise RuntimeError(f"SQL failed ({state}): {msg}\n--- statement ---\n{statement}")

    cols = []
    if resp.manifest and resp.manifest.schema and resp.manifest.schema.columns:
        cols = [c.name for c in resp.manifest.schema.columns]
    rows = (resp.result.data_array if resp.result and resp.result.data_array else []) or []
    return cols, rows


def print_table(cols, rows, max_rows: int = 50) -> None:
    """Minimal fixed-width printer so script output is readable in a terminal."""
    if not cols:
        print("  (no result set)")
        return
    shown = rows[:max_rows]
    widths = [len(c) for c in cols]
    for r in shown:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len("" if v is None else str(v)))
    line = "  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    print(line)
    print("  " + "-+-".join("-" * width for width in widths))
    for r in shown:
        print("  " + " | ".join(("NULL" if v is None else str(v)).ljust(widths[i]) for i, v in enumerate(r)))
    if len(rows) > max_rows:
        print(f"  ... {len(rows) - max_rows} more rows")
