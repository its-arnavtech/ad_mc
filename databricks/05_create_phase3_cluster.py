"""
Phase 3 -- create (or reuse) the all-purpose cluster the distributed simulation
runs on, and print its id.

WHY A CLUSTER AT ALL, when Phases 1-2 deliberately needed none
--------------------------------------------------------------
Phases 1-2 only ever ran SQL, which the Statement Execution API serves from a
SQL warehouse. Phase 3 runs `applyInPandas`, which is a Python UDF and needs a
real Spark session with Python workers.

Serverless compute was tried first and rejected on evidence: the session opens
and SQL runs, but every Python UDF fails server-side with

    [ISOLATION_STARTUP_FAILURE.SANDBOX_STARTUP]
    failed to load /databricks/python3/bin/python: exec format error

That is a Databricks-side container fault in the UDF sandbox, not a fault in
this repo's code, and it reproduced on repeated attempts.

ACCESS MODE -- deliberate, not a default
----------------------------------------
`SINGLE_USER` (dedicated). Shared / `USER_ISOLATION` clusters execute Python
UDFs inside the same isolated sandbox that fails on serverless, so a shared
cluster would likely reproduce the identical failure. Dedicated mode runs UDFs
in the ordinary executor process.

DBR 16.4 LTS ships Python 3.12, which must match the local client's Python
minor version for UDF pickling to work -- see `.venv-spark` in the README.

    python databricks/05_create_phase3_cluster.py
"""

from __future__ import annotations

import sys

from databricks.sdk.service.compute import DataSecurityMode

from _common import get_client

CLUSTER_NAME = "ad-mc-poc-phase3"
SPARK_VERSION = "16.4.x-scala2.12"  # LTS, Python 3.12
NODE_TYPE = "m5d.large"
NUM_WORKERS = 2
AUTOTERMINATION_MINUTES = 20

# Reused by every Phase 3 script that needs a cluster id.
LIVE_STATES = {"RUNNING", "PENDING", "RESIZING", "RESTARTING"}


def find_existing(w):
    for c in w.clusters.list():
        if c.cluster_name == CLUSTER_NAME:
            return c
    return None


def main() -> None:
    w = get_client()

    existing = find_existing(w)
    if existing:
        state = str(getattr(existing.state, "value", existing.state)).upper()
        print(f"cluster '{CLUSTER_NAME}' already exists: id={existing.cluster_id} state={state}")
        if state not in LIVE_STATES:
            print("  starting it...")
            w.clusters.start(existing.cluster_id).result()
            print("  started.")
        print(f"\nCLUSTER_ID={existing.cluster_id}")
        return

    print(f"creating cluster '{CLUSTER_NAME}'")
    print(f"  runtime      : {SPARK_VERSION} (Python 3.12)")
    print(f"  nodes        : {NODE_TYPE} driver + {NUM_WORKERS} x {NODE_TYPE} workers")
    print(f"  access mode  : SINGLE_USER (dedicated -- UDFs must not run sandboxed)")
    print(f"  auto-terminate: {AUTOTERMINATION_MINUTES} min idle")
    print("  waiting for it to come up (this takes several minutes)...")

    me = w.current_user.me().user_name

    created = w.clusters.create(
        cluster_name=CLUSTER_NAME,
        spark_version=SPARK_VERSION,
        node_type_id=NODE_TYPE,
        num_workers=NUM_WORKERS,
        autotermination_minutes=AUTOTERMINATION_MINUTES,
        data_security_mode=DataSecurityMode.SINGLE_USER,
        single_user_name=me,
        custom_tags={"project": "ad_mc_poc", "phase": "3"},
    ).result()

    print(f"\ncluster is up: id={created.cluster_id} "
          f"state={getattr(created.state, 'value', created.state)}")
    print(f"\nCLUSTER_ID={created.cluster_id}")


if __name__ == "__main__":
    sys.exit(main())
