"""
Databricks Connect session for the Phase 3 distributed simulation.

WHY PHASE 3 NEEDS A SPARK SESSION AT ALL
----------------------------------------
Phases 1-2 deliberately needed no cluster: every operation was SQL, served by
the Statement Execution API against a SQL warehouse (see `databricks/_common.py`).
Phase 3 runs the Phase 2 engine inside `applyInPandas`, which is a Python UDF.
A UDF needs real Spark workers running real Python, so from here on there is a
cluster.

WHY NOT SERVERLESS -- DO NOT "HELPFULLY" SWITCH THIS BACK
---------------------------------------------------------
Serverless compute was tried first and rejected on evidence, not preference.
The session opens and SQL runs fine, but every Python UDF fails server-side on
this workspace with:

    [ISOLATION_STARTUP_FAILURE.SANDBOX_STARTUP]
    failed to load /databricks/python3/bin/python: exec format error

That is a Databricks-side container fault in the UDF sandbox, reproducible
across attempts, and nothing in this repo can work around it. Shared /
USER_ISOLATION clusters run UDFs in that same sandbox, so they are expected to
fail the same way. Hence a dedicated (SINGLE_USER) all-purpose cluster, created
by `databricks/05_create_phase3_cluster.py`.

If you are reading this because serverless "should" be simpler: it is, and it
does not work here. Re-test the UDF path before changing it.

INTERPRETER REQUIREMENT -- Python 3.12, i.e. the repo's .venv-spark
--------------------------------------------------------------------
Databricks Connect requires the *client* Python minor version to match the
cluster runtime's, because UDFs are pickled locally and unpickled on the
executors. DBR 16.4 LTS ships Python 3.12. The machine's default `python` is
3.13, which will NOT work.

Run anything that imports this module with the repo's dedicated venv:

    C:\\ad_mc\\.venv-spark\\Scripts\\python.exe simulation/spark_session.py

`.venv-spark` has `databricks-connect==16.4.*` installed. The Phase 1-2 SQL
scripts are unaffected -- they only use the SDK and still run on plain `python`.

AUTH
----
Host and token come from the Databricks unified auth chain (env vars or
~/.databrickscfg), exactly as in `databricks/_common.py`. Nothing here holds a
literal token; the only thing this module injects into the session is a cluster
id.

    # resolve + smoke-test the session
    C:\\ad_mc\\.venv-spark\\Scripts\\python.exe simulation/spark_session.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "databricks"))

from _common import get_client  # noqa: E402  -- import after sys.path bootstrap

try:
    from databricks.connect import DatabricksSession  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "databricks-connect is not installed in this interpreter.\n"
        f"  running under: {sys.executable} (Python {sys.version_info.major}.{sys.version_info.minor})\n"
        "Phase 3 must run under the repo's dedicated venv, which has\n"
        "databricks-connect==16.4.* on Python 3.12:\n"
        r"      C:\ad_mc\.venv-spark\Scripts\python.exe <script>"
    ) from exc


# Must match `CLUSTER_NAME` in databricks/05_create_phase3_cluster.py. Kept as a
# literal rather than imported, because that module's name starts with a digit
# and is not importable normally. If the two ever drift, resolution fails loudly
# below and prints the cluster names that DO exist.
CLUSTER_NAME = "ad-mc-poc-phase3"

# DBR 16.4 LTS ships Python 3.12; Databricks Connect requires the client to
# match on the minor version.
REQUIRED_PYTHON = (3, 12)

# States we can hand to Databricks Connect. PENDING/RESTARTING/RESIZING are fine
# -- the client waits for the cluster to finish coming up. TERMINATED,
# TERMINATING, ERROR and UNKNOWN are not, and are reported as such.
USABLE_STATES = {"RUNNING", "PENDING", "RESTARTING", "RESIZING"}

_START_HINT = "  python databricks/05_create_phase3_cluster.py"


def require_matching_python() -> None:
    """
    Fail before connecting if the interpreter cannot run Phase 3's UDFs.

    Databricks Connect will eventually complain about this itself, but usually
    only once a UDF fails to unpickle on an executor, which reads like a bug in
    the simulation rather than an environment mismatch. Better to say it here.
    """
    actual = sys.version_info[:2]
    if actual == REQUIRED_PYTHON:
        return
    want = ".".join(str(v) for v in REQUIRED_PYTHON)
    have = ".".join(str(v) for v in actual)
    raise SystemExit(
        f"ERROR: Phase 3 needs Python {want}, but this is Python {have}.\n"
        f"  interpreter: {sys.executable}\n\n"
        "Databricks Connect pickles UDFs locally and unpickles them on the\n"
        f"cluster's executors, so the client minor version must match the\n"
        f"runtime's. DBR 16.4 LTS = Python {want}.\n\n"
        "Re-run with the repo's dedicated venv:\n"
        r"  C:\ad_mc\.venv-spark\Scripts\python.exe " + " ".join(sys.argv)
    )


def resolve_cluster_id(w=None) -> str:
    """
    Cluster id for the Phase 3 cluster.

    Order:
      1. DATABRICKS_CLUSTER_ID env var, if set (escape hatch -- still validated)
      2. the cluster named CLUSTER_NAME in this workspace

    Resolving by name rather than hardcoding an id means the id can change
    (cluster deleted and recreated) without editing code. Raises SystemExit
    with an actionable message if the cluster is missing or not usable, rather
    than letting Databricks Connect fail later with a connection error.
    """
    w = w or get_client()

    explicit = os.environ.get("DATABRICKS_CLUSTER_ID")
    if explicit:
        try:
            cluster = w.clusters.get(explicit)
        except Exception as exc:  # noqa: BLE001 - any lookup failure is fatal here
            raise SystemExit(
                f"ERROR: DATABRICKS_CLUSTER_ID={explicit} could not be read "
                f"({type(exc).__name__}).\n"
                "Unset it to fall back to resolving the cluster by name "
                f"('{CLUSTER_NAME}')."
            ) from exc
        state = str(getattr(cluster.state, "value", cluster.state)).upper()
        print(f"  cluster (from DATABRICKS_CLUSTER_ID): {cluster.cluster_name} "
              f"[{explicit}] state={state}")
        _require_usable(cluster.cluster_name, explicit, state)
        return explicit

    clusters = list(w.clusters.list())
    match = [c for c in clusters if c.cluster_name == CLUSTER_NAME]
    if not match:
        existing = ", ".join(sorted(c.cluster_name for c in clusters)) or "(none)"
        raise SystemExit(
            f"ERROR: no cluster named '{CLUSTER_NAME}' in this workspace.\n"
            f"  clusters that do exist: {existing}\n\n"
            "Create (or restart) it with:\n"
            f"{_START_HINT}\n\n"
            "Or point at a specific cluster with DATABRICKS_CLUSTER_ID=<id>."
        )
    if len(match) > 1:
        ids = ", ".join(c.cluster_id for c in match)
        raise SystemExit(
            f"ERROR: {len(match)} clusters are named '{CLUSTER_NAME}' ({ids}).\n"
            "Resolution by name is ambiguous. Delete the duplicates, or pin one\n"
            "with DATABRICKS_CLUSTER_ID=<id>."
        )

    cluster = match[0]
    state = str(getattr(cluster.state, "value", cluster.state)).upper()
    print(f"  cluster: {CLUSTER_NAME} [{cluster.cluster_id}] state={state}")
    _require_usable(CLUSTER_NAME, cluster.cluster_id, state)
    return cluster.cluster_id


def _require_usable(name: str, cluster_id: str, state: str) -> None:
    if state in USABLE_STATES:
        return
    raise SystemExit(
        f"ERROR: cluster '{name}' [{cluster_id}] is {state}, so Phase 3 cannot "
        "connect to it.\n"
        "Start it (the script is idempotent -- it reuses the existing cluster):\n"
        f"{_START_HINT}\n\n"
        "Note it auto-terminates after 20 idle minutes by design, so seeing\n"
        "TERMINATED between runs is normal, not a failure."
    )


def get_spark(cluster_id: str | None = None):
    """
    A Databricks Connect SparkSession bound to the Phase 3 cluster.

    `cluster_id` is optional; by default it is resolved as described in
    `resolve_cluster_id`. Auth (host + token) is left entirely to the SDK's
    unified auth chain -- passing only the cluster id here means no credential
    ever appears in this repo.
    """
    require_matching_python()
    cluster_id = cluster_id or resolve_cluster_id()
    return DatabricksSession.builder.clusterId(cluster_id).getOrCreate()


def _smoke_test(spark) -> None:
    """
    Prove the two things Phase 3 actually depends on: that we can run SQL, and
    that we can run a *Python UDF*. The second is the one that fails on
    serverless, so it is the one worth checking every time the plumbing changes.
    """
    import pandas as pd

    print("\n== SQL round-trip ==")
    row = spark.sql("SELECT current_catalog() AS catalog, spark_version() AS version").collect()[0]
    print(f"  catalog={row['catalog']} runtime={row['version']}")

    print("\n== Python UDF (applyInPandas) round-trip ==")

    def _double(pdf: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"grp": pdf["grp"], "doubled": pdf["value"] * 2})

    df = spark.createDataFrame(
        [("a", 1.0), ("a", 2.0), ("b", 3.0)], "grp string, value double"
    )
    out = (
        df.groupBy("grp")
        .applyInPandas(_double, schema="grp string, doubled double")
        .orderBy("grp", "doubled")
        .collect()
    )
    print(f"  {[(r['grp'], r['doubled']) for r in out]}")
    print("  UDF sandbox OK -- this is what fails on serverless.")


def main() -> None:
    print("== Resolving Phase 3 cluster ==")
    require_matching_python()
    print(f"  interpreter: {sys.executable} "
          f"(Python {sys.version_info.major}.{sys.version_info.minor})")
    cluster_id = resolve_cluster_id()

    print("\n== Opening Databricks Connect session ==")
    spark = get_spark(cluster_id)
    print("  connected")

    _smoke_test(spark)
    print("\nSpark session helper OK.")


if __name__ == "__main__":
    main()
