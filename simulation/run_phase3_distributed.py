"""
Phase 3 -- run the distributed Monte Carlo batch and populate the silver layer.

This is the LOCAL DRIVER. It packages the frozen simulation modules, ships them
to Databricks, executes `phase3_notebook_source.py` there, and validates what
came back. The Monte Carlo itself runs entirely on the cluster.

    python simulation/run_phase3_distributed.py
    python simulation/run_phase3_distributed.py --checks-only   # re-run SQL checks
    python simulation/run_phase3_distributed.py --keep-notebook


WHY A SUBMITTED NOTEBOOK RUN, AND NOT DATABRICKS CONNECT
========================================================
Three platform facts about this workspace, each established by trying it:

1. DATABRICKS CONNECT CANNOT RUN PYTHON UDFs HERE. Every variant --
   applyInPandas, mapInPandas, pandas_udf, plain udf -- fails server-side with

       [ISOLATION_STARTUP_FAILURE.SANDBOX_STARTUP]
       failed to load /databricks/python3/bin/python: exec format error

   Plain SQL over Connect is fine; the UDF sandbox is not. `spark_session.py`
   (the Connect helper) is therefore a dead end for Phase 3 and nothing here
   imports it.

2. THERE IS NO CLASSIC COMPUTE PLANE. `clusters.list_zones()` fails with
   `No such workerEnvironment 'serverless-...'`, and a create call hangs and
   times out. All-purpose clusters cannot be created, so
   `05_create_phase3_cluster.py` is unusable in this workspace.

3. A NOTEBOOK EXECUTED VIA A TRANSIENT `jobs.submit` ON SERVERLESS JOB COMPUTE
   RUNS PYTHON UDFs CORRECTLY. Same serverless family, different execution
   path, and applyInPandas works.

This is a ONE-OFF SUBMISSION, not Phase 5 work: nothing is persisted as a job
definition, there is no schedule, no multi-task DAG, and no MLflow. The run
disappears when it finishes.


MODULE DISTRIBUTION -- THE ACTUAL TECHNICAL PROBLEM
===================================================
The UDF closure calls `cell.simulate_cell`, which lives in local .py files. Those
modules have to be importable ON THE EXECUTORS, which is a stricter requirement
than being importable on the driver. Three mechanisms were available:

  (a) build a wheel and install it                     <- USED
  (b) upload simulation/ as workspace files + sys.path
  (c) ship source through a UC volume + sys.path

(b) and (c) both work by mutating `sys.path` in the driver process. The Python
UDF workers are separate processes that do not inherit it, which is why those
approaches characteristically import fine on the driver and then raise
ModuleNotFoundError inside the UDF. They can be rescued with
`spark.addArtifact` / `addPyFile`, but that is one more thing to get right per
session and it is invisible until an executor actually runs.

(a) is installed by the runtime itself, before any user code runs, into the
environment the whole compute shares -- driver and UDF workers alike. Here the
wheel is named in the serverless ENVIRONMENT SPEC
(`compute.Environment(dependencies=[...])`) rather than `%pip`-installed in a
cell, so it is resolved as part of bringing the compute up.

The wheel is built here, from the repo's own files, with NO edits to them:
engine.py, cell.py, scenarios.py, allocations.py, seeding.py and config.py are
copied verbatim into a package `ad_mc_sim`. Their existing
`try: from . import x / except ImportError: import x` preamble means the
verbatim copies import correctly as a package. `engine.py` in particular is
FROZEN Phase 2 code and is shipped byte-for-byte.

The wheel VERSION carries a hash of those files' contents. That is not
decoration: serverless caches resolved environments by spec, so a fixed
filename plus changed contents could silently run stale code. A content-derived
name makes that impossible.

Whether this actually worked on the executors is not assumed -- the notebook's
first action is an applyInPandas that imports `ad_mc_sim` inside the UDF, calls
into the engine, and reports the worker's hostname, pid and library versions.
The run aborts there if it fails.


WRITE SEMANTICS
===============
INSERT OVERWRITE. `silver.simulated_allocation_outcomes` has no run_id, so
appending would stack duplicate grids that Phase 4 would silently average.
Seeds are a pure function of (allocation_id, scenario_id), so a re-run
reproduces the identical 840,000 rows -- full replacement is the honest
semantics and leaves exactly 840,000 rows after any number of runs.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "databricks"))
sys.path.insert(0, str(REPO_ROOT / "simulation"))

from _common import (  # noqa: E402
    TBL_ALLOCATION_CANDIDATES,
    TBL_SCENARIOS,
    TBL_SIMULATED_OUTCOMES,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)

from databricks.sdk.service import compute, jobs  # noqa: E402
from databricks.sdk.service.workspace import ImportFormat, Language  # noqa: E402

import engine  # noqa: E402
import scenarios as scenarios_mod  # noqa: E402
from config import CORRELATION_SOURCE, N_PATHS, RANDOM_SEED, TOTAL_BUDGET, build_allocation  # noqa: E402
from data_access import load_assumptions, load_correlation_matrix  # noqa: E402

# --- What gets packaged -------------------------------------------------------
# data_access.py is deliberately EXCLUDED: it reaches back through
# databricks/_common.py to a SQL warehouse, which is exactly the wrong way to
# read data from inside a Spark job. The notebook reads the same tables through
# Spark instead. run_phase2_simulation.py and spark_session.py are excluded as
# entry points with no role on the cluster.
PACKAGE_NAME = "ad_mc_sim"
PACKAGE_MODULES = ["config.py", "engine.py", "cell.py", "scenarios.py",
                   "allocations.py", "seeding.py"]

VOLUME_DIR = "/Volumes/ad_mc_poc/bronze/landing/wheels"
ANCHOR_VOLUME_PATH = "/Volumes/ad_mc_poc/bronze/landing/phase3/anchor_revenue_cluster.npy"
NOTEBOOK_SOURCE = REPO_ROOT / "simulation" / "phase3_notebook_source.py"
STALE_PROBE_NOTEBOOK = "_phase3_udf_probe.py"

EXPECTED_CELLS = 84
EXPECTED_ROWS = EXPECTED_CELLS * N_PATHS

# Phase 2's verified numbers, restated so the comparison has a fixed target.
PHASE2_MEAN, PHASE2_STD, PHASE2_VAR95 = 968_027.0, 169_364.0, 715_966.0

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}{(' -- ' + detail) if detail else ''}")


def money(x) -> str:
    return f"${float(x):,.2f}"


# =============================================================================
# 1. Build the wheel
# =============================================================================

def build_wheel() -> tuple[str, bytes]:
    """Package the frozen simulation modules as `ad_mc_sim`. Returns (filename, bytes).

    Hand-assembled rather than shelled out to setuptools/build so it has no
    build-time dependency and produces byte-identical output for identical
    inputs -- which is what makes the content-hashed version meaningful.
    """
    src_dir = REPO_ROOT / "simulation"
    sources = {}
    for name in PACKAGE_MODULES:
        path = src_dir / name
        if not path.exists():
            raise SystemExit(f"ERROR: {path} does not exist; cannot build the wheel")
        sources[name] = path.read_bytes()

    init = (
        f'"""Phase 2 simulation modules, packaged verbatim for Spark executors.\n\n'
        f"Built by simulation/run_phase3_distributed.py from the repo's own files.\n"
        f'No module here is edited -- engine.py in particular is frozen."""\n'
    ).encode()
    sources["__init__.py"] = init

    digest = hashlib.sha256()
    for name in sorted(sources):
        digest.update(name.encode())
        digest.update(sources[name])
    content_hash = digest.hexdigest()[:12]
    # PEP 440 `.postN` keeps the version purely alphanumeric+dots, so the wheel
    # filename needs no escaping. A `+local` label would also be legal but has
    # to be escaped in the filename, which some tooling gets wrong.
    version = f"0.1.0.post{int(content_hash[:8], 16)}"
    dist_info = f"{PACKAGE_NAME}-{version}.dist-info"
    filename = f"{PACKAGE_NAME}-{version}-py3-none-any.whl"

    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {PACKAGE_NAME}\n"
        f"Version: {version}\n"
        "Summary: Ad Revenue Monte Carlo POC -- frozen Phase 2 simulation modules\n"
        "Requires-Python: >=3.9\n"
        "Requires-Dist: numpy\n"
        "Requires-Dist: pandas\n"
        "Requires-Dist: scipy\n"
        f"\nSource content sha256[:12] = {content_hash}\n"
    ).encode()
    wheel_meta = (
        "Wheel-Version: 1.0\n"
        "Generator: ad_mc_poc-phase3-runner\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode()

    entries: list[tuple[str, bytes]] = []
    for name in sorted(sources):
        entries.append((f"{PACKAGE_NAME}/{name}", sources[name]))
    entries.append((f"{dist_info}/METADATA", metadata))
    entries.append((f"{dist_info}/WHEEL", wheel_meta))

    record_lines = []
    for arc, data in entries:
        sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record_lines.append(f"{arc},sha256={sha},{len(data)}")
    record_lines.append(f"{dist_info}/RECORD,,")
    entries.append((f"{dist_info}/RECORD", ("\n".join(record_lines) + "\n").encode()))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, data in entries:
            info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)

    print(f"  built {filename} ({buf.tell():,} bytes)")
    print(f"    modules: {sorted(sources)}")
    print(f"    content sha256[:12] = {content_hash}")
    return filename, buf.getvalue()


# =============================================================================
# 2. Ship it and run it
# =============================================================================

def upload_wheel(w, filename: str, data: bytes) -> str:
    try:
        w.files.create_directory(VOLUME_DIR)
    except Exception:  # noqa: BLE001 - already exists is fine
        pass
    path = f"{VOLUME_DIR}/{filename}"
    w.files.upload(path, io.BytesIO(data), overwrite=True)
    print(f"  uploaded wheel -> {path}")
    return path


def upload_notebook(w, user: str) -> str:
    """Upload the source notebook. Note the .py suffix -- it is load-bearing.

    `workspace.upload` with format=SOURCE lands the object at `<path>.py`, and
    `notebook_task.notebook_path` must then include that `.py`. Passing the
    extensionless path gives 'Unable to access the notebook'.
    """
    base = f"/Users/{user}/_ad_mc_phase3_distributed"
    src = NOTEBOOK_SOURCE.read_bytes()
    w.workspace.upload(base + ".py", io.BytesIO(src), format=ImportFormat.SOURCE,
                       language=Language.PYTHON, overwrite=True)
    print(f"  uploaded notebook -> {base}.py ({len(src):,} bytes)")
    return base + ".py"


def submit_and_wait(w, notebook_path: str, wheel_path: str, params: dict,
                    timeout_min: int) -> tuple[dict, dict]:
    """Submit the transient run, poll it, and return (run_metadata, notebook_output)."""
    waiter = w.jobs.submit(
        run_name=f"ad_mc_poc phase3 distributed simulation {int(time.time())}",
        tasks=[
            jobs.SubmitTask(
                task_key="phase3_distributed_simulation",
                notebook_task=jobs.NotebookTask(
                    notebook_path=notebook_path,
                    base_parameters={k: str(v) for k, v in params.items()},
                ),
                environment_key="phase3_env",
            )
        ],
        environments=[
            jobs.JobEnvironment(
                environment_key="phase3_env",
                spec=compute.Environment(client="4", dependencies=[wheel_path]),
            )
        ],
    )
    run_id = waiter.run_id
    url = f"{w.config.host}/#job/run/{run_id}"
    print(f"  submitted run {run_id}")
    print(f"  {url}")

    t0 = time.time()
    last = None
    while True:
        run = w.jobs.get_run(run_id)
        state = run.state
        life = str(getattr(state.life_cycle_state, "value", state.life_cycle_state))
        if life != last:
            print(f"    [{time.time() - t0:6.1f}s] {life}"
                  + (f" -- {state.state_message}" if state.state_message else ""))
            last = life
        if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        if time.time() - t0 > timeout_min * 60:
            raise SystemExit(f"ERROR: run {run_id} did not finish within {timeout_min} min")
        time.sleep(10)

    result_state = str(getattr(run.state.result_state, "value", run.state.result_state))
    elapsed = time.time() - t0
    print(f"  finished: {result_state} in {elapsed:.1f}s")

    meta = {"run_id": run_id, "url": url, "result_state": result_state,
            "wall_clock_s": round(elapsed, 1)}

    task_run_id = run.tasks[0].run_id if run.tasks else run_id
    out = w.jobs.get_run_output(task_run_id)
    if out.error:
        print("\n  RUN ERROR:\n", out.error)
        if out.error_trace:
            print(out.error_trace[:4000])
    raw = out.notebook_output.result if out.notebook_output else None
    if not raw:
        raise SystemExit(f"ERROR: run {run_id} produced no notebook output "
                         f"(result_state={result_state})")
    return meta, json.loads(raw)


# =============================================================================
# 3. Validation
# =============================================================================

def local_phase2_anchor(w, warehouse_id) -> np.ndarray:
    """Recompute Phase 2's anchor revenue vector locally, from live bronze data.

    This is the reference the cluster's pinned-seed run is diffed against. It is
    recomputed rather than read from a saved artefact so the comparison cannot
    be contaminated by a stale file.
    """
    a = load_assumptions(w, warehouse_id)
    channels = a["channel_id"].tolist()
    corr = load_correlation_matrix(w, warehouse_id, channels, CORRELATION_SOURCE)
    result = engine.simulate(
        channels=channels,
        allocation=build_allocation(channels, "even_split", TOTAL_BUDGET),
        mean_cvr=a["mean_cvr"].to_numpy(), std_cvr=a["std_cvr"].to_numpy(),
        mean_cpc=a["mean_cpc"].to_numpy(), std_cpc=a["std_cpc"].to_numpy(),
        mean_rpc=a["mean_revenue_per_conversion"].to_numpy(),
        std_rpc=a["std_revenue_per_conversion"].to_numpy(),
        correlation=corr, n_paths=N_PATHS, seed=RANDOM_SEED,
    )
    return result.total_revenue


def compare_anchor(local_rev: np.ndarray, payload: dict) -> None:
    """Exact-reproduction side test: cluster vs local, at the same pinned seed."""
    blob = payload.get("anchor_vector_gz_b64")
    if not blob:
        raise SystemExit("ERROR: notebook returned no anchor vector")
    cluster_rev = np.frombuffer(gzip.decompress(base64.b64decode(blob)), dtype=np.float64)
    stats = payload["anchor"]

    print(f"\n  vector length: local {len(local_rev):,}  cluster {len(cluster_rev):,}")
    check("anchor vectors are the same length", len(local_rev) == len(cluster_rev))

    same_bits = np.array_equal(local_rev.view(np.uint64), cluster_rev.view(np.uint64))
    n_diff = int((local_rev != cluster_rev).sum())
    abs_diff = np.abs(local_rev - cluster_rev)
    rel_diff = abs_diff / np.abs(local_rev)

    print(f"  bitwise identical            : {same_bits}")
    print(f"  paths differing at all       : {n_diff:,} of {len(local_rev):,} "
          f"({n_diff / len(local_rev):.2%})")
    print(f"  max abs difference           : {abs_diff.max():.6e}   ({money(abs_diff.max())})")
    print(f"  max relative difference      : {rel_diff.max():.6e}")
    print(f"  mean abs difference          : {abs_diff.mean():.6e}")

    lm = float(local_rev.mean())
    ls = float(local_rev.std(ddof=1))
    lv = float(np.percentile(local_rev, 5.0))
    lc = float(local_rev[local_rev <= lv].mean())
    print(f"\n  {'metric':<10} {'Phase 2 (local)':>20} {'cluster':>20} {'difference':>18}")
    for name, loc, clu in (("mean", lm, stats["mean"]), ("std", ls, stats["std"]),
                           ("VaR-95", lv, stats["var95"]), ("CVaR-95", lc, stats["cvar95"])):
        print(f"  {name:<10} {money(loc):>20} {money(clu):>20} {clu - loc:>18.10f}")

    check("cluster anchor mean matches Phase 2 to the cent",
          abs(stats["mean"] - lm) < 0.005, f"delta {stats['mean'] - lm:.10f}")
    check("cluster anchor std matches Phase 2 to the cent",
          abs(stats["std"] - ls) < 0.005, f"delta {stats['std'] - ls:.10f}")
    check("cluster anchor VaR-95 matches Phase 2 to the cent",
          abs(stats["var95"] - lv) < 0.005, f"delta {stats['var95'] - lv:.10f}")
    check("cluster anchor mean matches the published $968,027",
          round(stats["mean"]) == PHASE2_MEAN, f"got {money(stats['mean'])}")
    check("cluster anchor std matches the published $169,364",
          round(stats["std"]) == PHASE2_STD, f"got {money(stats['std'])}")
    check("cluster anchor VaR-95 matches the published $715,966",
          round(stats["var95"]) == PHASE2_VAR95, f"got {money(stats['var95'])}")


def in_grid_agreement(w, wid) -> None:
    """Statistical agreement: the DERIVED-seed anchor cell vs Phase 2's mean."""
    cols, rows = sql(w, wid, f"""
        SELECT COUNT(*)                                        AS n,
               AVG(total_simulated_revenue)                    AS mean,
               STDDEV_SAMP(total_simulated_revenue)            AS std,
               PERCENTILE(total_simulated_revenue, 0.05)       AS var95
        FROM {TBL_SIMULATED_OUTCOMES}
        WHERE allocation_id = 'even_split' AND scenario_id = 'normal'
    """)
    r = dict(zip(cols, rows[0]))
    n, mean, std = int(r["n"]), float(r["mean"]), float(r["std"])
    var95 = float(r["var95"])

    _c, tail = sql(w, wid, f"""
        SELECT AVG(total_simulated_revenue) FROM {TBL_SIMULATED_OUTCOMES}
        WHERE allocation_id = 'even_split' AND scenario_id = 'normal'
          AND total_simulated_revenue <= (
              SELECT PERCENTILE(total_simulated_revenue, 0.05)
              FROM {TBL_SIMULATED_OUTCOMES}
              WHERE allocation_id = 'even_split' AND scenario_id = 'normal')
    """)
    cvar95 = float(tail[0][0])

    sem = std / np.sqrt(n)
    z = (mean - PHASE2_MEAN) / sem
    print(f"  paths                        : {n:,}")
    print(f"  mean                         : {money(mean)}")
    print(f"  std                          : {money(std)}")
    print(f"  VaR-95                       : {money(var95)}")
    print(f"  CVaR-95                      : {money(cvar95)}")
    print(f"  Monte Carlo standard error   : {money(sem)}")
    print(f"  Phase 2 mean                 : {money(PHASE2_MEAN)}")
    print(f"  difference                   : {money(mean - PHASE2_MEAN)}  = {z:+.2f} SE")
    check("derived-seed anchor mean is within 3 SE of Phase 2's mean",
          abs(z) < 3.0, f"{z:+.2f} SE")
    check("derived-seed anchor std is within 5% of Phase 2's std",
          abs(std / PHASE2_STD - 1) < 0.05, f"{std / PHASE2_STD - 1:+.2%}")


def sanity_checks(w, wid) -> None:
    """Data quality on the full 840,000-row table."""
    n = int(sql(w, wid, f"SELECT COUNT(*) FROM {TBL_SIMULATED_OUTCOMES}")[1][0][0])
    print(f"  row count: {n:,}")
    check(f"row count is exactly {EXPECTED_ROWS:,}", n == EXPECTED_ROWS,
          f"found {n:,}")

    _c, r = sql(w, wid, f"""
        SELECT COUNT(*) AS cells,
               MIN(n) AS min_paths, MAX(n) AS max_paths,
               SUM(CASE WHEN n <> {N_PATHS} THEN 1 ELSE 0 END) AS wrong_cells
        FROM (SELECT allocation_id, scenario_id, COUNT(*) AS n
              FROM {TBL_SIMULATED_OUTCOMES} GROUP BY allocation_id, scenario_id)
    """)
    cells, min_p, max_p, wrong = (int(x) for x in r[0])
    print(f"  cells: {cells}, paths per cell min/max: {min_p}/{max_p}")
    check(f"exactly {EXPECTED_CELLS} (allocation, scenario) pairs", cells == EXPECTED_CELLS,
          f"found {cells}")
    check(f"every pair has exactly {N_PATHS:,} paths", wrong == 0,
          f"{wrong} pair(s) off; range {min_p}-{max_p}")

    _c, r = sql(w, wid, f"""
        SELECT SUM(CASE WHEN allocation_id IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN scenario_id IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN path_number IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN total_simulated_revenue IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN total_spend IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN roas IS NULL THEN 1 ELSE 0 END)
        FROM {TBL_SIMULATED_OUTCOMES}
    """)
    nulls = [int(x) for x in r[0]]
    print(f"  nulls per column (allocation_id, scenario_id, path_number, revenue, spend, roas): {nulls}")
    check("zero nulls in every column", sum(nulls) == 0, f"{nulls}")

    _c, r = sql(w, wid, f"""
        SELECT SUM(CASE WHEN total_spend <= 0 THEN 1 ELSE 0 END)                AS bad_spend,
               SUM(CASE WHEN total_simulated_revenue < 0 THEN 1 ELSE 0 END)     AS neg_rev,
               SUM(CASE WHEN total_spend <> {TOTAL_BUDGET} THEN 1 ELSE 0 END)   AS wrong_budget,
               SUM(CASE WHEN ABS(roas - total_simulated_revenue / total_spend)
                             > 1e-12 THEN 1 ELSE 0 END)                         AS roas_mismatch,
               MIN(total_simulated_revenue) AS min_rev, MAX(total_simulated_revenue) AS max_rev,
               MIN(path_number) AS min_path, MAX(path_number) AS max_path
        FROM {TBL_SIMULATED_OUTCOMES}
    """)
    bad_spend, neg_rev, wrong_budget, roas_mismatch = (int(x) for x in r[0][:4])
    min_rev, max_rev = float(r[0][4]), float(r[0][5])
    min_path, max_path = int(r[0][6]), int(r[0][7])
    print(f"  revenue range: {money(min_rev)} .. {money(max_rev)}")
    print(f"  path_number range: {min_path} .. {max_path}")
    check("no zero or negative total_spend", bad_spend == 0, f"{bad_spend} row(s)")
    check("no negative total_simulated_revenue", neg_rev == 0, f"{neg_rev} row(s)")
    check(f"every total_spend equals the ${TOTAL_BUDGET:,.0f} budget", wrong_budget == 0,
          f"{wrong_budget} row(s)")
    check("roas == revenue / spend on every row (rel. 1e-12)", roas_mismatch == 0,
          f"{roas_mismatch} row(s)")
    check("path_number is 0-based and contiguous to n_paths-1",
          min_path == 0 and max_path == N_PATHS - 1, f"{min_path}..{max_path}")

    _c, r = sql(w, wid, f"""
        SELECT (SELECT COUNT(DISTINCT allocation_id) FROM {TBL_SIMULATED_OUTCOMES}),
               (SELECT COUNT(DISTINCT scenario_id) FROM {TBL_SIMULATED_OUTCOMES}),
               (SELECT COUNT(*) FROM (SELECT DISTINCT allocation_id FROM {TBL_SIMULATED_OUTCOMES}) o
                 LEFT ANTI JOIN {TBL_ALLOCATION_CANDIDATES} a ON o.allocation_id = a.allocation_id),
               (SELECT COUNT(*) FROM (SELECT DISTINCT scenario_id FROM {TBL_SIMULATED_OUTCOMES}) o
                 LEFT ANTI JOIN {TBL_SCENARIOS} s ON o.scenario_id = s.scenario_id),
               (SELECT COUNT(*) FROM (SELECT DISTINCT allocation_id FROM {TBL_ALLOCATION_CANDIDATES}) a
                 LEFT ANTI JOIN (SELECT DISTINCT allocation_id FROM {TBL_SIMULATED_OUTCOMES}) o
                   ON o.allocation_id = a.allocation_id),
               (SELECT COUNT(*) FROM {TBL_SCENARIOS} s
                 LEFT ANTI JOIN (SELECT DISTINCT scenario_id FROM {TBL_SIMULATED_OUTCOMES}) o
                   ON o.scenario_id = s.scenario_id)
    """)
    n_alloc, n_scen, orphan_a, orphan_s, missing_a, missing_s = (int(x) for x in r[0])
    print(f"  distinct allocation_ids: {n_alloc}, distinct scenario_ids: {n_scen}")
    check("all 21 allocation_ids present", n_alloc == 21, f"found {n_alloc}")
    check("all 4 scenario_ids present", n_scen == 4, f"found {n_scen}")
    check("no allocation_id without a row in silver.allocation_candidates",
          orphan_a == 0, f"{orphan_a} orphan(s)")
    check("no scenario_id without a row in bronze.scenario_definitions",
          orphan_s == 0, f"{orphan_s} orphan(s)")
    check("every candidate allocation was simulated", missing_a == 0,
          f"{missing_a} not simulated")
    check("every defined scenario was simulated", missing_s == 0,
          f"{missing_s} not simulated")

    _c, r = sql(w, wid, f"""
        SELECT COUNT(*) FROM (
            SELECT allocation_id, scenario_id, path_number, COUNT(*) AS n
            FROM {TBL_SIMULATED_OUTCOMES}
            GROUP BY allocation_id, scenario_id, path_number HAVING COUNT(*) > 1)
    """)
    check("grain (allocation_id, scenario_id, path_number) is unique",
          int(r[0][0]) == 0, f"{r[0][0]} duplicated key(s)")


def directional_checks(w, wid) -> None:
    """Scenario ordering must track each scenario's net_revenue_factor."""
    expected = {s.scenario_id: s.net_revenue_factor for s in scenarios_mod.SCENARIOS}

    cols, rows = sql(w, wid, f"""
        WITH m AS (
            SELECT allocation_id, scenario_id, AVG(total_simulated_revenue) AS mean_rev
            FROM {TBL_SIMULATED_OUTCOMES} GROUP BY allocation_id, scenario_id),
        n AS (SELECT allocation_id, mean_rev AS normal_rev FROM m WHERE scenario_id = 'normal')
        SELECT m.scenario_id,
               COUNT(*)                          AS allocations,
               AVG(m.mean_rev / n.normal_rev)    AS mean_ratio_to_normal,
               MIN(m.mean_rev / n.normal_rev)    AS min_ratio,
               MAX(m.mean_rev / n.normal_rev)    AS max_ratio
        FROM m JOIN n USING (allocation_id)
        GROUP BY m.scenario_id ORDER BY mean_ratio_to_normal DESC
    """)
    print()
    print(f"  {'scenario':<22} {'allocs':>7} {'mean ratio':>12} {'min':>10} {'max':>10} "
          f"{'expected':>10} {'error':>9}")
    for sid, allocs, ratio, lo, hi in rows:
        exp = expected[sid]
        ratio, lo, hi = float(ratio), float(lo), float(hi)
        print(f"  {sid:<22} {int(allocs):>7} {ratio:>12.4f} {lo:>10.4f} {hi:>10.4f} "
              f"{exp:>10.4f} {ratio / exp - 1:>+8.2%}")
        check(f"{sid}: mean revenue ratio tracks net_revenue_factor {exp:.4f} within 2%",
              abs(ratio / exp - 1) < 0.02, f"observed {ratio:.4f}")

    _c, r = sql(w, wid, f"""
        WITH m AS (
            SELECT allocation_id, scenario_id, AVG(total_simulated_revenue) AS mean_rev
            FROM {TBL_SIMULATED_OUTCOMES} GROUP BY allocation_id, scenario_id)
        SELECT SUM(CASE WHEN r.mean_rev >= nn.mean_rev THEN 1 ELSE 0 END) AS violations,
               COUNT(*) AS pairs
        FROM m r JOIN m nn USING (allocation_id)
        WHERE r.scenario_id = 'recession' AND nn.scenario_id = 'normal'
    """)
    violations, pairs = int(r[0][0]), int(r[0][1])
    check("recession expected revenue < normal for EVERY allocation",
          violations == 0, f"{violations} of {pairs} allocations violate it")

    _c, r = sql(w, wid, f"""
        WITH m AS (
            SELECT allocation_id, scenario_id, AVG(total_simulated_revenue) AS mean_rev
            FROM {TBL_SIMULATED_OUTCOMES} GROUP BY allocation_id, scenario_id)
        SELECT SUM(CASE WHEN p.mean_rev >= r.mean_rev THEN 1 ELSE 0 END), COUNT(*)
        FROM m p JOIN m r USING (allocation_id)
        WHERE p.scenario_id = 'platform_algo_change' AND r.scenario_id = 'recession'
    """)
    check("platform_algo_change is the worst scenario for EVERY allocation "
          "(factor 0.757 < 0.823)", int(r[0][0]) == 0,
          f"{r[0][0]} of {r[0][1]} allocations violate it")


def report_grid(w, wid) -> None:
    print("\n== Per-scenario aggregates over all 21 allocations ==")
    cols, rows = sql(w, wid, f"""
        SELECT scenario_id, COUNT(*) AS paths,
               ROUND(AVG(total_simulated_revenue), 2)         AS mean_revenue,
               ROUND(STDDEV_SAMP(total_simulated_revenue), 2) AS std_revenue,
               ROUND(PERCENTILE(total_simulated_revenue, 0.05), 2) AS var95,
               ROUND(AVG(roas), 4) AS mean_roas
        FROM {TBL_SIMULATED_OUTCOMES} GROUP BY scenario_id ORDER BY mean_revenue DESC
    """)
    print_table(cols, rows)

    print("\n== 'normal' scenario, every allocation, by expected revenue ==")
    cols, rows = sql(w, wid, f"""
        SELECT allocation_id, COUNT(*) AS paths,
               ROUND(AVG(total_simulated_revenue), 0)         AS mean_revenue,
               ROUND(STDDEV_SAMP(total_simulated_revenue), 0) AS std_revenue,
               ROUND(PERCENTILE(total_simulated_revenue, 0.05), 0) AS var95,
               ROUND(AVG(roas), 3) AS mean_roas
        FROM {TBL_SIMULATED_OUTCOMES} WHERE scenario_id = 'normal'
        GROUP BY allocation_id ORDER BY mean_revenue DESC
    """)
    print_table(cols, rows, max_rows=25)


# =============================================================================

def cleanup_stale_probe(w, user: str) -> None:
    path = f"/Users/{user}/{STALE_PROBE_NOTEBOOK}"
    try:
        w.workspace.delete(path)
        print(f"  deleted stale probe notebook {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  stale probe notebook {path}: {type(exc).__name__} (already gone?)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks-only", action="store_true",
                        help="Skip the cluster run; re-run the SQL checks on the existing table.")
    parser.add_argument("--n-paths", type=int, default=N_PATHS)
    parser.add_argument("--shuffle-partitions", type=int, default=EXPECTED_CELLS)
    parser.add_argument("--write-mode", default="overwrite",
                        choices=["overwrite", "append", "none"])
    parser.add_argument("--timeout-min", type=int, default=45)
    parser.add_argument("--keep-notebook", action="store_true",
                        help="Leave the uploaded notebook in the workspace.")
    args = parser.parse_args()

    print("=" * 78)
    print("PHASE 3 -- DISTRIBUTED BATCH SIMULATION")
    print("=" * 78)

    w = get_client()
    wid = resolve_warehouse_id(w)
    user = w.current_user.me().user_name
    payload: dict = {}
    meta: dict = {}

    if not args.checks_only:
        print("\n== Packaging the frozen simulation modules ==")
        filename, data = build_wheel()

        print("\n== Shipping to the workspace ==")
        wheel_path = upload_wheel(w, filename, data)
        notebook_path = upload_notebook(w, user)

        print("\n== Submitting the run (transient, serverless job compute) ==")
        meta, payload = submit_and_wait(
            w, notebook_path, wheel_path,
            {
                "n_paths": args.n_paths,
                "shuffle_partitions": args.shuffle_partitions,
                "write_mode": args.write_mode,
                "anchor_seed": RANDOM_SEED,
                "anchor_out_path": ANCHOR_VOLUME_PATH,
                "catalog": "ad_mc_poc",
            },
            args.timeout_min,
        )

        if payload.get("status") != "ok":
            print("\nNOTEBOOK REPORTED A FAILURE")
            print(json.dumps({k: v for k, v in payload.items()
                              if k != "anchor_vector_gz_b64"}, indent=2)[:6000])
            raise SystemExit(1)

        print("\n== Executor import probe (module distribution proof) ==")
        probe = payload["executor_probe"]
        print(json.dumps(probe, indent=2))
        check("ad_mc_sim imported inside the UDF on every group",
              probe["rows"] == 8, f"{probe['rows']} groups reported")
        check("the UDF ran in a process other than the driver",
              probe["pid_differs_from_driver"],
              f"driver pid {probe['driver_pid']}, udf pids {probe['distinct_pids']}")
        if probe.get("sandbox_is_pid_namespaced"):
            print("  NOTE: the serverless UDF sandbox is PID-namespaced -- every worker "
                  "reports pid 1 and an empty hostname, so (host, pid) CANNOT count "
                  "worker processes here. Worker identity below uses a per-process "
                  "UUID stashed on the imported module instead.")
        check("the frozen engine executed on the executor",
              probe["engine_call_ok"])
        check("ad_mc_sim resolved from an installed site-packages path, "
              "not a sys.path hack",
              "site-packages" in probe["executor_module_file"],
              probe["executor_module_file"])

        print("\n== Grid ==")
        print(json.dumps(payload["grid"], indent=2))
        check("grid is 84 cells", payload["grid"]["n_cells"] == EXPECTED_CELLS)
        check("all 84 cell seeds are distinct",
              payload["grid"]["distinct_seeds"] == EXPECTED_CELLS)

        print("\n== Write ==")
        print(json.dumps(payload["write"], indent=2))

        print("\n== Parallelism (measured, not asserted) ==")
        par = payload["parallelism"]
        print(json.dumps(par, indent=2))
        check("the 84 cells ran across more than one UDF worker process",
              par["n_distinct_workers"] > 1,
              f"{par['n_distinct_workers']} worker process(es); "
              f"cells per worker {par['cells_per_worker']}")
        check("more than one cell was in flight at the same time",
              par["peak_concurrent_cells"] > 1,
              f"peak concurrency {par['peak_concurrent_cells']}")
        print(f"  NOTE: this grid is ~{par['sum_of_cell_cpu_s']:.0f}s of single-threaded "
              f"numpy in total. Spark is demonstrating the mechanism here, not "
              f"rescuing an intractable workload.")

        print("\n" + "=" * 78)
        print("VALIDATION 1 -- EXACT REPRODUCTION AT PHASE 2'S PINNED SEED "
              f"({RANDOM_SEED})")
        print("=" * 78)
        print("  recomputing the Phase 2 anchor locally from live bronze data...")
        local_rev = local_phase2_anchor(w, wid)
        compare_anchor(local_rev, payload)

    print("\n" + "=" * 78)
    print("VALIDATION 2 -- IN-GRID STATISTICAL AGREEMENT (derived seed)")
    print("=" * 78)
    in_grid_agreement(w, wid)

    print("\n" + "=" * 78)
    print("SANITY CHECKS AT SCALE")
    print("=" * 78)
    sanity_checks(w, wid)

    print("\n== Directional checks vs scenario net_revenue_factor ==")
    directional_checks(w, wid)

    report_grid(w, wid)

    if not args.checks_only and not args.keep_notebook:
        print("\n== Cleanup ==")
        try:
            w.workspace.delete(f"/Users/{user}/_ad_mc_phase3_distributed.py")
            print("  deleted the uploaded run notebook")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not delete run notebook: {exc}")
        cleanup_stale_probe(w, user)

    print("\n" + "=" * 78)
    if meta:
        print(f"run: {meta['url']}  ({meta['result_state']}, {meta['wall_clock_s']}s end to end)")
    if FAILURES:
        print(f"FAILED -- {len(FAILURES)} check(s) did not pass:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("PHASE 3 DISTRIBUTED SIMULATION -- ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
