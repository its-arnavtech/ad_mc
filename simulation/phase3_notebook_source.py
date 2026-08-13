# Databricks notebook source
# =============================================================================
# PHASE 3 -- DISTRIBUTED BATCH SIMULATION
#
# This file is a Databricks SOURCE-format notebook. It is not run locally and
# not imported by anything; `simulation/run_phase3_distributed.py` uploads it to
# the workspace and executes it as a one-off `jobs.submit` run on serverless job
# compute. Read that module first -- it explains why this shape was forced.
#
# WHAT RUNS HERE
#   1. prove `ad_mc_sim` (the packaged Phase 2 simulation modules) imports on the
#      EXECUTORS, not just the driver -- see the probe below
#   2. read bronze/silver reference tables through Spark
#   3. build the 21 x 4 = 84 cell grid in the DRIVER, with each cell's seed
#      computed here so seeds are visible in the job's own lineage
#   4. groupBy(allocation_id, scenario_id).applyInPandas(...) -- one 10,000-path
#      Monte Carlo batch per group, through the FROZEN Phase 2 engine
#   5. INSERT OVERWRITE ad_mc_poc.silver.simulated_allocation_outcomes
#   6. a side test: the anchor cell at Phase 2's pinned seed, for exact comparison
#   7. a diagnostics pass that measures how the work actually spread
#
# MODULE DISTRIBUTION
#   `ad_mc_sim` is a wheel built from simulation/{engine,cell,scenarios,
#   allocations,seeding,config}.py and installed through the serverless
#   ENVIRONMENT SPEC (`dependencies=[<wheel on a UC volume>]`), not through
#   `%pip` in a cell and not through sys.path hacking. The environment spec is
#   resolved before the notebook starts and applies to the whole compute,
#   including the Python UDF workers -- which is the failure mode the probe in
#   step 1 exists to rule out.
#
# NOTHING HERE REIMPLEMENTS THE MATH. `ad_mc_sim.cell.simulate_cell` wraps
# `ad_mc_sim.engine.simulate`, both byte-identical to the validated Phase 2
# modules.
# =============================================================================

# COMMAND ----------

import base64
import gzip
import json
import os
import socket
import sys
import time
import traceback

import numpy as np
import pandas as pd

from pyspark.sql import functions as F

START_TS = time.time()
RESULT: dict = {"status": "running"}

dbutils.widgets.text("n_paths", "10000", "Paths per cell")
dbutils.widgets.text("shuffle_partitions", "84", "spark.sql.shuffle.partitions")
dbutils.widgets.text("write_mode", "overwrite", "overwrite | append | none")
dbutils.widgets.text("anchor_seed", "20260808", "Phase 2 pinned seed for the side test")
dbutils.widgets.text("anchor_out_path",
                     "/Volumes/ad_mc_poc/bronze/landing/phase3/anchor_revenue_cluster.npy",
                     "Where to write the anchor revenue vector")
dbutils.widgets.text("catalog", "ad_mc_poc", "Catalog")

N_PATHS = int(dbutils.widgets.get("n_paths"))
SHUFFLE_PARTITIONS = int(dbutils.widgets.get("shuffle_partitions"))
WRITE_MODE = dbutils.widgets.get("write_mode").strip().lower()
ANCHOR_SEED = int(dbutils.widgets.get("anchor_seed"))
ANCHOR_OUT_PATH = dbutils.widgets.get("anchor_out_path")
CATALOG = dbutils.widgets.get("catalog")

TBL_ASSUMPTIONS = f"{CATALOG}.bronze.channel_assumptions"
TBL_CVR_CORR = f"{CATALOG}.bronze.channel_cvr_correlation_matrix"
TBL_SCENARIOS = f"{CATALOG}.bronze.scenario_definitions"
TBL_ALLOCATIONS = f"{CATALOG}.silver.allocation_candidates"
TBL_OUTCOMES = f"{CATALOG}.silver.simulated_allocation_outcomes"

DRIVER_ENV = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "hostname": socket.gethostname(),
    "pid": os.getpid(),
}
try:
    import scipy
    DRIVER_ENV["scipy"] = scipy.__version__
except Exception as exc:  # noqa: BLE001
    DRIVER_ENV["scipy"] = f"MISSING ({exc})"

print("driver environment:", json.dumps(DRIVER_ENV, indent=2))


def finish(status: str, **extra):
    """Single exit point, so a failure still returns structured output."""
    RESULT.update(extra)
    RESULT["status"] = status
    RESULT["driver_env"] = DRIVER_ENV
    RESULT["wall_clock_s"] = round(time.time() - START_TS, 2)
    dbutils.notebook.exit(json.dumps(RESULT, default=str))


# COMMAND ----------

# --- 0. SPARK CONF ----------------------------------------------------------
# Set before anything else runs, because it governs the probe too.
#
# groupBy(...).applyInPandas does its own shuffle on the grouping keys, so the
# knob that actually controls task count is spark.sql.shuffle.partitions -- a
# repartition() before the groupBy is discarded by that shuffle. AQE would
# otherwise coalesce 84 tiny partitions into a handful, which is the right call
# for a normal query and exactly wrong here, where each "tiny" row of the grid
# is 10,000 Monte Carlo paths. Both settings are attempted and whichever the
# platform refuses is reported rather than assumed.
conf_applied = {}
for _key, _value in (
    ("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS)),
    ("spark.sql.adaptive.coalescePartitions.enabled", "false"),
):
    try:
        spark.conf.set(_key, _value)
        conf_applied[_key] = spark.conf.get(_key)
    except Exception as exc:  # noqa: BLE001
        conf_applied[_key] = f"REFUSED: {type(exc).__name__}: {exc}"
print("spark conf:", json.dumps(conf_applied, indent=2))
RESULT["spark_conf"] = conf_applied
RESULT["spark_version"] = spark.version


# COMMAND ----------

# --- 1. EXECUTOR IMPORT PROBE ------------------------------------------------
# THE point of failure this notebook was designed around. Importing ad_mc_sim on
# the driver proves nothing about the Python UDF workers: workspace-file and
# sys.path based distribution routinely imports fine on the driver and raises
# ModuleNotFoundError inside the executor's UDF sandbox. So before a single
# Monte Carlo path is drawn, run the smallest possible applyInPandas that
# imports the package INSIDE the UDF, calls into the frozen engine, and reports
# the interpreter it ran under. If this cell does not produce rows whose pid
# differs from the driver's, the run stops here rather than producing numbers
# from an unknown code path.

PROBE_SCHEMA = (
    "grp int, host string, pid int, worker_uid string, module_file string, "
    "py string, numpy string, scipy string, beta_a double, beta_b double"
)


def _worker_uid() -> str:
    """Stable identity for the Python process this UDF is executing in.

    (hostname, pid) DOES NOT WORK on this platform. The serverless UDF sandbox
    is namespaced: every worker reports `os.getpid() == 1` and
    `socket.gethostname() == ""`, so all workers look like one worker and any
    count built on them is wrong -- measured, not assumed (see the first run's
    probe output). This instead stashes a UUID on the already-imported
    `ad_mc_sim` module object. `sys.modules` caches that module for the life of
    the interpreter, so the value is created once per PROCESS and shared by
    every task that process runs -- which is exactly the notion of "distinct
    worker" that matters, and it survives PID namespacing.

    PySpark reuses Python workers across tasks by default, so a repeated uid
    means genuine reuse rather than a lost identity.
    """
    import uuid

    import ad_mc_sim as _pkg
    uid = getattr(_pkg, "_phase3_worker_uid", None)
    if uid is None:
        uid = uuid.uuid4().hex[:12]
        _pkg._phase3_worker_uid = uid
    return uid


def _probe(pdf: pd.DataFrame) -> pd.DataFrame:
    import os as _os
    import socket as _socket
    import sys as _sys

    import numpy as _np
    import scipy as _scipy

    from ad_mc_sim import cell as _cell   # noqa: F401  -- imported to prove it resolves
    from ad_mc_sim import engine as _engine

    # a real call into the frozen engine, not just an import
    a, b = _engine.fit_beta_moments(0.05, 0.01)
    return pd.DataFrame({
        "grp": [int(pdf["grp"].iloc[0])],
        "host": [_socket.gethostname()],
        "pid": [_os.getpid()],
        "worker_uid": [_worker_uid()],
        "module_file": [_engine.__file__],
        "py": [_sys.version.split()[0]],
        "numpy": [_np.__version__],
        "scipy": [_scipy.__version__],
        "beta_a": [float(a)],
        "beta_b": [float(b)],
    })


try:
    probe_df = spark.range(0, 16).select((F.col("id") % 8).cast("int").alias("grp"))
    probe_out = (
        probe_df.groupBy("grp")
        .applyInPandas(_probe, schema=PROBE_SCHEMA)
        .toPandas()
        .sort_values("grp")
    )
    print(probe_out.to_string(index=False))
    probe_info = {
        "rows": int(len(probe_out)),
        "distinct_hosts": sorted(probe_out["host"].unique().tolist()),
        "distinct_pids": sorted(int(p) for p in probe_out["pid"].unique()),
        "distinct_worker_uids": sorted(probe_out["worker_uid"].unique().tolist()),
        "driver_pid": os.getpid(),
        "driver_host": socket.gethostname(),
        "sandbox_is_pid_namespaced": bool(
            set(probe_out["pid"].unique()) == {1}
            and set(probe_out["host"].unique()) == {""}
        ),
        "executor_module_file": probe_out["module_file"].iloc[0],
        "executor_python": probe_out["py"].iloc[0],
        "executor_numpy": probe_out["numpy"].iloc[0],
        "executor_scipy": probe_out["scipy"].iloc[0],
        "engine_call_ok": bool(np.allclose(probe_out["beta_a"], 0.05 * (0.05 * 0.95 / 1e-4 - 1))),
        "pid_differs_from_driver": bool(
            all(int(p) != os.getpid() for p in probe_out["pid"].unique())
        ),
    }
    RESULT["executor_probe"] = probe_info
    print("\nexecutor probe:", json.dumps(probe_info, indent=2))
    if len(probe_out) != 8:
        raise RuntimeError(f"probe returned {len(probe_out)} groups, expected 8")
except Exception as exc:  # noqa: BLE001
    finish("failed", stage="executor_probe", error=repr(exc),
           traceback=traceback.format_exc())


# COMMAND ----------

# --- 2. READ THE REFERENCE TABLES -------------------------------------------
# Everything the simulation is parameterised by comes FROM THE TABLES. Nothing
# below is a literal copied out of a Python module or an earlier report.

try:
    assumptions = (
        spark.table(TBL_ASSUMPTIONS)
        .select("channel_id", "mean_cvr", "std_cvr", "mean_cpc", "std_cpc",
                "mean_revenue_per_conversion", "std_revenue_per_conversion")
        .orderBy("channel_id")
        .toPandas()
    )
    channels = assumptions["channel_id"].tolist()
    k = len(channels)

    corr_long = spark.table(TBL_CVR_CORR).select(
        "channel_id_a", "channel_id_b", "correlation_coefficient"
    ).toPandas()
    lookup = {
        (a, b): float(v)
        for a, b, v in corr_long.itertuples(index=False, name=None)
    }
    corr = np.empty((k, k), dtype=float)
    for i, a in enumerate(channels):
        for j, b in enumerate(channels):
            if (a, b) not in lookup:
                raise RuntimeError(f"{TBL_CVR_CORR} is missing the ({a}, {b}) pair")
            corr[i, j] = lookup[(a, b)]

    scen_pdf = spark.table(TBL_SCENARIOS).select(
        "scenario_id", "cvr_multiplier", "cpc_multiplier", "revenue_multiplier"
    ).orderBy("scenario_id").toPandas()

    alloc_pdf = spark.table(TBL_ALLOCATIONS).select(
        "allocation_id", "channel_id", "spend_dollars", "total_budget"
    ).toPandas()

    # allocation_id -> {channel_id: dollars}. Small enough (21 x 5 floats) to
    # ride along in the UDF closure; carrying it as a Spark column would mean a
    # map<string,double> per row for no benefit.
    alloc_dollars: dict[str, dict[str, float]] = {}
    for aid, cid, dollars, _budget in alloc_pdf.itertuples(index=False, name=None):
        alloc_dollars.setdefault(str(aid), {})[str(cid)] = float(dollars)

    bad = {
        aid: sorted(set(channels) - set(d))
        for aid, d in alloc_dollars.items()
        if set(d) != set(channels)
    }
    if bad:
        raise RuntimeError(f"allocations missing channels: {bad}")

    moments = {
        "mean_cvr": assumptions["mean_cvr"].to_numpy(float).tolist(),
        "std_cvr": assumptions["std_cvr"].to_numpy(float).tolist(),
        "mean_cpc": assumptions["mean_cpc"].to_numpy(float).tolist(),
        "std_cpc": assumptions["std_cpc"].to_numpy(float).tolist(),
        "mean_rpc": assumptions["mean_revenue_per_conversion"].to_numpy(float).tolist(),
        "std_rpc": assumptions["std_revenue_per_conversion"].to_numpy(float).tolist(),
    }
    corr_list = corr.tolist()

    print(f"channels ({k}): {channels}")
    print(f"scenarios ({len(scen_pdf)}): {scen_pdf['scenario_id'].tolist()}")
    print(f"allocations ({len(alloc_dollars)})")
    RESULT["inputs"] = {
        "channels": channels,
        "n_scenarios": int(len(scen_pdf)),
        "n_allocations": int(len(alloc_dollars)),
        "corr_offdiag_mean": float(corr[~np.eye(k, dtype=bool)].mean()),
    }
except Exception as exc:  # noqa: BLE001
    finish("failed", stage="read_reference", error=repr(exc),
           traceback=traceback.format_exc())


# COMMAND ----------

# --- 3. BUILD THE CELL GRID, WITH SEEDS COMPUTED IN THE DRIVER ---------------
# Seeds are derived here, on the driver, from `ad_mc_sim.seeding.seed_for_cell`
# and carried as a COLUMN. They are therefore part of the job's own data
# lineage: every simulated row can be traced back to the exact 64-bit seed that
# produced it without re-running the derivation. Recomputing the seed inside the
# UDF would work too, but it would hide the one input that makes the run
# reproducible.
#
# seed is carried as a STRING. The derived seeds are UNSIGNED 64-bit (the anchor
# cell's is 12283380380621132216, which is above 2^63-1), and Spark's BIGINT is
# SIGNED -- so a LongType column would overflow. DECIMAL(20,0) would also work;
# a decimal string is exact, obvious in the table, and converts with int().

try:
    from ad_mc_sim import seeding

    scen_mults = {
        str(r.scenario_id): (float(r.cvr_multiplier), float(r.cpc_multiplier),
                             float(r.revenue_multiplier))
        for r in scen_pdf.itertuples(index=False)
    }
    allocation_ids = sorted(alloc_dollars)
    scenario_ids = sorted(scen_mults)

    seed_map = seeding.seed_grid(allocation_ids, scenario_ids)  # asserts distinctness
    grid_rows = []
    for aid in allocation_ids:
        for sid in scenario_ids:
            cvr_m, cpc_m, rev_m = scen_mults[sid]
            grid_rows.append((aid, sid, str(seed_map[(aid, sid)]),
                              cvr_m, cpc_m, rev_m, N_PATHS))

    grid = spark.createDataFrame(
        grid_rows,
        "allocation_id string, scenario_id string, seed string, "
        "cvr_multiplier double, cpc_multiplier double, revenue_multiplier double, "
        "n_paths int",
    )
    n_cells = grid.count()
    print(f"grid: {n_cells} cells = {len(allocation_ids)} allocations "
          f"x {len(scenario_ids)} scenarios; {n_cells * N_PATHS:,} paths total")
    print(f"anchor derived seed (even_split, normal): {seed_map[('even_split', 'normal')]}")
    RESULT["grid"] = {
        "n_cells": int(n_cells),
        "n_paths_per_cell": N_PATHS,
        "total_paths": int(n_cells) * N_PATHS,
        "anchor_derived_seed": str(seed_map[("even_split", "normal")]),
        "distinct_seeds": len({v for v in seed_map.values()}),
    }
except Exception as exc:  # noqa: BLE001
    finish("failed", stage="build_grid", error=repr(exc),
           traceback=traceback.format_exc())


# COMMAND ----------

# --- 4. THE UDF -------------------------------------------------------------
# One Spark group == one (allocation, scenario) cell == one 10,000-path batch.
# The closure carries only small, constant reference data (moments, correlation,
# per-allocation dollars); everything cell-specific arrives as a column.

CELL_SCHEMA = (
    "allocation_id STRING, scenario_id STRING, path_number INT, "
    "total_simulated_revenue DOUBLE, total_spend DOUBLE, roas DOUBLE"
)

_channels = channels
_moments = moments
_corr_list = corr_list
_alloc_dollars = alloc_dollars


def simulate_group(pdf: pd.DataFrame) -> pd.DataFrame:
    """applyInPandas body: run one cell through the frozen Phase 2 engine."""
    from ad_mc_sim import cell as cell_mod

    if pdf.empty:
        return cell_mod.empty_cell_frame()
    if len(pdf) != 1:
        raise RuntimeError(
            f"expected exactly one grid row per group, got {len(pdf)} for "
            f"{pdf['allocation_id'].iloc[0]!r}/{pdf['scenario_id'].iloc[0]!r}"
        )

    row = pdf.iloc[0]
    allocation_id = str(row["allocation_id"])
    scenario_id = str(row["scenario_id"])

    return cell_mod.simulate_cell(
        allocation_id=allocation_id,
        scenario_id=scenario_id,
        channels=_channels,
        allocation=_alloc_dollars[allocation_id],
        mean_cvr=_moments["mean_cvr"], std_cvr=_moments["std_cvr"],
        mean_cpc=_moments["mean_cpc"], std_cpc=_moments["std_cpc"],
        mean_rpc=_moments["mean_rpc"], std_rpc=_moments["std_rpc"],
        correlation=_corr_list,
        cvr_multiplier=float(row["cvr_multiplier"]),
        cpc_multiplier=float(row["cpc_multiplier"]),
        revenue_multiplier=float(row["revenue_multiplier"]),
        n_paths=int(row["n_paths"]),
        seed=int(row["seed"]),          # decimal string -> unsigned 64-bit int
    )


outcomes = (
    grid.repartition(SHUFFLE_PARTITIONS, "allocation_id", "scenario_id")
    .groupBy("allocation_id", "scenario_id")
    .applyInPandas(simulate_group, schema=CELL_SCHEMA)
)


# COMMAND ----------

# --- 5. WRITE ---------------------------------------------------------------
# INSERT OVERWRITE, not append. `simulated_allocation_outcomes` has no run_id
# column, so an append would stack a second 840,000 rows on top of the first and
# every Phase 4 aggregate would silently average two runs together. The cell
# seeds are a pure function of (allocation_id, scenario_id), so a re-run
# reproduces the same 840,000 rows exactly -- which makes full replacement the
# honest semantics: the table always holds exactly one complete grid.
#
# INSERT OVERWRITE rather than df.write.mode("overwrite").saveAsTable(): it is a
# single atomic Delta commit that replaces the DATA and leaves the TABLE
# DEFINITION alone. saveAsTable can rewrite table metadata, which would discard
# the NOT NULL constraints, the column comments and the
# delta.autoOptimize.optimizeWrite property that 06_create_phase3_tables.py
# deliberately set.

try:
    write_info = {"mode": WRITE_MODE}
    if WRITE_MODE == "none":
        print("write_mode=none -- skipping the write")
    else:
        outcomes.createOrReplaceTempView("phase3_outcomes")
        verb = "INSERT OVERWRITE TABLE" if WRITE_MODE == "overwrite" else "INSERT INTO"
        t0 = time.time()
        spark.sql(f"""
            {verb} {TBL_OUTCOMES}
            SELECT allocation_id, scenario_id, path_number,
                   total_simulated_revenue, total_spend, roas
            FROM phase3_outcomes
        """)
        write_info["seconds"] = round(time.time() - t0, 2)
        write_info["rows_after"] = int(
            spark.sql(f"SELECT COUNT(*) AS n FROM {TBL_OUTCOMES}").collect()[0]["n"]
        )
        print(f"{verb} took {write_info['seconds']}s -> {write_info['rows_after']:,} rows")
    RESULT["write"] = write_info
except Exception as exc:  # noqa: BLE001
    finish("failed", stage="write", error=repr(exc), traceback=traceback.format_exc())


# COMMAND ----------

# --- 6. ANCHOR SIDE TEST AT PHASE 2'S PINNED SEED ---------------------------
# NOT a row in the grid. The grid's (even_split, normal) cell uses its DERIVED
# seed, so it is a different sample from the same distribution. This runs the
# same cell at Phase 2's pinned seed 20260808, through the same applyInPandas
# code path on an executor, and returns the whole 10,000-value revenue vector so
# the caller can diff it against the Phase 2 vector element by element.
#
# Expect a small discrepancy rather than bitwise equality: NumPy's PCG64 bit
# stream is version-stable, so the underlying uniforms should match, but
# scipy.stats.beta.ppf is a special-function evaluation and the cluster is on
# scipy 1.15.1 against Phase 2's 1.18.0.

try:
    anchor_grid = spark.createDataFrame(
        [("even_split", "normal", str(ANCHOR_SEED), 1.0, 1.0, 1.0, N_PATHS)],
        "allocation_id string, scenario_id string, seed string, "
        "cvr_multiplier double, cpc_multiplier double, revenue_multiplier double, "
        "n_paths int",
    )
    anchor = (
        anchor_grid.groupBy("allocation_id", "scenario_id")
        .applyInPandas(simulate_group, schema=CELL_SCHEMA)
        .orderBy("path_number")
        .toPandas()
    )
    rev = anchor["total_simulated_revenue"].to_numpy(np.float64)
    if len(rev) != N_PATHS:
        raise RuntimeError(f"anchor returned {len(rev)} paths, expected {N_PATHS}")

    var95 = float(np.percentile(rev, 5.0))
    tail = rev[rev <= var95]
    anchor_stats = {
        "seed": ANCHOR_SEED,
        "n": int(len(rev)),
        "mean": float(rev.mean()),
        "std": float(rev.std(ddof=1)),
        "var95": var95,
        "cvar95": float(tail.mean()),
        "min": float(rev.min()),
        "max": float(rev.max()),
        "total_spend": float(anchor["total_spend"].iloc[0]),
    }
    print(json.dumps(anchor_stats, indent=2))

    # Full vector out, two ways, so the comparison is never blocked on one of
    # them: a .npy on a UC volume, and a gzip+base64 blob in the exit payload.
    try:
        os.makedirs(os.path.dirname(ANCHOR_OUT_PATH), exist_ok=True)
        with open(ANCHOR_OUT_PATH, "wb") as fh:
            np.save(fh, rev)
        anchor_stats["npy_path"] = ANCHOR_OUT_PATH
    except Exception as exc:  # noqa: BLE001
        anchor_stats["npy_path"] = f"FAILED: {exc!r}"

    blob = base64.b64encode(gzip.compress(rev.tobytes(), 6)).decode("ascii")
    anchor_stats["vector_b64_bytes"] = len(blob)
    RESULT["anchor"] = anchor_stats
    RESULT["anchor_vector_gz_b64"] = blob
except Exception as exc:  # noqa: BLE001
    finish("failed", stage="anchor", error=repr(exc), traceback=traceback.format_exc())


# COMMAND ----------

# --- 7. DIAGNOSTICS: DID THIS ACTUALLY RUN IN PARALLEL? ---------------------
# The write above hides its own parallelism -- the rows land in Delta and the
# task layout is gone. So the identical grid is run a second time through an
# applyInPandas that does the SAME simulate_cell work per cell but returns ONE
# summary row carrying where and when it ran. Wall-clock of this pass, the set
# of distinct worker PROCESSES, and the concurrency implied by overlapping
# [t_start, t_end] intervals are then real measurements of the same workload,
# not an assertion about it.
#
# Worker identity comes from `_worker_uid()`, NOT from (hostname, pid) -- see
# that function for why: the sandbox is PID-namespaced and every worker claims
# pid 1 with an empty hostname. host and pid are still recorded, purely to
# document that.
#
# Honest framing: 84 cells x 10,000 paths is ~11s of single-threaded numpy. The
# distribution here demonstrates the mechanism and would matter at 100x the
# grid; it is not load-bearing at this size.

DIAG_SCHEMA = (
    "allocation_id string, scenario_id string, host string, pid int, "
    "worker_uid string, partition_id int, t_start double, t_end double, "
    "elapsed double, n int, mean double, std double, var95 double, "
    "cvar95 double, spend double, py string, numpy string, scipy string"
)


def diagnose_group(pdf: pd.DataFrame) -> pd.DataFrame:
    import os as _os
    import socket as _socket
    import sys as _sys
    import time as _time

    import numpy as _np
    import scipy as _scipy

    t0 = _time.time()
    out = simulate_group(pdf)
    t1 = _time.time()

    part = -1
    try:
        from pyspark import TaskContext
        ctx = TaskContext.get()
        if ctx is not None:
            part = int(ctx.partitionId())
    except Exception:  # noqa: BLE001
        pass

    rev = out["total_simulated_revenue"].to_numpy(_np.float64)
    v = float(_np.percentile(rev, 5.0))
    tail = rev[rev <= v]
    return pd.DataFrame({
        "allocation_id": [str(pdf["allocation_id"].iloc[0])],
        "scenario_id": [str(pdf["scenario_id"].iloc[0])],
        "host": [_socket.gethostname()],
        "pid": [int(_os.getpid())],
        "worker_uid": [_worker_uid()],
        "partition_id": [part],
        "t_start": [t0], "t_end": [t1], "elapsed": [t1 - t0],
        "n": [int(len(rev))],
        "mean": [float(rev.mean())],
        "std": [float(rev.std(ddof=1))],
        "var95": [v],
        "cvar95": [float(tail.mean())],
        "spend": [float(out["total_spend"].iloc[0])],
        "py": [_sys.version.split()[0]],
        "numpy": [_np.__version__],
        "scipy": [_scipy.__version__],
    })


try:
    t0 = time.time()
    diag = (
        grid.repartition(SHUFFLE_PARTITIONS, "allocation_id", "scenario_id")
        .groupBy("allocation_id", "scenario_id")
        .applyInPandas(diagnose_group, schema=DIAG_SCHEMA)
        .withColumn("out_partition_id", F.spark_partition_id())
        .toPandas()
    )
    diag_seconds = time.time() - t0

    # concurrency: max number of cells whose [t_start, t_end] overlap
    events = sorted([(float(r.t_start), 1) for r in diag.itertuples()] +
                    [(float(r.t_end), -1) for r in diag.itertuples()])
    cur = peak = 0
    for _t, delta in events:
        cur += delta
        peak = max(peak, cur)

    cells_per_worker = diag.groupby("worker_uid").size().sort_values(ascending=False)
    span = float(diag["t_end"].max() - diag["t_start"].min())
    cpu = float(diag["elapsed"].sum())
    parallel = {
        "diag_wall_clock_s": round(diag_seconds, 2),
        "task_span_s": round(span, 2),
        "sum_of_cell_cpu_s": round(cpu, 2),
        "speedup_vs_serial": round(cpu / span, 2) if span > 0 else None,
        "peak_concurrent_cells": int(peak),
        "n_distinct_workers": int(len(cells_per_worker)),
        "cells_per_worker": {str(k): int(v) for k, v in cells_per_worker.items()},
        "namespaced_pids_seen": sorted(int(p) for p in diag["pid"].unique()),
        "namespaced_hosts_seen": sorted(str(h) for h in diag["host"].unique()),
        "n_out_partitions_with_rows": int(diag["out_partition_id"].nunique()),
        "n_udf_partition_ids": int(diag["partition_id"].nunique()),
        "cells": int(len(diag)),
        "executor_python": diag["py"].iloc[0],
        "executor_numpy": diag["numpy"].iloc[0],
        "executor_scipy": diag["scipy"].iloc[0],
        "median_cell_seconds": round(float(diag["elapsed"].median()), 3),
    }
    print(json.dumps(parallel, indent=2))
    RESULT["parallelism"] = parallel

    # Explicit python scalars -- json.dumps cannot serialise numpy types, and a
    # `default=str` fallback would silently turn every metric into a string.
    RESULT["cell_summary"] = [
        {
            "allocation_id": str(r.allocation_id),
            "scenario_id": str(r.scenario_id),
            "n": int(r.n),
            "mean": float(r.mean),
            "std": float(r.std),
            "var95": float(r.var95),
            "cvar95": float(r.cvar95),
            "spend": float(r.spend),
            "worker_uid": str(r.worker_uid),
            "partition_id": int(r.partition_id),
            "out_partition_id": int(r.out_partition_id),
            "elapsed": round(float(r.elapsed), 4),
        }
        for r in diag.sort_values(["allocation_id", "scenario_id"]).itertuples(index=False)
    ]
except Exception as exc:  # noqa: BLE001
    finish("failed", stage="diagnostics", error=repr(exc), traceback=traceback.format_exc())


# COMMAND ----------

finish("ok")
