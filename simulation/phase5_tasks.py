"""Phase 5 -- the five Databricks Workflow task bodies, and their shared plumbing.

WHAT THIS IS
------------
Phase 4 ran the two-stage sweep LOCALLY (`load_phase4_gold.py`), single
threaded, writing gold over the SQL API. Phase 5 runs the SAME sweep as a
persisted Databricks Workflow. This module holds the task bodies; the five files
in `simulation/phase5_notebooks/` are thin notebook shells that read widgets,
call one function here, publish a task value and exit.

The code lives in a MODULE rather than in the notebooks because the notebooks
are uploaded as workspace objects while this ships inside the `ad_mc_sim` wheel
-- the same wheel the Spark executors install. Putting the UDF body in a
notebook cell would mean the driver and the executors were running code from two
different distribution mechanisms.

NO MODELLING CHANGES. Every number this produces comes from `frontier`,
`cell`, `engine`, `saturation`, `scenarios`, `seeding` and `sweep_seeding`
exactly as Phase 4 committed them. The gold frames are built by
`gold_assembly`, which the local Phase 4 loader also calls. What is new here is
orchestration: parameters, task boundaries, intermediate state, Spark
distribution and MLflow.

WHY THE DAG HAS THE SHAPE IT HAS
--------------------------------
The search is ADAPTIVE: stage 2 refines around stage 1's computed Pareto union.
That is a genuine dependency edge, not a convenience, and it is why this cannot
be one flat parallel job. It also forces intermediate state across a process
boundary, since Workflow tasks are separate processes.

    bronze_refresh -> stage1_simulate -> stage2_generate -> stage2_simulate
                                     \\                                    \\
                                      +--------------------------------> gold_aggregate

`gold_aggregate` depends on stage1_simulate as well as stage2_simulate because
it merges BOTH stages' results; expressing that edge means the DAG, not a
convention, records where its inputs come from.

INTERMEDIATE STATE: PARQUET ON A UC VOLUME
------------------------------------------
Under `/Volumes/<catalog>/bronze/landing/phase5/run_<job_run_id>/`. Parquet
because it stores float64 as IEEE754 bytes, so the round trip is exact -- and
exactness is the whole point here, since a seed or a spend that changed in the
last ulb between tasks would silently change the answer. JSON would need a
float-repr contract; CSV would not round-trip at all.

A UC VOLUME rather than a Delta table on purpose: this is scratch, and a Delta
table in silver/gold would be indistinguishable from real output to anything
that came along later. Scoping the directory by `job_run_id` also means two runs
never read each other's state.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

try:  # package-style import (ad_mc_sim.phase5_tasks)
    from . import bronze_sql
    from . import frontier as F
    from . import gold_assembly as GA
    from . import saturation as SAT
    from . import scenarios as scenarios_mod
    from .frontier import Candidate, Stage1Config, Stage2Config, SweepInputs
    from .scenarios import SCENARIOS
    from .sweep_seeding import CRN_STREAM_LABEL, crn_seed_plan
except ImportError:  # flat import, which is how the rest of this repo runs
    import bronze_sql
    import frontier as F
    import gold_assembly as GA
    import saturation as SAT
    import scenarios as scenarios_mod
    from frontier import Candidate, Stage1Config, Stage2Config, SweepInputs
    from scenarios import SCENARIOS
    from sweep_seeding import CRN_STREAM_LABEL, crn_seed_plan


# =============================================================================
# Parameters
# =============================================================================

# (widget name, description). Defaults are DELIBERATELY absent: every one of
# these arrives from a Databricks JOB PARAMETER, and an empty widget makes the
# task fail loudly rather than quietly running on a notebook-local default that
# nobody set. The real defaults live in the job definition
# (simulation/run_phase5_workflow.py::JOB_PARAMETERS) so they are visible and
# editable in the Workflows UI, which is the point of parameterising at all.
WIDGETS: tuple[tuple[str, str], ...] = (
    ("total_budget", "Total budget in dollars, spread across the channels"),
    ("master_seed", "Master seed; every derived stream hangs off this"),
    ("n_paths", "Monte Carlo paths per (allocation, scenario) cell"),
    ("stage1_dirichlet_total", "Total Dirichlet draws in stage 1 (450 -> 571 candidates)"),
    ("stage2_cap", "Cap on stage-2 candidates, applied per family"),
    ("catalog", "Unity Catalog catalog holding bronze/silver/gold"),
    ("mlflow_experiment", "Workspace path of the ONE MLflow experiment"),
    ("job_run_id", "{{job.run_id}} -- scopes the scratch volume directory"),
)


@dataclass(frozen=True)
class Params:
    total_budget: float
    master_seed: int
    n_paths: int
    stage1_dirichlet_total: int
    stage2_cap: int
    catalog: str
    mlflow_experiment: str
    job_run_id: str

    @property
    def scratch_dir(self) -> str:
        return f"/Volumes/{self.catalog}/bronze/landing/phase5/run_{self.job_run_id}"

    def path(self, name: str) -> str:
        return f"{self.scratch_dir}/{name}"


def resolve_params(raw: dict[str, str]) -> Params:
    """Widget strings -> typed Params. Raises on anything missing or unparsed."""
    missing = [n for n, _ in WIDGETS if not str(raw.get(n, "")).strip()]
    if missing:
        raise ValueError(
            f"job parameters not supplied: {missing}. These are declared as JOB "
            f"parameters on the Workflow and referenced by each task; an empty "
            f"value means the reference did not resolve."
        )
    unresolved = [n for n, _ in WIDGETS if "{{" in str(raw[n])]
    if unresolved:
        raise ValueError(
            f"parameters arrived as literal templates, not values: "
            f"{ {n: raw[n] for n in unresolved} }"
        )
    p = Params(
        total_budget=float(raw["total_budget"]),
        master_seed=int(raw["master_seed"]),
        n_paths=int(raw["n_paths"]),
        stage1_dirichlet_total=int(raw["stage1_dirichlet_total"]),
        stage2_cap=int(raw["stage2_cap"]),
        catalog=str(raw["catalog"]).strip(),
        mlflow_experiment=str(raw["mlflow_experiment"]).strip(),
        job_run_id=str(raw["job_run_id"]).strip(),
    )
    if p.total_budget <= 0:
        raise ValueError(f"total_budget must be > 0, got {p.total_budget}")
    if p.n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {p.n_paths}")
    if p.stage1_dirichlet_total < 0:
        raise ValueError(f"stage1_dirichlet_total must be >= 0, got {p.stage1_dirichlet_total}")
    if p.stage2_cap <= 0:
        raise ValueError(f"stage2_cap must be > 0, got {p.stage2_cap}")
    return p


def stage1_config_for(dirichlet_total: int) -> Stage1Config:
    """Scale `Stage1Config.dirichlet` to a total, preserving the default MIX.

    The stage-1 set is 21 Phase 3 candidates + a centroid + two simplex lattices
    + three Dirichlet blocks. Only the Dirichlet counts are worth exposing: the
    structured part is a fixed enumeration (a {5,3} and a {5,4} lattice are 35
    and 70 points, not a tunable), so a "how many candidates" knob would be
    lying about what it controls.

    The three alphas keep their default RATIO (225 : 125 : 100) and are
    apportioned by largest remainder, so the default total of 450 reproduces
    (225, 125, 100) EXACTLY -- asserted below, because a parameter whose default
    changes the candidate set would invalidate every comparison against the
    Phase 4 run.
    """
    base = Stage1Config()
    weights = [count for _alpha, count in base.dirichlet]
    base_total = sum(weights)
    if dirichlet_total == base_total:
        scaled = list(weights)
    else:
        exact = [dirichlet_total * wgt / base_total for wgt in weights]
        scaled = [int(math.floor(v)) for v in exact]
        remainder = dirichlet_total - sum(scaled)
        # Largest fractional part first; ties by position, so it is deterministic.
        order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - scaled[i]), i))
        for i in order[:remainder]:
            scaled[i] += 1
    if dirichlet_total == base_total and tuple(scaled) != tuple(weights):
        raise AssertionError(
            f"default apportionment changed the candidate set: {scaled} != {weights}"
        )
    return Stage1Config(
        include_phase3=base.include_phase3,
        lattice_degrees=base.lattice_degrees,
        include_centroid=base.include_centroid,
        dirichlet=tuple(
            (alpha, int(n)) for (alpha, _old), n in zip(base.dirichlet, scaled)
        ),
    )


# =============================================================================
# Inputs, read from the tables the pipeline just refreshed
# =============================================================================

ASSUMPTION_COLUMNS = ["channel_id", "mean_cvr", "std_cvr", "mean_cpc", "std_cpc",
                      "mean_revenue_per_conversion", "std_revenue_per_conversion"]


def load_inputs(spark, p: Params) -> tuple[SweepInputs, list[str]]:
    """`SweepInputs` from live bronze -- the Spark twin of load_phase4_gold.build_inputs.

    Same tables, same channel ORDER (sorted by channel_id, which is what
    data_access.load_assumptions does), same theta and spend-floor vectors from
    `saturation`. Read through Spark rather than the Statement Execution API
    because this runs on the job compute, where reaching back out to a warehouse
    would be the wrong way round.
    """
    a = (spark.table(f"{p.catalog}.bronze.channel_assumptions")
         .select(*ASSUMPTION_COLUMNS).orderBy("channel_id").toPandas())
    ch = [str(c) for c in a["channel_id"].tolist()]
    if len(ch) < 2:
        raise RuntimeError(f"expected >= 2 channels in bronze, got {ch}")

    corr_long = (spark.table(f"{p.catalog}.bronze.channel_cvr_correlation_matrix")
                 .select("channel_id_a", "channel_id_b", "correlation_coefficient")
                 .toPandas())
    lookup = {(str(x), str(y)): float(v)
              for x, y, v in corr_long.itertuples(index=False, name=None)}
    k = len(ch)
    corr = np.empty((k, k), dtype=float)
    for i, x in enumerate(ch):
        for j, y in enumerate(ch):
            if (x, y) not in lookup:
                raise RuntimeError(f"cvr correlation matrix is missing the ({x}, {y}) pair")
            corr[i, j] = lookup[(x, y)]

    inputs = SweepInputs(
        channels=tuple(ch),
        mean_cvr=a["mean_cvr"].to_numpy(float), std_cvr=a["std_cvr"].to_numpy(float),
        mean_cpc=a["mean_cpc"].to_numpy(float), std_cpc=a["std_cpc"].to_numpy(float),
        mean_rpc=a["mean_revenue_per_conversion"].to_numpy(float),
        std_rpc=a["std_revenue_per_conversion"].to_numpy(float),
        correlation=corr,
        theta=SAT.theta_vector(ch, dict(SAT.DEFAULT_THETA)),
        spend_floor=SAT.spend_floor_vector(ch),
        total_budget=p.total_budget,
        n_paths=p.n_paths,
    )
    return inputs, ch


def inputs_fingerprint(inputs: SweepInputs) -> str:
    """sha256 over the exact float64 bytes of everything a cell is priced with.

    Each task re-reads bronze for itself rather than shipping `SweepInputs`
    between tasks, which is simpler and avoids a serialisation contract -- but it
    means a bronze change mid-run would split the sweep across two worlds. The
    fingerprint is how that gets caught instead of averaged.
    """
    h = hashlib.sha256()
    h.update("|".join(inputs.channels).encode())
    for name in ("mean_cvr", "std_cvr", "mean_cpc", "std_cpc", "mean_rpc", "std_rpc",
                 "correlation", "theta", "spend_floor"):
        arr = getattr(inputs, name)
        h.update(name.encode())
        h.update(b"None" if arr is None else np.ascontiguousarray(arr, dtype=float).tobytes())
    h.update(repr((inputs.total_budget, inputs.n_paths, inputs.reference_spend,
                   inputs.saturate_std_cpc)).encode())
    return h.hexdigest()[:16]


# =============================================================================
# Scratch state on the volume
# =============================================================================

def _write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def write_parquet(pdf: pd.DataFrame, path: str) -> int:
    """Serialise in memory, then one open()/write() to the volume.

    Buffered rather than handing the path straight to `to_parquet` so the only
    thing touching the FUSE mount is a plain sequential write of a finished
    byte string -- the same call Phase 3 used for its .npy artefact.

    `attrs` IS CLEARED, and that is not cosmetic. `DataFrame.attrs` is written
    into the parquet key-value metadata as JSON, and Databricks attaches a
    `PlanMetrics` object to every frame that came out of `toPandas()` -- so the
    first real run died with `Object of type PlanMetrics is not JSON
    serializable` after finishing the entire 22.8M-path sweep. The attrs carry
    query-plan telemetry, nothing this pipeline uses.
    """
    out = pdf.copy()
    out.attrs = {}
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    data = buf.getvalue()
    _write_bytes(path, data)
    return len(data)


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_read_bytes(path)))


def candidates_to_frame(candidates: Sequence[Candidate], channels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        row = {"allocation_id": c.allocation_id, "family": c.family, "stage": int(c.stage),
               "parents_json": json.dumps(list(c.parents))}
        for ch, s in zip(channels, c.spend):
            row[f"spend_{ch}"] = float(s)
        rows.append(row)
    return pd.DataFrame(rows)


def frame_to_candidates(df: pd.DataFrame, channels: Sequence[str]) -> list[Candidate]:
    out = []
    for r in df.to_dict("records"):
        out.append(Candidate(
            allocation_id=str(r["allocation_id"]), family=str(r["family"]),
            stage=int(r["stage"]),
            spend=tuple(float(r[f"spend_{ch}"]) for ch in channels),
            parents=tuple(json.loads(r["parents_json"])),
        ))
    return out


# =============================================================================
# The distributed sweep
# =============================================================================

_SPARK_TYPES = {
    "allocation_id": "STRING", "scenario_id": "STRING", "n_paths": "INT",
    "seed": "STRING", "total_spend": "DOUBLE", "mean_revenue": "DOUBLE",
    "std_revenue": "DOUBLE", "se_mean_revenue": "DOUBLE", "min_revenue": "DOUBLE",
    "max_revenue": "DOUBLE", "median_revenue": "DOUBLE", "var_95": "DOUBLE",
    "cvar_95": "DOUBLE", "expected_roas": "DOUBLE",
    "extrapolation_floor_applied": "BOOLEAN", "n_channels_floored": "INT",
    "prob_below_breakeven": "DOUBLE",
}

_missing_types = [c for c in F.SWEEP_OUTPUT_COLUMNS if c not in _SPARK_TYPES]
if _missing_types:  # pragma: no cover -- import-time guard
    raise ImportError(
        f"frontier.SWEEP_OUTPUT_COLUMNS grew columns with no Spark type here: "
        f"{_missing_types}. Add them rather than letting applyInPandas drop them."
    )

# Built from SWEEP_OUTPUT_COLUMNS so the schema cannot drift from the dict the
# UDF returns. (frontier.SWEEP_OUTPUT_SPARK_SCHEMA_DDL predates the spend floor
# and is missing its two columns; it is left alone -- frontier.py is frozen for
# this phase.)
SWEEP_SPARK_SCHEMA: str = ", ".join(
    f"{c} {_SPARK_TYPES[c]}" for c in F.SWEEP_OUTPUT_COLUMNS
)


def build_plan(candidates: Sequence[Candidate], channels: Sequence[str],
               master_seed: int, n_chunks: int,
               scenarios=SCENARIOS) -> pd.DataFrame:
    """One row per (scenario, candidate) cell, seeds resolved on the driver.

    Scenario-major, matching `frontier.run_sweep`'s loop, so the plan's row order
    is already the canonical order and the seeds are visible in the job's own
    lineage rather than being recomputed inside a UDF.

    COMMON RANDOM NUMBERS: the seed is a function of the SCENARIO only, so every
    allocation in a scenario shares one stream and the frontier's pairwise
    comparisons are paired. `crn_seed_plan` asserts the four scenarios did not
    collide onto one seed, which would silently pair the scenarios too.
    """
    plan = crn_seed_plan([s.scenario_id for s in scenarios], master_seed)
    rows = []
    for scenario in scenarios:
        seed = plan[scenario.scenario_id]
        for cand in candidates:
            row = {"allocation_id": cand.allocation_id,
                   "scenario_id": scenario.scenario_id,
                   "seed": str(seed)}
            for ch, s in zip(channels, cand.spend):
                row[f"spend_{ch}"] = float(s)
            rows.append(row)
    for i, row in enumerate(rows):
        row["chunk_id"] = i % max(1, n_chunks)
    return pd.DataFrame(rows)


def chunks_for(n_cells: int, cells_per_chunk: int = 32, max_chunks: int = 200) -> int:
    """Chunk count. One Spark task per CHUNK, not per cell.

    Phase 3 grouped by (allocation, scenario) because it had 84 groups. The sweep
    has 2,284 and 1,600, and one task per cell would spend more time scheduling
    than simulating -- each cell is only ~75 ms. Chunking amortises that while
    keeping enough tasks to fill the autoscaled compute.
    """
    if n_cells <= 0:
        raise ValueError("no cells to simulate")
    return max(1, min(max_chunks, math.ceil(n_cells / max(1, cells_per_chunk))))


def _make_chunk_udf(inputs: SweepInputs, scenarios=SCENARIOS) -> Callable:
    scenarios_by_id = {s.scenario_id: s for s in scenarios}

    def simulate_chunk(pdf: pd.DataFrame) -> pd.DataFrame:
        # Imported INSIDE the UDF: this body executes in the executor's Python
        # sandbox, where the driver's module objects do not exist. The wheel is
        # installed through the serverless environment spec, so the import is the
        # same one the driver did -- Phase 3's probe is what established that.
        try:
            from ad_mc_sim import frontier as _F
        except ImportError:  # local reference run, flat imports
            import frontier as _F

        rows = [_F.summarize_cell_row(r, inputs, scenarios_by_id)
                for r in pdf.to_dict("records")]
        out = pd.DataFrame(rows, columns=list(_F.SWEEP_OUTPUT_COLUMNS))
        out["seed"] = out["seed"].map(str)
        out["n_paths"] = out["n_paths"].astype("int32")
        out["n_channels_floored"] = out["n_channels_floored"].astype("int32")
        out["extrapolation_floor_applied"] = out["extrapolation_floor_applied"].astype(bool)
        return out

    return simulate_chunk


ENV_PROBE_SCHEMA = ("slot INT, python STRING, numpy STRING, scipy STRING, "
                    "pandas STRING, wheel STRING, worker_uid STRING")


def executor_environment(spark, n: int = 8) -> list[dict]:
    """What the UDF workers are actually running, measured not assumed.

    Two things this pins down that nothing else does. First, the WHEEL VERSION
    resolved on the executor -- the version string carries the content hash of
    the packaged modules, so it proves which code drew the numbers. Second the
    numpy/scipy versions, which is the only defensible way to attribute a
    last-ulp difference against a local run to the environment rather than to a
    code change.

    Driver versions would not do: importing on the driver proves nothing about
    the Python UDF sandbox, which is the failure Phase 3's probe was built for.
    """
    def probe(pdf: pd.DataFrame) -> pd.DataFrame:
        import sys as _sys
        import uuid as _uuid

        import numpy as _np
        import pandas as _pd
        try:
            import scipy as _scipy
            scipy_v = _scipy.__version__
        except Exception as exc:  # noqa: BLE001
            scipy_v = f"MISSING: {exc!r}"
        try:
            from importlib.metadata import version as _version
            wheel = _version("ad_mc_sim")
        except Exception as exc:  # noqa: BLE001
            wheel = f"UNKNOWN: {exc!r}"
        try:
            # PID and hostname are namespaced to identical values in the
            # serverless UDF sandbox (Phase 3 measured this), so identity is
            # stashed on the cached module object instead.
            import ad_mc_sim as _pkg
            uid = getattr(_pkg, "_p5_worker_uid", None)
            if uid is None:
                uid = _uuid.uuid4().hex[:8]
                _pkg._p5_worker_uid = uid
        except Exception:  # noqa: BLE001
            uid = "n/a"
        return _pd.DataFrame([{
            "slot": int(pdf["slot"].iloc[0]), "python": _sys.version.split()[0],
            "numpy": _np.__version__, "scipy": scipy_v, "pandas": _pd.__version__,
            "wheel": wheel, "worker_uid": uid,
        }])

    try:
        frame = spark.createDataFrame(pd.DataFrame({"slot": list(range(n))}))
        return (frame.repartition(n, "slot").groupBy("slot")
                .applyInPandas(probe, schema=ENV_PROBE_SCHEMA)
                .toPandas().sort_values("slot").to_dict("records"))
    except Exception as exc:  # noqa: BLE001 -- diagnostics must never fail a run
        return [{"error": repr(exc)}]


def simulate_distributed(spark, candidates: Sequence[Candidate], inputs: SweepInputs,
                         p: Params, scenarios=SCENARIOS) -> tuple[pd.DataFrame, dict]:
    """`applyInPandas` over chunked cells. Returns (results, diagnostics)."""
    channels = list(inputs.channels)
    n_cells = len(candidates) * len(scenarios)
    n_chunks = chunks_for(n_cells)

    conf = {}
    for key, value in (("spark.sql.shuffle.partitions", str(n_chunks)),
                       ("spark.sql.adaptive.coalescePartitions.enabled", "false")):
        try:
            spark.conf.set(key, value)
            conf[key] = spark.conf.get(key)
        except Exception as exc:  # noqa: BLE001 -- reported, never assumed
            conf[key] = f"REFUSED: {type(exc).__name__}: {exc}"

    plan = build_plan(candidates, channels, p.master_seed, n_chunks, scenarios)
    plan_sdf = spark.createDataFrame(plan)

    t0 = time.time()
    out = (plan_sdf.repartition(n_chunks, "chunk_id")
           .groupBy("chunk_id")
           .applyInPandas(_make_chunk_udf(inputs, scenarios), schema=SWEEP_SPARK_SCHEMA)
           .toPandas())
    elapsed = time.time() - t0

    if len(out) != n_cells:
        raise RuntimeError(f"sweep returned {len(out)} rows, expected {n_cells}")
    diag = {"n_cells": n_cells, "n_chunks": n_chunks, "seconds": round(elapsed, 2),
            "paths": n_cells * p.n_paths, "spark_conf": conf,
            "distinct_seeds": int(out["seed"].nunique())}
    return out, diag


# =============================================================================
# MLflow
# =============================================================================

def start_or_resume(experiment_path: str, run_id: str | None, run_name: str | None = None):
    """Set the experiment and either create THE run or resume it.

    One experiment for the project, one run per Workflow run. Tasks are separate
    processes, so the run id is created in `bronze_refresh` and handed forward
    through `dbutils.jobs.taskValues`; every later task resumes the same run so
    the parameters, the metrics and the artefacts land in one place instead of
    five disconnected runs.
    """
    import mlflow

    exp = mlflow.set_experiment(experiment_path)
    if run_id:
        active = mlflow.start_run(run_id=run_id)
    else:
        active = mlflow.start_run(run_name=run_name)
    return mlflow, exp, active


def end_intermediate(mlflow) -> None:
    """Close the task's handle without declaring the pipeline finished."""
    try:
        mlflow.end_run(status="RUNNING")
    except Exception:  # noqa: BLE001 -- status vocabulary differs by version
        mlflow.end_run()


def mark_run_failed(experiment_path: str, run_id: str | None) -> bool:
    """Terminate the pipeline's MLflow run as FAILED. Returns whether it did.

    WHY THIS EXISTS. `end_intermediate` deliberately closes each task's handle
    with status RUNNING, because the NEXT task resumes the same run -- that is
    what keeps one Workflow run to one MLflow run instead of five disconnected
    ones. The cost is that an exception bypasses every end_run call, so a failed
    task leaves the run RUNNING forever. That is not merely untidy: the run
    still carries metrics, and the keys it carries overlap with the ones a
    successful run logs (`bronze__*`, `wall_clock_s__bronze_refresh`), so any
    aggregate over the experiment that does not filter on status silently mixes
    a partial run into the numbers. Job run 529952200505846 left exactly such a
    run behind.

    SWALLOWS ITS OWN ERRORS ON PURPOSE. This is called from an exception
    handler; if the tracking server is unreachable, the ORIGINAL failure is the
    one worth surfacing, and masking it with a bookkeeping error would be
    strictly worse. It returns False rather than raising so a caller can say so.
    """
    if not run_id:
        return False
    try:
        import mlflow

        mlflow.set_experiment(experiment_path)
        # An already-open handle in this process must be closed as FAILED;
        # otherwise resume the run just to terminate it.
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        else:
            mlflow.start_run(run_id=run_id)
            mlflow.end_run(status="FAILED")
        return True
    except Exception as exc:  # noqa: BLE001 -- never mask the real failure
        print(f"  WARNING: could not mark MLflow run {run_id} FAILED: {exc!r}")
        return False


def _metric_key(*parts: str) -> str:
    """MLflow metric name. Spaces are legal but make the UI awkward to filter."""
    return "__".join(str(x).strip().replace(" ", "_") for x in parts)


# =============================================================================
# Task 1 -- bronze_refresh
# =============================================================================

def run_bronze_refresh(spark, p: Params) -> dict:
    """Rebuild the derived bronze tables, then open the MLflow run for the pipeline.

    Idempotent on unchanged source: the three assumption/correlation tables are
    CREATE OR REPLACE from `channel_performance_history`, and
    `scenario_definitions` is a single MERGE that inserts, updates and deletes so
    the table ends up exactly equal to `scenarios.scenario_rows()`.

    The SQL is imported from `bronze_sql`, which `databricks/03_build_assumptions.py`
    also uses, so there is one definition of what `mean_cpc` means.
    """
    t0 = time.time()
    cat = p.catalog
    history = f"{cat}.bronze.channel_performance_history"
    tbl_assumptions = f"{cat}.bronze.channel_assumptions"
    tbl_corr = f"{cat}.bronze.channel_correlation_matrix"
    tbl_cvr = f"{cat}.bronze.channel_cvr_correlation_matrix"
    tbl_scen = f"{cat}.bronze.scenario_definitions"

    out: dict = {"task": "bronze_refresh"}

    spark.sql(bronze_sql.assumptions_statement(tbl_assumptions, history))
    spark.sql(bronze_sql.revenue_correlation_statement(tbl_corr, history))
    spark.sql(bronze_sql.cvr_correlation_statement(tbl_cvr, history))

    # --- scenario_definitions: one atomic MERGE from scenarios.py -------------
    # Same three-clause shape as databricks/07_load_phase3_reference.py, sourced
    # from a Spark view instead of a VALUES list. `net_revenue_factor` is dropped
    # on purpose: it is exactly cvr*rev/cpc of three columns that ARE stored, and
    # the table does not declare it.
    scen_cols = ["scenario_id", "scenario_name", "description",
                 "cvr_multiplier", "cpc_multiplier", "revenue_multiplier", "rationale"]
    scen_rows = [{c: r[c] for c in scen_cols} for r in scenarios_mod.scenario_rows()]
    spark.createDataFrame(pd.DataFrame(scen_rows)).createOrReplaceTempView("_p5_scenarios")
    set_clause = ", ".join(f"t.{c} = s.{c}" for c in scen_cols)
    spark.sql(f"""
        MERGE INTO {tbl_scen} t
        USING _p5_scenarios s
          ON t.scenario_id = s.scenario_id
        WHEN MATCHED THEN UPDATE SET {set_clause}
        WHEN NOT MATCHED THEN INSERT ({', '.join(scen_cols)})
             VALUES ({', '.join('s.' + c for c in scen_cols)})
        WHEN NOT MATCHED BY SOURCE THEN DELETE
    """)

    # --- data quality ---------------------------------------------------------
    checks: dict[str, object] = {}
    a = (spark.table(tbl_assumptions).select(*ASSUMPTION_COLUMNS)
         .orderBy("channel_id").toPandas())
    ch = [str(c) for c in a["channel_id"].tolist()]
    k = len(ch)
    checks["n_channels"] = k
    checks["n_scenarios"] = int(spark.table(tbl_scen).count())
    checks["n_corr_rows"] = int(spark.table(tbl_corr).count())
    checks["n_cvr_corr_rows"] = int(spark.table(tbl_cvr).count())
    if checks["n_corr_rows"] != k * k or checks["n_cvr_corr_rows"] != k * k:
        raise RuntimeError(f"correlation matrices are not {k}x{k}: {checks}")
    nulls = int(a.isna().to_numpy().sum())
    checks["null_assumption_cells"] = nulls
    if nulls:
        raise RuntimeError(f"{nulls} null cells in {tbl_assumptions}")

    corr_long = (spark.table(tbl_cvr)
                 .select("channel_id_a", "channel_id_b", "correlation_coefficient").toPandas())
    lookup = {(str(x), str(y)): float(v)
              for x, y, v in corr_long.itertuples(index=False, name=None)}
    corr = np.array([[lookup[(x, y)] for y in ch] for x in ch], dtype=float)
    checks["corr_symmetric_max_abs_diff"] = float(np.abs(corr - corr.T).max())
    try:  # positive-definite is what the Cholesky step actually needs
        np.linalg.cholesky(corr)
        checks["cvr_correlation_positive_definite"] = True
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"cvr correlation matrix is not positive definite: {exc}") from exc

    # The frozen snapshots in saturation.py must still describe live bronze.
    live_impr = {str(r["channel_id"]): float(r["avg_impr"]) for r in spark.sql(
        f"SELECT channel_id, AVG(impressions) AS avg_impr FROM {history} GROUP BY channel_id"
    ).collect()}
    checks["impressions_snapshot_max_rel"] = float(
        max(SAT.check_impressions_snapshot(live_impr).values()))
    q = float(SAT.BRONZE_SPEND_FLOOR_PERCENTILE)
    live_floor = {str(r["channel_id"]): float(r["p"]) for r in spark.sql(
        f"SELECT channel_id, PERCENTILE(spend, {q}) AS p FROM {history} GROUP BY channel_id"
    ).collect()}
    checks["spend_floor_snapshot_max_rel"] = float(
        max(SAT.check_spend_floor_snapshot(live_floor).values()))
    out["checks"] = checks

    # --- MLflow: create THE run for this Workflow run -------------------------
    mlflow, exp, active = start_or_resume(
        p.mlflow_experiment, None, run_name=f"phase5-workflow-run-{p.job_run_id}")
    run_id = active.info.run_id
    inputs, _ch = load_inputs(spark, p)
    params = {
        "total_budget": p.total_budget,
        "master_seed": p.master_seed,
        "n_paths": p.n_paths,
        "stage1_dirichlet_total": p.stage1_dirichlet_total,
        "stage2_cap": p.stage2_cap,
        "catalog": p.catalog,
        "job_run_id": p.job_run_id,
        "reference_spend": inputs.reference_spend,
        "saturate_std_cpc": inputs.saturate_std_cpc,
        "crn_stream_label": CRN_STREAM_LABEL,
        "correlation_source": "cvr",
        "var_confidence": F.VAR_CONFIDENCE,
        "spend_floor_percentile": SAT.BRONZE_SPEND_FLOOR_PERCENTILE,
        "inputs_fingerprint": inputs_fingerprint(inputs),
    }
    # theta and the spend floor PER CHANNEL, logged as the values the run
    # actually used -- read off `saturation`, not re-derived here.
    for channel, theta in zip(inputs.channels, inputs.theta):
        params[f"theta__{channel}"] = float(theta)
    for channel, floor in zip(inputs.channels, inputs.spend_floor):
        params[f"spend_floor__{channel}"] = float(floor)
    mlflow.log_params(params)
    mlflow.set_tags({"phase": "5", "pipeline": "ad_mc_poc_frontier",
                     "job_run_id": p.job_run_id})
    for key, value in checks.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(_metric_key("bronze", key), float(value))
    mlflow.log_metric("wall_clock_s__bronze_refresh", time.time() - t0)
    end_intermediate(mlflow)

    out.update({"mlflow_run_id": run_id, "mlflow_experiment_id": exp.experiment_id,
                "mlflow_experiment": p.mlflow_experiment,
                "inputs_fingerprint": params["inputs_fingerprint"],
                "channels": ch, "seconds": round(time.time() - t0, 2)})
    return out


# =============================================================================
# Task 2 -- stage1_simulate
# =============================================================================

def run_stage1_simulate(spark, p: Params, mlflow_run_id: str) -> dict:
    t0 = time.time()
    mlflow, _exp, _run = start_or_resume(p.mlflow_experiment, mlflow_run_id)
    inputs, ch = load_inputs(spark, p)
    fingerprint = inputs_fingerprint(inputs)

    cfg = stage1_config_for(p.stage1_dirichlet_total)
    candidates = F.generate_stage1(ch, p.total_budget, config=cfg, master_seed=p.master_seed)
    F.assert_candidate_set_valid(candidates, ch, p.total_budget)

    env = executor_environment(spark)
    print("executor environment:", json.dumps(env, indent=2))

    results, diag = simulate_distributed(spark, candidates, inputs, p)

    n_cand = write_parquet(candidates_to_frame(candidates, ch), p.path("stage1_candidates.parquet"))
    n_res = write_parquet(results, p.path("stage1_results.parquet"))

    families = pd.Series([c.family for c in candidates]).value_counts().to_dict()
    env_params = {}
    if env and "error" not in env[0]:
        first = env[0]
        env_params = {f"executor_{k}": str(first[k])
                      for k in ("python", "numpy", "scipy", "pandas", "wheel")}
        env_params["executor_distinct_workers"] = len({e["worker_uid"] for e in env})
    mlflow.log_params({"stage1_candidates": len(candidates),
                       "stage1_dirichlet_counts": json.dumps(list(cfg.dirichlet)),
                       **env_params})
    mlflow.log_metric("stage1_cells", diag["n_cells"])
    mlflow.log_metric("stage1_paths", diag["paths"])
    mlflow.log_metric("stage1_chunks", diag["n_chunks"])
    mlflow.log_metric("stage1_sweep_seconds", diag["seconds"])
    mlflow.log_metric("wall_clock_s__stage1_simulate", time.time() - t0)
    end_intermediate(mlflow)

    return {"task": "stage1_simulate", "n_candidates": len(candidates),
            "families": {k: int(v) for k, v in families.items()},
            "dirichlet_counts": [list(x) for x in cfg.dirichlet],
            "executor_environment": env,
            "inputs_fingerprint": fingerprint, "diag": diag,
            "bytes_written": {"candidates": n_cand, "results": n_res},
            "scratch_dir": p.scratch_dir, "seconds": round(time.time() - t0, 2)}


# =============================================================================
# Task 3 -- stage2_generate
# =============================================================================

def run_stage2_generate(spark, p: Params, mlflow_run_id: str) -> dict:
    """The adaptive edge: refine around stage 1's Pareto union.

    This is why the DAG is not one flat parallel job. It is cheap (no simulation)
    and runs on the driver, but it cannot start before stage 1 finishes and
    stage 2 cannot start before it.
    """
    t0 = time.time()
    mlflow, _exp, _run = start_or_resume(p.mlflow_experiment, mlflow_run_id)
    inputs, ch = load_inputs(spark, p)

    s1 = frame_to_candidates(read_parquet(p.path("stage1_candidates.parquet")), ch)
    df1 = read_parquet(p.path("stage1_results.parquet"))
    df1 = GA.reorder_cells(df1, s1)

    seed_ids = set(F.pareto_union_ids(df1))
    seeds = [c for c in s1 if c.allocation_id in seed_ids]

    cfg = Stage2Config(max_candidates=p.stage2_cap)
    s2 = F.generate_stage2(ch, seeds, p.total_budget, config=cfg, master_seed=p.master_seed)
    F.assert_candidate_set_valid(s2, ch, p.total_budget)

    families = pd.Series([c.family for c in s2]).value_counts().to_dict()
    # The stage-2 cap is per-family round robin (`_cap_preserving_families`). A
    # flat truncation used to keep 400 perturbations and ZERO blends, deleting
    # the only move that samples BETWEEN two frontier points. Assert the fix
    # holds rather than trusting it: with >= 2 seed points, blends must survive.
    if len(seeds) >= 2 and int(families.get("blend", 0)) == 0:
        raise RuntimeError(
            f"stage 2 produced no blends from {len(seeds)} seed points -- the "
            f"per-family cap has regressed to a flat truncation"
        )
    write_parquet(candidates_to_frame(s2, ch), p.path("stage2_candidates.parquet"))

    mlflow.log_params({"stage2_candidates": len(s2), "stage2_cap": p.stage2_cap})
    mlflow.log_metric("stage1_pareto_union", len(seeds))
    for fam, n in families.items():
        mlflow.log_metric(_metric_key("stage2_family", fam), float(n))
    mlflow.log_metric("wall_clock_s__stage2_generate", time.time() - t0)
    end_intermediate(mlflow)

    return {"task": "stage2_generate", "stage1_pareto_union": len(seeds),
            "n_candidates": len(s2), "families": {k: int(v) for k, v in families.items()},
            "inputs_fingerprint": inputs_fingerprint(inputs),
            "seconds": round(time.time() - t0, 2)}


# =============================================================================
# Task 4 -- stage2_simulate
# =============================================================================

def run_stage2_simulate(spark, p: Params, mlflow_run_id: str,
                        expect_fingerprint: str | None = None) -> dict:
    t0 = time.time()
    mlflow, _exp, _run = start_or_resume(p.mlflow_experiment, mlflow_run_id)
    inputs, ch = load_inputs(spark, p)
    fingerprint = inputs_fingerprint(inputs)
    if expect_fingerprint and fingerprint != expect_fingerprint:
        raise RuntimeError(
            f"bronze moved mid-run: stage 1 priced cells with inputs {expect_fingerprint}, "
            f"stage 2 sees {fingerprint}. The two stages would not be comparable."
        )

    s2 = frame_to_candidates(read_parquet(p.path("stage2_candidates.parquet")), ch)
    results, diag = simulate_distributed(spark, s2, inputs, p)
    write_parquet(results, p.path("stage2_results.parquet"))

    mlflow.log_metric("stage2_cells", diag["n_cells"])
    mlflow.log_metric("stage2_paths", diag["paths"])
    mlflow.log_metric("stage2_chunks", diag["n_chunks"])
    mlflow.log_metric("stage2_sweep_seconds", diag["seconds"])
    mlflow.log_metric("wall_clock_s__stage2_simulate", time.time() - t0)
    end_intermediate(mlflow)

    return {"task": "stage2_simulate", "n_candidates": len(s2), "diag": diag,
            "inputs_fingerprint": fingerprint, "seconds": round(time.time() - t0, 2)}


# =============================================================================
# Task 5 -- gold_aggregate
# =============================================================================

GOLD_SPARK_SCHEMAS = {
    "sweep": ("allocation_id STRING, scenario_id STRING, stage INT, family STRING, "
              "n_paths INT, seed STRING, total_spend DOUBLE, mean_revenue DOUBLE, "
              "se_mean_revenue DOUBLE, std_revenue DOUBLE, median_revenue DOUBLE, "
              "min_revenue DOUBLE, max_revenue DOUBLE, var_95 DOUBLE, cvar_95 DOUBLE, "
              "expected_roas DOUBLE, extrapolation_floor_applied BOOLEAN, "
              "n_channels_floored INT"),
    "frontier": ("scenario_id STRING, objective_pair STRING, allocation_id STRING, "
                 "mean_revenue DOUBLE, std_revenue DOUBLE, var_95 DOUBLE, cvar_95 DOUBLE, "
                 "expected_roas DOUBLE, rank_by_return INT, "
                 "extrapolation_floor_applied BOOLEAN"),
    "recs": ("scenario_id STRING, objective_pair STRING, recommendation STRING, "
             "allocation_id STRING, mean_revenue DOUBLE, std_revenue DOUBLE, "
             "var_95 DOUBLE, cvar_95 DOUBLE, expected_roas DOUBLE, n_efficient INT, "
             "balanced_is_degenerate BOOLEAN, nearest_neighbour_gap DOUBLE, "
             "ordering_unresolved BOOLEAN, extrapolation_floor_applied BOOLEAN"),
}

_INT_COLUMNS = {"stage", "n_paths", "n_channels_floored", "rank_by_return", "n_efficient"}
_BOOL_COLUMNS = {"extrapolation_floor_applied", "balanced_is_degenerate",
                 "ordering_unresolved"}


def _overwrite_table(spark, pdf: pd.DataFrame, table: str, columns: list[str],
                     schema: str, view: str) -> int:
    """INSERT OVERWRITE, not saveAsTable and not append.

    Same reasoning Phase 3 and Phase 4 used: these tables carry no run_id, so an
    append would stack a second sweep that any aggregate would silently average;
    and `saveAsTable` rewrites table metadata, which would discard the column
    comments `08_create_gold_frontier.py` deliberately maintains. INSERT
    OVERWRITE is one atomic Delta commit that replaces the DATA only.
    """
    if pdf.empty:
        raise ValueError(f"refusing to write zero rows to {table}")
    out = pdf[columns].copy()
    for c in columns:
        if c in _INT_COLUMNS:
            out[c] = out[c].astype("int32")
        elif c in _BOOL_COLUMNS:
            out[c] = out[c].astype(bool)
    spark.createDataFrame(out, schema=schema).createOrReplaceTempView(view)
    spark.sql(f"INSERT OVERWRITE TABLE {table} "
              f"SELECT {', '.join(columns)} FROM {view}")
    return int(spark.sql(f"SELECT COUNT(*) AS n FROM {table}").collect()[0]["n"])


def _frontier_plots(sweep: pd.DataFrame, frontier: pd.DataFrame, outdir: str) -> list[str]:
    """Scatter of every candidate with the efficient set traced on top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    pairs = [(ret, risk, f"{ret} vs {risk}") for ret, risk, _hb in F.OBJECTIVE_PAIRS]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sub = sweep[sweep["scenario_id"] == "normal"]
    for ax, (ret, risk, label) in zip(axes, pairs):
        ax.scatter(sub[risk] / 1e3, sub[ret] / 1e3, s=8, alpha=0.25,
                   color="#9aa5b1", label="candidates")
        eff = frontier[(frontier["scenario_id"] == "normal") &
                       (frontier["objective_pair"] == label)].sort_values(ret)
        ax.plot(eff[risk] / 1e3, eff[ret] / 1e3, "-o", ms=4, lw=1.2,
                color="#c0392b", label=f"efficient (n={len(eff)})")
        ax.set_xlabel(f"{risk}  ($k)")
        ax.set_ylabel(f"{ret}  ($k)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Efficient frontier, scenario = normal", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "frontier_normal.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    scenarios = sorted(sweep["scenario_id"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.6 * len(scenarios), 4.4),
                             squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        sub = sweep[sweep["scenario_id"] == scenario]
        ax.scatter(sub["var_95"] / 1e3, sub["mean_revenue"] / 1e3, s=8, alpha=0.25,
                   color="#9aa5b1")
        eff = frontier[(frontier["scenario_id"] == scenario) &
                       (frontier["objective_pair"] == "mean_revenue vs var_95")]
        eff = eff.sort_values("var_95")
        ax.plot(eff["var_95"] / 1e3, eff["mean_revenue"] / 1e3, "-o", ms=5, lw=1.2,
                color="#c0392b")
        ax.set_title(f"{scenario}  (n_efficient={len(eff)})", fontsize=10)
        ax.set_xlabel("VaR-95  ($k)")
        ax.set_ylabel("mean revenue  ($k)")
        ax.grid(alpha=0.2)
    fig.suptitle("mean_revenue vs VaR-95 by scenario -- a short arc, not a rich curve",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "frontier_mean_var95_by_scenario.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def run_gold_aggregate(spark, p: Params, mlflow_run_id: str,
                       expect_fingerprint: str | None = None) -> dict:
    t0 = time.time()
    mlflow, _exp, _run = start_or_resume(p.mlflow_experiment, mlflow_run_id)
    inputs, ch = load_inputs(spark, p)
    fingerprint = inputs_fingerprint(inputs)
    if expect_fingerprint and fingerprint != expect_fingerprint:
        raise RuntimeError(
            f"bronze moved mid-run: the sweep priced cells with {expect_fingerprint}, "
            f"gold_aggregate sees {fingerprint}"
        )

    s1 = frame_to_candidates(read_parquet(p.path("stage1_candidates.parquet")), ch)
    s2 = frame_to_candidates(read_parquet(p.path("stage2_candidates.parquet")), ch)
    df1 = read_parquet(p.path("stage1_results.parquet"))
    df2 = read_parquet(p.path("stage2_results.parquet"))

    candidates = list(s1) + list(s2)
    df = pd.concat([df1, df2], ignore_index=True)

    # reorder=True: Spark returns cells in whatever order its tasks finished in,
    # and the frontier helpers break exact ties by row position. Putting the rows
    # back into `run_sweep` order removes scheduling from the answer.
    sweep, frontier_df, recs = GA.build_gold_frames(
        df, candidates, inputs.spend_floor, reorder=True)

    gold = f"{p.catalog}.gold"
    tables = {"sweep": f"{gold}.allocation_sweep_results",
              "frontier": f"{gold}.efficient_frontier",
              "recs": f"{gold}.frontier_recommendations"}
    for table in tables.values():
        try:
            spark.sql(f"DESCRIBE TABLE {table}")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"{table} is not readable ({exc}). Gold DDL is owned by "
                f"databricks/08_create_gold_frontier.py; this task writes DATA only, "
                f"so the column comments that script maintains are never rewritten "
                f"by a saveAsTable here."
            ) from exc

    written = {
        "sweep": _overwrite_table(spark, sweep, tables["sweep"], GA.SWEEP_TABLE_COLUMNS,
                                  GOLD_SPARK_SCHEMAS["sweep"], "_p5_sweep"),
        "frontier": _overwrite_table(spark, frontier_df, tables["frontier"],
                                     GA.FRONTIER_TABLE_COLUMNS,
                                     GOLD_SPARK_SCHEMAS["frontier"], "_p5_frontier"),
        "recs": _overwrite_table(spark, recs, tables["recs"], GA.REC_TABLE_COLUMNS,
                                 GOLD_SPARK_SCHEMAS["recs"], "_p5_recs"),
    }

    # --- data quality, against the LIVE tables -------------------------------
    failures = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(f"{label}{(' -- ' + detail) if detail else ''}")

    def scalar(query: str):
        return spark.sql(query).collect()[0][0]

    n_expected = len(candidates) * len(SCENARIOS)
    check("sweep rows == candidates x scenarios", written["sweep"] == n_expected,
          f"{written['sweep']} vs {n_expected}")
    check("no null metrics", scalar(
        f"SELECT COUNT(*) FROM {tables['sweep']} WHERE mean_revenue IS NULL "
        f"OR var_95 IS NULL OR cvar_95 IS NULL") == 0)
    check("CVaR-95 <= VaR-95 everywhere", scalar(
        f"SELECT COUNT(*) FROM {tables['sweep']} WHERE cvar_95 > var_95") == 0)
    check("every candidate spends the full budget", scalar(
        f"SELECT COUNT(*) FROM {tables['sweep']} "
        f"WHERE ABS(total_spend - {p.total_budget}) > 1e-6") == 0)
    check("every frontier row references a real sweep row", scalar(
        f"SELECT COUNT(*) FROM {tables['frontier']} f LEFT JOIN {tables['sweep']} s "
        f"ON f.allocation_id = s.allocation_id AND f.scenario_id = s.scenario_id "
        f"WHERE s.allocation_id IS NULL") == 0)
    check("the extrapolation flag agrees with the channel count it summarises", scalar(
        f"SELECT COUNT(*) FROM {tables['sweep']} "
        f"WHERE (n_channels_floored > 0) <> extrapolation_floor_applied") == 0)
    check("3 recommendations x 4 scenarios for each of 3 objective pairs",
          written["recs"] == 3 * 4 * len(F.OBJECTIVE_PAIRS),
          f"{written['recs']} rows")
    check("no duplicate (allocation, scenario) keys in the sweep", scalar(
        f"SELECT COUNT(*) FROM (SELECT allocation_id, scenario_id FROM {tables['sweep']} "
        f"GROUP BY allocation_id, scenario_id HAVING COUNT(*) > 1)") == 0)
    if failures:
        raise RuntimeError("gold data-quality checks failed: " + "; ".join(failures))

    # --- metrics --------------------------------------------------------------
    sizes = (frontier_df.groupby(["scenario_id", "objective_pair"]).size()
             .reset_index(name="n"))
    for r in sizes.itertuples(index=False):
        mlflow.log_metric(
            _metric_key("frontier_size", r.scenario_id,
                        r.objective_pair.replace("mean_revenue vs ", "mean_vs_")),
            float(r.n))

    best = sweep.loc[sweep["mean_revenue"].astype(float).idxmax()]
    best_normal = (sweep[sweep["scenario_id"] == "normal"]
                   .sort_values("mean_revenue", ascending=False).iloc[0])
    n_sweep, n_front, n_recs = len(sweep), len(frontier_df), len(recs)
    flags = {
        "floor_flagged_sweep": int(sweep["extrapolation_floor_applied"].sum()),
        "floor_flagged_frontier": int(frontier_df["extrapolation_floor_applied"].sum()),
        "floor_flagged_recs": int(recs["extrapolation_floor_applied"].sum()),
        "ordering_unresolved_recs": int(recs["ordering_unresolved"].sum()),
    }
    metrics = {
        "best_expected_revenue": float(best["mean_revenue"]),
        "best_expected_revenue_normal": float(best_normal["mean_revenue"]),
        "total_paths_simulated": float(n_sweep * p.n_paths),
        "n_candidates_total": float(len(candidates)),
        "gold_sweep_rows": float(n_sweep),
        "gold_frontier_rows": float(n_front),
        "gold_recommendation_rows": float(n_recs),
        "floor_flagged_sweep_count": float(flags["floor_flagged_sweep"]),
        "floor_flagged_sweep_pct": 100.0 * flags["floor_flagged_sweep"] / n_sweep,
        "floor_flagged_frontier_count": float(flags["floor_flagged_frontier"]),
        "floor_flagged_frontier_pct": 100.0 * flags["floor_flagged_frontier"] / n_front,
        "floor_flagged_recs_count": float(flags["floor_flagged_recs"]),
        "floor_flagged_recs_pct": 100.0 * flags["floor_flagged_recs"] / n_recs,
        "ordering_unresolved_count": float(flags["ordering_unresolved_recs"]),
        "ordering_unresolved_pct": 100.0 * flags["ordering_unresolved_recs"] / n_recs,
    }
    mlflow.log_metrics(metrics)
    mlflow.log_params({"best_allocation_overall": str(best["allocation_id"]),
                       "best_allocation_normal": str(best_normal["allocation_id"]),
                       "total_candidates": len(candidates)})

    # --- artefacts ------------------------------------------------------------
    outdir = "/tmp/ad_mc_phase5_artifacts"
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "frontier_recommendations.csv")
    recs.to_csv(csv_path, index=False)
    artifacts = [csv_path]
    try:
        artifacts += _frontier_plots(sweep.astype({"mean_revenue": float, "var_95": float,
                                                   "cvar_95": float, "std_revenue": float}),
                                     frontier_df, outdir)
        plot_error = None
    except Exception as exc:  # noqa: BLE001 -- reported, never silently skipped
        plot_error = repr(exc)
    for path in artifacts:
        mlflow.log_artifact(path)

    mlflow.log_metric("wall_clock_s__gold_aggregate", time.time() - t0)
    mlflow.end_run()  # the pipeline's run finishes here

    return {"task": "gold_aggregate", "written": written,
            "frontier_sizes": sizes.to_dict("records"),
            "metrics": metrics, "flags": flags,
            "best_allocation_overall": str(best["allocation_id"]),
            "best_scenario_overall": str(best["scenario_id"]),
            "best_allocation_normal": str(best_normal["allocation_id"]),
            "best_mean_revenue_normal": float(best_normal["mean_revenue"]),
            "artifacts": [os.path.basename(a) for a in artifacts],
            "plot_error": plot_error,
            "inputs_fingerprint": fingerprint,
            "seconds": round(time.time() - t0, 2)}
