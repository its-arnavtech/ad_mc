"""Phase 5 -- deploy the persisted Databricks Workflow, run it, and check it.

    python simulation/run_phase5_workflow.py                 # deploy + run + report
    python simulation/run_phase5_workflow.py --deploy-only   # create/update, do not run
    python simulation/run_phase5_workflow.py --run-only      # run the existing job
    python simulation/run_phase5_workflow.py --compare-only --reference-dir DIR

THE FIRST PERSISTED JOB IN THIS PROJECT
---------------------------------------
Phases 3 and 4 deliberately used transient `jobs.submit` runs -- a one-off
execution, not a job definition -- to stay inside the "no Workflows before
Phase 5" rule. This creates a real, named, re-runnable job with a DAG, job-level
parameters and five tasks.

WHAT IT RUNS ON, AND WHY THERE IS NO CHOICE
-------------------------------------------
Serverless job compute with the simulation modules installed from a wheel named
in the environment spec. This workspace has NO classic compute plane (cluster
creation times out; `clusters.list_zones()` fails) and Databricks Connect cannot
run Python UDFs here at all (`ISOLATION_STARTUP_FAILURE.SANDBOX_STARTUP`). Phase
3 established the one path that works and built the machinery; this module
reuses `build_wheel` / `upload_wheel` / `upload_notebook` from
`run_phase3_distributed.py` rather than owning a second copy.

The Phase 5 wheel carries TWO MODULES PHASE 3'S DID NOT -- `frontier.py` and
`sweep_seeding.py`, because the sweep's UDF calls `summarize_cell_row` and the
CRN seeds come from `crn_seed` -- plus `gold_assembly.py`, `bronze_sql.py` and
`phase5_tasks.py`. More modules means a different content hash and therefore a
different wheel version, which is the provenance mechanism working.

PARAMETERS
----------
Five job parameters, defaulting to EXACTLY the values behind the verified
`a289016` run, plus `catalog` and `mlflow_experiment` so nothing is hardcoded in
five notebooks. Each task also references them explicitly through
`{{job.parameters.<name>}}` in its `base_parameters`, so the value is delivered
whether or not job-parameter push-down is doing it, and the notebook widgets
default to EMPTY so a parameter that failed to resolve fails the task instead of
silently reverting to a local default.

`stage1_dirichlet_total` is named for what it actually controls. The stage-1 set
is 21 Phase 3 candidates + a centroid + a {5,3} and a {5,4} simplex lattice (35
and 70 points, a fixed enumeration) + three Dirichlet blocks. Only the last is a
real knob, so a parameter called "stage1_candidates" would be describing
something it cannot set. At the default 450 the three blocks are (225, 125, 100)
exactly and the deduped total is 571 -- the same 571 as Phase 4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "databricks"))
sys.path.insert(0, str(REPO_ROOT / "simulation"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from databricks.sdk.service import compute, jobs  # noqa: E402

from _common import get_client, resolve_warehouse_id, sql  # noqa: E402
from run_phase3_distributed import build_wheel, upload_notebook, upload_wheel  # noqa: E402

import gold_assembly as GA  # noqa: E402
from config import N_PATHS, RANDOM_SEED, TOTAL_BUDGET  # noqa: E402
from frontier import Stage1Config, Stage2Config  # noqa: E402

JOB_NAME = "ad_mc_poc -- frontier pipeline (Phase 5)"
ENVIRONMENT_KEY = "ad_mc_sim_env"
NOTEBOOK_DIR = REPO_ROOT / "simulation" / "phase5_notebooks"
CATALOG = "ad_mc_poc"

# Everything the executors import. The Phase 3 list plus what the sweep needs:
# frontier (the UDF body and the Pareto/recommendation rules), sweep_seeding
# (the CRN seeds), gold_assembly (shared with the local Phase 4 loader),
# bronze_sql (shared with databricks/03_build_assumptions.py) and phase5_tasks.
PACKAGE_MODULES = ["config.py", "engine.py", "cell.py", "scenarios.py",
                   "allocations.py", "seeding.py", "saturation.py",
                   "sweep_seeding.py", "frontier.py", "gold_assembly.py",
                   "bronze_sql.py", "phase5_tasks.py"]
REQUIRED_IN_WHEEL = ("engine.py", "cell.py", "saturation.py", "seeding.py",
                     "frontier.py", "sweep_seeding.py", "gold_assembly.py",
                     "phase5_tasks.py")

# task_key -> (notebook file, [upstream task_keys])
TASKS: list[tuple[str, str, list[str], str]] = [
    ("bronze_refresh", "01_bronze_refresh.py", [],
     "Rebuild channel_assumptions and both correlation matrices from history; "
     "MERGE scenario_definitions; open the MLflow run."),
    ("stage1_simulate", "02_stage1_simulate.py", ["bronze_refresh"],
     "Generate the broad stage-1 candidates and simulate every "
     "(candidate, scenario) cell with applyInPandas."),
    ("stage2_generate", "03_stage2_generate.py", ["stage1_simulate"],
     "The adaptive edge: Pareto union of stage 1, then refinement and blend "
     "candidates under the per-family cap."),
    ("stage2_simulate", "04_stage2_simulate.py", ["stage2_generate"],
     "Simulate the refinement candidates on the same scenario seeds."),
    ("gold_aggregate", "05_gold_aggregate.py", ["stage1_simulate", "stage2_simulate"],
     "Merge both stages, compute the frontier and the recommendations, write "
     "the three gold tables, log metrics and artefacts."),
]

# Defaults reproduce the verified a289016 run exactly.
JOB_PARAMETER_DEFAULTS: dict[str, str] = {
    "total_budget": repr(float(TOTAL_BUDGET)),
    "master_seed": str(int(RANDOM_SEED)),
    "n_paths": str(int(N_PATHS)),
    "stage1_dirichlet_total": str(sum(n for _a, n in Stage1Config().dirichlet)),
    "stage2_cap": str(int(Stage2Config().max_candidates)),
    "catalog": CATALOG,
    # filled in at deploy time with the workspace user
    "mlflow_experiment": "",
}

TBL_SWEEP = f"{CATALOG}.gold.allocation_sweep_results"
TBL_FRONTIER = f"{CATALOG}.gold.efficient_frontier"
TBL_RECS = f"{CATALOG}.gold.frontier_recommendations"

REFERENCE_TABLES = {
    "allocation_sweep_results": (TBL_SWEEP, ["allocation_id", "scenario_id"],
                                 {"extrapolation_floor_applied"}),
    "efficient_frontier": (TBL_FRONTIER,
                           ["scenario_id", "objective_pair", "allocation_id"],
                           {"extrapolation_floor_applied"}),
    "frontier_recommendations": (TBL_RECS,
                                 ["scenario_id", "objective_pair", "recommendation"],
                                 {"balanced_is_degenerate", "ordering_unresolved",
                                  "extrapolation_floor_applied"}),
}


# =============================================================================
# Deploy
# =============================================================================

def build_task_list(notebook_paths: dict[str, str]) -> list[jobs.Task]:
    """One notebook task per DAG node, with the dependency edges."""
    param_refs = {name: "{{job.parameters." + name + "}}"
                  for name in JOB_PARAMETER_DEFAULTS}
    out = []
    for task_key, filename, upstream, description in TASKS:
        base = dict(param_refs)
        # Not a job parameter: it is per-RUN, and it scopes the scratch directory
        # so two runs never read each other's intermediate state.
        base["job_run_id"] = "{{job.run_id}}"
        out.append(jobs.Task(
            task_key=task_key,
            description=description,
            depends_on=[jobs.TaskDependency(task_key=u) for u in upstream] or None,
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_paths[filename], base_parameters=base),
            environment_key=ENVIRONMENT_KEY,
            timeout_seconds=3600,
            max_retries=0,
        ))
    return out


def deploy(w, wheel_path: str, user: str) -> int:
    """Create the job, or update it in place if it already exists. Returns job_id."""
    # `workspace.upload` will not create intermediate folders.
    w.workspace.mkdirs(f"/Users/{user}/_ad_mc_phase5")
    notebook_paths = {}
    for _key, filename, _up, _desc in TASKS:
        stem = filename[:-3]
        path = upload_notebook(w, user, source=NOTEBOOK_DIR / filename,
                               base_name=f"_ad_mc_phase5/{stem}")
        notebook_paths[filename] = path

    defaults = dict(JOB_PARAMETER_DEFAULTS)
    defaults["mlflow_experiment"] = f"/Users/{user}/ad_mc_poc"

    # Typed kwargs, shared by create() and reset() so the two paths cannot define
    # different jobs. `JobSettings(**kwargs)` keeps the SDK objects intact;
    # `settings.as_dict()` would flatten them and jobs.create() wants the objects.
    kwargs = dict(
        name=JOB_NAME,
        description=(
            "Two-stage budget-allocation optimization sweep: bronze refresh -> "
            "stage-1 broad simplex coverage -> Pareto union -> stage-2 refinement "
            "-> gold frontier and recommendations. Serverless job compute; the "
            "simulation modules are installed from a content-hashed wheel."),
        tasks=build_task_list(notebook_paths),
        parameters=[jobs.JobParameterDefinition(name=k, default=v)
                    for k, v in defaults.items()],
        environments=[jobs.JobEnvironment(
            environment_key=ENVIRONMENT_KEY,
            spec=compute.Environment(client="4", dependencies=[wheel_path]))],
        max_concurrent_runs=1,
        queue=jobs.QueueSettings(enabled=True),
        tags={"project": "ad_mc_poc", "phase": "5"},
        timeout_seconds=7200,
    )

    existing = [j for j in w.jobs.list(name=JOB_NAME)]
    if existing:
        job_id = existing[0].job_id
        w.jobs.reset(job_id=job_id, new_settings=jobs.JobSettings(**kwargs))
        print(f"  updated existing job {job_id}")
    else:
        created = w.jobs.create(**kwargs)
        job_id = created.job_id
        print(f"  created job {job_id}")
    print(f"  {w.config.host}/jobs/{job_id}")
    print(f"  MLflow experiment: {defaults['mlflow_experiment']}")
    for k, v in defaults.items():
        print(f"    param {k:<24} = {v}")
    return job_id


# =============================================================================
# Run
# =============================================================================

def _ts(ms) -> str:
    if not ms:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")


def trigger_and_wait(w, job_id: int, timeout_min: int = 90) -> dict:
    waiter = w.jobs.run_now(job_id=job_id)
    run_id = waiter.run_id
    print(f"\n  triggered run {run_id}")
    print(f"  {w.config.host}/jobs/{job_id}/runs/{run_id}")

    t0 = time.time()
    seen: dict[str, str] = {}
    while True:
        run = w.jobs.get_run(run_id)
        for task in (run.tasks or []):
            state = str(getattr(task.state.life_cycle_state, "value",
                                task.state.life_cycle_state)) if task.state else "?"
            result = str(getattr(task.state.result_state, "value",
                                 task.state.result_state)) if task.state else ""
            key = f"{state}/{result}"
            if seen.get(task.task_key) != key:
                print(f"    [{time.time() - t0:6.1f}s] {task.task_key:<18} {state}"
                      + (f" ({result})" if result and result != "None" else ""))
                seen[task.task_key] = key
        life = str(getattr(run.state.life_cycle_state, "value",
                           run.state.life_cycle_state))
        if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        if time.time() - t0 > timeout_min * 60:
            raise SystemExit(f"ERROR: run {run_id} did not finish in {timeout_min} min")
        time.sleep(10)

    run = w.jobs.get_run(run_id)
    result_state = str(getattr(run.state.result_state, "value", run.state.result_state))
    print(f"\n  run {run_id} finished: {result_state} "
          f"in {time.time() - t0:.1f}s wall clock")

    print(f"\n  {'task':<18} {'state':<10} {'start':<9} {'end':<9} "
          f"{'queue':>7} {'setup':>7} {'exec':>8} {'clean':>7}")
    print("  " + "-" * 82)
    tasks_out = []
    for task in sorted(run.tasks or [], key=lambda t: t.start_time or 0):
        res = str(getattr(task.state.result_state, "value",
                          task.state.result_state)) if task.state else "?"
        row = {
            "task_key": task.task_key, "run_id": task.run_id, "result_state": res,
            "start_time": task.start_time, "end_time": task.end_time,
            "queue_duration_s": (task.queue_duration or 0) / 1000,
            "setup_duration_s": (task.setup_duration or 0) / 1000,
            "execution_duration_s": (task.execution_duration or 0) / 1000,
            "cleanup_duration_s": (task.cleanup_duration or 0) / 1000,
        }
        row["total_s"] = ((task.end_time or 0) - (task.start_time or 0)) / 1000
        tasks_out.append(row)
        print(f"  {task.task_key:<18} {res:<10} {_ts(task.start_time):<9} "
              f"{_ts(task.end_time):<9} {row['queue_duration_s']:>7.1f} "
              f"{row['setup_duration_s']:>7.1f} {row['execution_duration_s']:>8.1f} "
              f"{row['cleanup_duration_s']:>7.1f}")

    outputs = {}
    for task in (run.tasks or []):
        try:
            out = w.jobs.get_run_output(task.run_id)
        except Exception as exc:  # noqa: BLE001
            outputs[task.task_key] = {"error": repr(exc)}
            continue
        payload = {}
        if out.error:
            payload["error"] = out.error
            payload["error_trace"] = (out.error_trace or "")[:6000]
        raw = out.notebook_output.result if out.notebook_output else None
        if raw:
            try:
                payload["result"] = json.loads(raw)
            except json.JSONDecodeError:
                payload["result_raw"] = raw[:4000]
        outputs[task.task_key] = payload

    return {"run_id": run_id, "job_id": job_id, "result_state": result_state,
            "wall_clock_s": round(time.time() - t0, 1), "tasks": tasks_out,
            "outputs": outputs,
            "url": f"{w.config.host}/jobs/{job_id}/runs/{run_id}"}


# =============================================================================
# Reproduction check against a captured reference
# =============================================================================

def _is_texty(s: pd.Series) -> bool:
    # pandas 3.0 reports the new string dtype as "str", 2.x as "string"/object.
    return s.dtype == object or str(s.dtype) in ("str", "string") or \
        str(s.dtype).startswith("string")


def _normalize(df: pd.DataFrame, bool_cols: set[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=range(len(df)))
    for c in df.columns:
        s = df[c].reset_index(drop=True)
        if c in bool_cols:
            if _is_texty(s):
                parsed = s.astype(str).str.lower().map({"true": True, "false": False})
            else:
                parsed = s.astype(bool)
            if parsed.isna().any():
                raise ValueError(f"{c}: unparsable boolean values")
            out[c] = parsed.astype(bool)
        elif _is_texty(s):
            try:
                out[c] = s.astype(float)
            except (TypeError, ValueError):
                out[c] = s.astype(str)
        elif s.dtype.kind in "fiub":
            out[c] = s.astype(float)
        else:
            out[c] = s.astype(str)
    return out


def fetch_table(w, wid, table: str, order_by: list[str]) -> pd.DataFrame:
    cols, _ = sql(w, wid, f"SELECT * FROM {table} LIMIT 0")
    frames, offset, page = [], 0, 1000
    while True:
        c, rows = sql(w, wid, f"SELECT * FROM {table} ORDER BY {', '.join(order_by)} "
                              f"LIMIT {page} OFFSET {offset}")
        if not rows:
            break
        frames.append(pd.DataFrame(rows, columns=c))
        offset += page
        if len(rows) < page:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def compare_to_reference(w, wid, reference_dir: Path) -> dict:
    """Diff the live gold tables against a captured reference, column by column."""
    print("\n" + "=" * 78)
    print(f"REPRODUCTION CHECK vs {reference_dir}")
    print("=" * 78)
    report = {}
    for name, (table, keys, bool_cols) in REFERENCE_TABLES.items():
        ref_path = reference_dir / f"{name}.parquet"
        if not ref_path.exists():
            print(f"\n{name}: NO REFERENCE at {ref_path}")
            continue
        ref = pd.read_parquet(ref_path)
        live = fetch_table(w, wid, table, keys)
        print(f"\n{name}: live {len(live)} rows, reference {len(ref)} rows")
        entry = {"rows_live": int(len(live)), "rows_reference": int(len(ref)),
                 "columns": {}, "identical": len(live) == len(ref)}
        if len(live) != len(ref):
            print("  ROW COUNT DIFFERS")
            report[name] = entry
            continue
        cols = list(ref.columns)
        a = _normalize(live[cols], bool_cols).sort_values(keys, kind="stable")
        b = _normalize(ref[cols], bool_cols).sort_values(keys, kind="stable")
        a, b = a.reset_index(drop=True), b.reset_index(drop=True)
        for c in cols:
            x, y = a[c], b[c]
            if x.dtype == bool:
                n = int((x.to_numpy() != y.to_numpy()).sum())
                entry["columns"][c] = {"kind": "bool", "mismatches": n}
                entry["identical"] &= n == 0
                print(f"  {c:<30} bool mismatches {n}")
            elif x.dtype.kind == "f":
                xa, ya = x.to_numpy(float), y.to_numpy(float)
                exact = int((xa == ya).sum())
                d = np.abs(xa - ya)
                rel = d / np.where(np.abs(ya) > 0, np.abs(ya), 1.0)
                entry["columns"][c] = {"kind": "float", "exact": exact, "n": int(len(xa)),
                                       "max_abs": float(d.max()),
                                       "max_rel": float(rel.max())}
                entry["identical"] &= exact == len(xa)
                print(f"  {c:<30} exact {exact}/{len(xa)}  max|d| {d.max():.3e}  "
                      f"max rel {rel.max():.3e}")
            else:
                n = int((x.astype(str).to_numpy() != y.astype(str).to_numpy()).sum())
                entry["columns"][c] = {"kind": "string", "mismatches": n}
                entry["identical"] &= n == 0
                print(f"  {c:<30} string mismatches {n}")
        # Membership, which is what a frontier actually asserts.
        if name != "allocation_sweep_results":
            live_keys = set(map(tuple, a[keys].astype(str).to_numpy().tolist()))
            ref_keys = set(map(tuple, b[keys].astype(str).to_numpy().tolist()))
            entry["key_set_identical"] = live_keys == ref_keys
            entry["only_live"] = sorted(map(list, live_keys - ref_keys))[:10]
            entry["only_reference"] = sorted(map(list, ref_keys - live_keys))[:10]
            print(f"  key set identical: {entry['key_set_identical']}")
        entry["identical"] = bool(entry["identical"])
        print(f"  -> {'IDENTICAL' if entry['identical'] else 'DIFFERS'}")
        report[name] = entry
    return report


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-only", action="store_true")
    ap.add_argument("--run-only", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--reference-dir", default=None,
                    help="directory holding <table>.parquet reference snapshots")
    ap.add_argument("--out", default=None, help="write the run report as JSON here")
    ap.add_argument("--timeout-min", type=int, default=90)
    args = ap.parse_args()

    print("=" * 78)
    print("PHASE 5 -- PERSISTED WORKFLOW + MLFLOW")
    print("=" * 78)
    w = get_client()
    user = w.current_user.me().user_name
    print(f"  workspace: {w.config.host}")
    print(f"  user: {user}")

    report: dict = {}

    if args.compare_only:
        wid = resolve_warehouse_id(w)
        report["comparison"] = compare_to_reference(w, wid, Path(args.reference_dir))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        return

    job_id = None
    if not args.run_only:
        print("\n-- wheel --")
        filename, data = build_wheel(PACKAGE_MODULES, REQUIRED_IN_WHEEL)
        wheel_path = upload_wheel(w, filename, data)
        print("\n-- job --")
        job_id = deploy(w, wheel_path, user)
        report["wheel"] = {"filename": filename, "path": wheel_path, "bytes": len(data)}
    else:
        existing = [j for j in w.jobs.list(name=JOB_NAME)]
        if not existing:
            raise SystemExit(f"ERROR: no job named {JOB_NAME!r}; deploy it first")
        job_id = existing[0].job_id
        print(f"  using existing job {job_id}")
    report["job_id"] = job_id

    if args.deploy_only:
        print("\n--deploy-only: not triggering a run")
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        return

    print("\n-- run --")
    run = trigger_and_wait(w, job_id, timeout_min=args.timeout_min)
    report["run"] = run

    if args.reference_dir:
        wid = resolve_warehouse_id(w)
        report["comparison"] = compare_to_reference(w, wid, Path(args.reference_dir))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.out}")

    if run["result_state"] != "SUCCESS":
        raise SystemExit(f"run finished {run['result_state']}")


if __name__ == "__main__":
    main()
