# Databricks notebook source
# =============================================================================
# PHASE 5 TASK 1 of 5 -- bronze_refresh
#
# Rebuilds the derived bronze tables (channel_assumptions, both correlation
# matrices) and MERGEs scenario_definitions, then opens the ONE MLflow run this
# Workflow run reports into and hands its run_id forward as a task value.
#
# This notebook is deliberately thin. Everything it does lives in
# `ad_mc_sim.phase5_tasks`, which ships inside the same wheel the Spark
# executors install -- so the driver and the executors run code from one
# distribution mechanism, not two. Read simulation/phase5_tasks.py for the
# reasoning; read simulation/run_phase5_workflow.py for how this is deployed.
# =============================================================================

import json

import ad_mc_sim.phase5_tasks as T

# Widget defaults are EMPTY on purpose: every value must arrive from a job
# parameter. An empty widget makes `resolve_params` raise instead of silently
# running on a default nobody set.
for _name, _desc in T.WIDGETS:
    dbutils.widgets.text(_name, "", _desc)

PARAMS = T.resolve_params({name: dbutils.widgets.get(name) for name, _ in T.WIDGETS})
print("resolved job parameters:")
print(json.dumps(PARAMS.__dict__, indent=2, default=str))
print(f"scratch dir: {PARAMS.scratch_dir}")

# The MLflow run spans all five tasks, so each task closes its handle with
# status RUNNING for the next one to resume. An exception skips that, which
# is how job run 529952200505846 left a run RUNNING forever -- carrying
# metric keys that overlap the successful runs', so an unfiltered aggregate
# over the experiment silently mixed a partial run in. Mark it FAILED and
# re-raise: the task must still fail, the run must not stay open.
try:
    RESULT = T.run_bronze_refresh(spark, PARAMS)
except BaseException:
    import mlflow as _mlflow
    _active = _mlflow.active_run()
    T.mark_run_failed(PARAMS.mlflow_experiment,
                      _active.info.run_id if _active else None)
    raise

# The pipeline's MLflow run id crosses the task boundary here. Later tasks resume
# this run rather than starting their own, so one Workflow run == one MLflow run.
dbutils.jobs.taskValues.set(key="mlflow_run_id", value=RESULT["mlflow_run_id"])
dbutils.jobs.taskValues.set(key="mlflow_experiment_id", value=RESULT["mlflow_experiment_id"])
dbutils.jobs.taskValues.set(key="inputs_fingerprint", value=RESULT["inputs_fingerprint"])

print(json.dumps(RESULT, indent=2, default=str))
dbutils.notebook.exit(json.dumps(RESULT, default=str))
