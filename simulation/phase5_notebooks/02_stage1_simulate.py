# Databricks notebook source
# =============================================================================
# PHASE 5 TASK 2 of 5 -- stage1_simulate
#
# Generates the broad stage-1 candidate set and simulates every
# (candidate, scenario) cell with `applyInPandas`, saturation and the spend floor
# exactly as Phase 4 built them. Writes the candidates and the per-cell
# aggregates to the run's scratch directory on a UC volume.
#
# This is the task where Spark earns its keep: 571 x 4 x 10,000 = 22.8 million
# paths. Phase 3's 84-cell grid ran in 7.4s single threaded and only DEMONSTRATED
# distribution; this does not.
# =============================================================================

import json

import ad_mc_sim.phase5_tasks as T

for _name, _desc in T.WIDGETS:
    dbutils.widgets.text(_name, "", _desc)

PARAMS = T.resolve_params({name: dbutils.widgets.get(name) for name, _ in T.WIDGETS})
MLFLOW_RUN_ID = dbutils.jobs.taskValues.get(
    taskKey="bronze_refresh", key="mlflow_run_id", debugValue="")
if not MLFLOW_RUN_ID:
    raise RuntimeError("no mlflow_run_id task value from bronze_refresh")

print(f"resuming MLflow run {MLFLOW_RUN_ID} in {PARAMS.mlflow_experiment}")
# The MLflow run spans all five tasks, so each task closes its handle with
# status RUNNING for the next one to resume. An exception skips that, which
# is how job run 529952200505846 left a run RUNNING forever -- carrying
# metric keys that overlap the successful runs', so an unfiltered aggregate
# over the experiment silently mixed a partial run in. Mark it FAILED and
# re-raise: the task must still fail, the run must not stay open.
try:
    RESULT = T.run_stage1_simulate(spark, PARAMS, MLFLOW_RUN_ID)
except BaseException:
    T.mark_run_failed(PARAMS.mlflow_experiment, MLFLOW_RUN_ID)
    raise

dbutils.jobs.taskValues.set(key="stage1_candidates", value=int(RESULT["n_candidates"]))
dbutils.jobs.taskValues.set(key="inputs_fingerprint", value=RESULT["inputs_fingerprint"])

print(json.dumps(RESULT, indent=2, default=str))
dbutils.notebook.exit(json.dumps(RESULT, default=str))
