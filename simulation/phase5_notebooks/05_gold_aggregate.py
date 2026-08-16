# Databricks notebook source
# =============================================================================
# PHASE 5 TASK 5 of 5 -- gold_aggregate
#
# Merges both stages, builds the three gold frames through `gold_assembly` (the
# same code the local Phase 4 loader calls), INSERT OVERWRITEs the three gold
# tables, runs the data-quality checks against the live tables, logs the
# headline metrics and the artefacts, and closes the MLflow run.
#
# BOTH caveat flags are populated here: `ordering_unresolved` (the pick is not
# distinguishable from its neighbour at this sweep's CRN noise) and
# `extrapolation_floor_applied` (at least one funded channel was priced AT its
# evidence floor). Phase 6 must respect both.
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
EXPECT_FINGERPRINT = dbutils.jobs.taskValues.get(
    taskKey="stage1_simulate", key="inputs_fingerprint", debugValue="")

# The MLflow run spans all five tasks, so each task closes its handle with
# status RUNNING for the next one to resume. An exception skips that, which
# is how job run 529952200505846 left a run RUNNING forever -- carrying
# metric keys that overlap the successful runs', so an unfiltered aggregate
# over the experiment silently mixed a partial run in. Mark it FAILED and
# re-raise: the task must still fail, the run must not stay open.
try:
    RESULT = T.run_gold_aggregate(spark, PARAMS, MLFLOW_RUN_ID, EXPECT_FINGERPRINT or None)
except BaseException:
    T.mark_run_failed(PARAMS.mlflow_experiment, MLFLOW_RUN_ID)
    raise

print(json.dumps(RESULT, indent=2, default=str))
dbutils.notebook.exit(json.dumps(RESULT, default=str))
