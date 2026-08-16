# Databricks notebook source
# =============================================================================
# PHASE 5 TASK 4 of 5 -- stage2_simulate
#
# Simulates the refinement candidates, same engine, same scenario seeds. The
# seeds are a function of the SCENARIO only, so stage 2's cells share stage 1's
# random stream -- common random numbers hold ACROSS the two stages as well as
# within them, which is what makes a stage-2 blend comparable to the stage-1
# points it was blended from.
#
# The inputs fingerprint from stage 1 is re-checked here: if bronze moved between
# the two stages the halves would have been priced in different worlds, and this
# stops that being averaged into one frontier.
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
    RESULT = T.run_stage2_simulate(spark, PARAMS, MLFLOW_RUN_ID, EXPECT_FINGERPRINT or None)
except BaseException:
    T.mark_run_failed(PARAMS.mlflow_experiment, MLFLOW_RUN_ID)
    raise

print(json.dumps(RESULT, indent=2, default=str))
dbutils.notebook.exit(json.dumps(RESULT, default=str))
