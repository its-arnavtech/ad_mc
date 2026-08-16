# Databricks notebook source
# =============================================================================
# PHASE 5 TASK 3 of 5 -- stage2_generate
#
# THE ADAPTIVE EDGE. Reads stage 1's results, computes the Pareto union across
# every (scenario, objective pair), and generates the refinement + blend
# candidates around it under the CORRECTED per-family round-robin cap.
#
# No simulation happens here, so it is cheap -- but it cannot start before
# stage 1 finishes and stage 2 cannot start before it. That is the dependency
# that stops this pipeline being one flat parallel job.
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

# The MLflow run spans all five tasks, so each task closes its handle with
# status RUNNING for the next one to resume. An exception skips that, which
# is how job run 529952200505846 left a run RUNNING forever -- carrying
# metric keys that overlap the successful runs', so an unfiltered aggregate
# over the experiment silently mixed a partial run in. Mark it FAILED and
# re-raise: the task must still fail, the run must not stay open.
try:
    RESULT = T.run_stage2_generate(spark, PARAMS, MLFLOW_RUN_ID)
except BaseException:
    T.mark_run_failed(PARAMS.mlflow_experiment, MLFLOW_RUN_ID)
    raise

dbutils.jobs.taskValues.set(key="stage1_pareto_union", value=int(RESULT["stage1_pareto_union"]))
dbutils.jobs.taskValues.set(key="stage2_candidates", value=int(RESULT["n_candidates"]))

print(json.dumps(RESULT, indent=2, default=str))
dbutils.notebook.exit(json.dumps(RESULT, default=str))
