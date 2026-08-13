# Ad Revenue Monte Carlo POC — Project Context

## What this project is

Portfolio project applying portfolio-style Monte Carlo / Value-at-Risk
simulation to advertising channel budget allocation, on Databricks (Delta
Lake, Workflows, MLflow). Ad channels are treated like portfolio assets:
channel ROAS is "return," cross-channel correlation is "risk." Goal: an
efficient frontier of budget allocations by risk tolerance.

Structured to demonstrate three roles in one system:
- **Data Engineering** — Delta Lake pipeline, Databricks Workflows orchestration, Spark batch parallelism
- **Data Science** — stochastic simulation design, correlated Monte Carlo modeling, allocation optimization, MLflow tracking
- **Data Analysis** — exploration, visualization, risk/return recommendation reporting

## Current status

- **Phase 1 (Data Foundation):** complete, committed (`39c750c`) on branch
  `phase-1-data-foundation`. Bronze layer validated: `channel_performance_history`,
  `channel_assumptions` (extended with `mean_cpc`/`std_cpc`), `channel_correlation_matrix`.
- **Phase 2 (Single-Node Simulation Engine):** complete, committed (`43c9163`)
  on branch `phase-2-simulation-engine` (branched off `phase-1-data-foundation`).
  Correlated CVR Monte Carlo validated against an analytic Jensen's-inequality
  target and a negative-control correlation check.
- **CVR correlation matrix:** complete, committed (`bcfeba3`) on branch
  `phase-2-simulation-engine`. The revenue-based proxy is replaced by a
  dedicated `channel_cvr_correlation_matrix`; the revenue matrix is retained as
  a diagnostic and no longer feeds the sim. `verifier` (2026-08-08) against live
  Databricks: bronze checks PASS — independent CVR recompute matches the stored
  table to 8.9e-16, both matrices positive-definite, engine wired to
  `CORRELATION_SOURCE="cvr"`, correction claims (mean off-diag 0.371 revenue vs
  0.406 CVR, higher on 18/20 pairs) reproduce exactly. Engine A/B reproduction
  PASS — expected/std/VaR/CVaR and the +14.9%/+16.6% floor comparison reproduce
  to the dollar at seed 20260808. The commit was originally `85b89a7`; its
  message was amended (→ `bcfeba3`) to correct three self-reported prose figures
  the verifier caught (moment-mean 0.42%→0.45%, independence baseline
  0.5477→0.5278, separation 34.7x→33.5x); code was unchanged and always correct.
- **Branch strategy:** phases stay on separate branches; all merge into `main`
  together after all phases are done. Don't merge early without being told to.
- **Phase 3 (Distributed Batch Simulation):** IN PROGRESS on branch
  `phase-3-distributed-simulation` (branched off `phase-2-simulation-engine` at
  `bcfeba3`). Distributes the validated Phase 2 engine across Spark
  (`applyInPandas`) over 21 allocations x 4 scenarios x 10,000 paths
  (840,000 paths), populating the silver layer. New tables:
  `bronze.scenario_definitions`, `silver.allocation_candidates`,
  `silver.simulated_allocation_outcomes` — all three created and confirmed live
  with the spec'd schemas, unpartitioned (at ~840k small rows Hive partitioning
  would make 80 tiny files and slow Phase 4's full-table aggregate down).
  NOT Phase 4 (no optimization sweep) and NOT Phase 5 (no Workflows/MLflow).

  **COMPUTE — read this before touching Phase 3 plumbing.** Phases 1-2 needed no
  cluster because they only ran SQL. `applyInPandas` is a Python UDF and needs
  real Spark, and this workspace constrains how that can happen:
  - **Databricks Connect cannot run Python UDFs here.** All four paths
    (`applyInPandas`, `mapInPandas`, `pandas_udf`, plain `udf`) fail with
    `[ISOLATION_STARTUP_FAILURE.SANDBOX_STARTUP]` /
    `failed to load /databricks/python3/bin/python: exec format error` — a
    Databricks-side container fault, not a repo bug. SQL over Connect is fine.
    `simulation/spark_session.py` is kept but is a dead end for UDFs.
  - **There is no classic compute plane.** `clusters.list_zones()` fails with
    `No such workerEnvironment 'serverless-...'`; all-purpose cluster creation
    hangs 5 min then times out. `databricks/05_create_phase3_cluster.py` is
    unusable in this workspace — do not run it.
  - **What works:** a notebook uploaded to the workspace and executed by a
    transient `jobs.submit` run on serverless job compute (`Environment(client='4')`).
    Verified: `applyInPandas` runs there. Server side is Python 3.12.3 /
    numpy 2.1.3 / scipy 1.15.1, vs local 3.13 / 2.5.1 / 1.18.0. User approved
    this transient submission as NOT a "Workflow" — it is a one-off run, not a
    persisted job definition, schedule, or DAG. Phase 5 still owns Workflows.

  **Scenario multipliers are ASSUMPTIONS, not measured values** (cvr / cpc /
  revenue, with net factor `cvr*rev/cpc`): `normal` 1.00/1.00/1.00 (1.0000);
  `seasonal_peak` 1.20/1.25/1.10 (1.0560 — the auction competes away most of a
  demand shock); `platform_algo_change` 0.85/1.10/0.98 (0.7573 — the only
  scenario with no offsetting relief); `recession` 0.75/0.82/0.90 (0.8232 —
  CVR and CPC fall together, since the contraction that stops consumers buying
  also stops advertisers bidding).

  **Seeding:** `blake2b(id, 8 bytes)` → stable ints, combined with the master
  seed 20260808 through `numpy.random.SeedSequence`. Builtin `hash()` is
  deliberately avoided — it is salted per process and would break reproducibility
  across executors (verified: it returned different values in two interpreters).
  The grid is **uniformly derived — no pinned anchor cell.** That keeps two
  claims separable: "wrapping didn't change the math" is proven exactly by a
  side test at seed 20260808, while "the estimate is seed-stable" is proven
  statistically by the grid's own derived-seed anchor.

  **The wrapped engine reproduces Phase 2 bitwise** (max abs diff 0.000e+00 on
  the 10,000-path revenue vector) when run locally at seed 20260808, because
  `normal`'s multipliers are exactly 1.0 and `engine.simulate()` is called
  unmodified, preserving its RNG draw order.

  **Known limits carried forward, deliberately not fixed here:** correlation is
  held fixed across scenarios, so recession tail risk is understated (real
  cross-channel correlation rises in a downturn); only CVR is correlated at all.
- **Later:** Phase 4 (optimization / efficient frontier), Phase 5 (Workflows +
  MLflow orchestration), Phase 6 (analysis & reporting).

  **WARNING FOR PHASE 4 — the frontier is currently degenerate, and it is a
  modeling artifact, not a bug.** The engine has no saturation / diminishing-
  returns curve: within a channel `clicks = spend / CPC`, so revenue is exactly
  LINEAR in spend. Bronze gives `paid_search` an expected ROAS of 3.26x against
  1.97x for the next best (+65%), an edge large enough that concentrating also
  raises the 5th percentile. Measured across all 21 candidates x 4 scenarios,
  100% paid_search simultaneously maximises expected revenue, VaR-95 AND
  CVaR-95 — so the mean-VaR Pareto set is a single point and a corner walk
  confirms the optimum is at the corner, not interior. Only mean-VARIANCE
  yields a real tradeoff (13 of 21 non-dominated). Phase 4 must add a spend
  response curve before an efficient frontier means anything; that was out of
  scope for Phase 3.

## How this project is orchestrated

This repo uses four specialized subagents, all reporting to this main session:

- `data-engineer` — Delta Lake schema, Databricks Workflows, Spark batch processing
- `data-scientist` — stochastic modeling, simulation engine, optimization logic, MLflow
- `data-analyst` — exploratory analysis, visualization, reporting
- `verifier` — read-only. Independently re-checks completed work against the
  live Databricks workspace before anything is reported as "done." Never
  trust a subagent's own self-report without the verifier confirming it
  against real queries.

### Rules for this main session

1. Route work to the matching subagent rather than doing broad implementation
   work yourself in this context.
2. After any subagent finishes work that touches Databricks or the repo,
   delegate to `verifier` before telling the user it's done. Report the
   verifier's actual findings (pass/fail/partial with evidence), not just a
   summary claim.
3. Never silently fix something the verifier finds wrong — report it and ask
   first, same as every previous phase in this project.
4. Keep this file's "Current status" section updated as work completes,
   since every subagent's context starts fresh and this file is what gives
   it continuity.
5. Don't merge branches without being told to — current plan is to merge
   everything together after Phase 3.
