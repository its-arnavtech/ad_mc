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
  (`applyInPandas`/`mapInPandas`) over ~20 allocations x 4-5 scenarios x 10,000
  paths (~1M paths), populating the silver layer. New tables:
  `bronze.scenario_definitions`, `silver.allocation_candidates`,
  `silver.simulated_allocation_outcomes`. NOT Phase 4 (no optimization sweep)
  and NOT Phase 5 (no Workflows/MLflow). Validate the even-split/normal case
  against Phase 2's verified numbers ($968,027 / $169,364 / $715,966).
- **Later:** Phase 4 (optimization / efficient frontier), Phase 5 (Workflows +
  MLflow orchestration), Phase 6 (analysis & reporting).

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
