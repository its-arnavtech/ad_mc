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

  **RUN COMPLETE — silver populated by the cluster.** 840,000 rows
  (21 x 4 x 10,000) written by `applyInPandas` on serverless job compute.
  Module distribution to executors is via a WHEEL built from the repo's own
  files (version carries a content hash) named in the serverless environment
  spec — not `sys.path` hacking, which imports fine on the driver and then
  fails on executors. Write is `INSERT OVERWRITE` (not append, not
  `saveAsTable`): the table has no `run_id`, so append would stack a second
  grid that Phase 4 would silently average, and `saveAsTable` would discard
  the column comments by rewriting table metadata.

  **Reproduction result, stated precisely** — two different claims. (These
  figures are the CORRECTED ones; see "verifier corrections" below for what
  they replace and why.)
  - *Exactness of the wrap, cluster vs local at the pinned seed 20260808:*
    the cluster produces mean 968027.4399933233, std 169363.86043850295,
    VaR-95 715966.1342067234 — which reproduces Phase 2's committed record
    ($968,027 / $169,364 / $715,966) exactly at the precision Phase 2 actually
    recorded. Aggregates are **bitwise identical across environments**:
    three interpreters spanning scipy 1.15.1 → 1.18.0 and numpy 1.26.4 → 2.5.1
    all return the same mean and std to all 17 digits, and `math.fsum` (exact
    summation) confirms ...233 is the true mean of the vector.
    Individual PATHS do drift across environments — cluster vs local differs on
    7,843 of 10,000 paths, max abs 4.19e-09, max REL 3.06e-15, consistent with
    `scipy.stats.beta.ppf` moving in the last ulp while NumPy's PCG64 stream
    stays version-stable. Those perturbations cancel exactly in the aggregates.
    So: paths are near-exact, aggregates are exact.
  - *In-grid statistical agreement at the derived seed:* the grid's own
    (even_split, normal) cell gives mean $972,326 / std $172,945 /
    VaR-95 $710,953 / CVaR-95 $660,662. Against Phase 2's draw the correct
    comparison uses the SE of the DIFFERENCE of two independent estimates
    (sqrt(1729.45² + 1693.64²) = 2420.62), giving **+1.78 SE, p = 0.076** —
    not the +2.49 SE that an SE-of-one-estimate comparison would suggest.
    Sharper still, both draws can be scored against the exact analytic Jensen
    target $971,112.25: Phase 2 sits at −1.82 SE, the grid draw at +0.70 SE,
    so they straddle the truth and the grid draw is the more accurate of the
    two. Across all 21 allocations under `normal`, mean z = +0.209 with 1 of
    21 outside |1.96| — a textbook null, no systematic effect.

  **Sanity checks at scale, all PASS:** exactly 840,000 rows and exactly
  10,000 per cell across all 84 pairs; path_number 0..9999; zero nulls in
  every column; no non-positive spend, no negative revenue; every
  `total_spend` == $500,000; `max|roas - revenue/spend| == 0.0`; no duplicate
  (allocation, scenario, path) keys; all 21 allocations and 4 scenarios
  present; recession < normal for all 21 allocations (zero exceptions, ratios
  0.819893 to 0.828342).

  **Scenario means vs predicted `net_revenue_factor` — state the grain.**
  POOLED across all 21 allocations the ratios hold to <= 9.4e-04 relative
  (1.055006 / 0.756754 / 0.823136 vs 1.0560 / 0.7572727 / 0.8231707). At the
  `even_split` allocation alone: seasonal_peak 1.054681 (1.25e-03),
  platform_algo_change 0.754992 (**3.01e-03**), recession 0.822662 (6.18e-04).
  Per INDIVIDUAL (allocation, scenario) cell the spread is wider still —
  18 of 63 exceed 3.0e-3, worst 9.24e-03 (heavy_paid_search /
  platform_algo_change). That is ordinary Monte Carlo noise: the ratio of two
  means has a relative SE around 0.25%. Do not quote a single tight bound
  without saying which grain it applies to.

  **Honest scale note:** the full 84-cell grid runs in **7.4s** single-threaded
  locally (timed). Spark here demonstrates the distribution mechanism; it is
  not required by the workload size.

  **Known limits carried forward, deliberately not fixed here:** correlation is
  held fixed across scenarios, so recession tail risk is understated (real
  cross-channel correlation rises in a downturn); only CVR is correlated at all.

  **`verifier` (2026-08-13) against live Databricks — PASS on data, PARTIAL on
  the original prose.** Clean: engine byte-unchanged from `bcfeba3`
  (`git diff` empty; the whole phase is 3,900 insertions and 0 deletions),
  reference data, silver at scale, seeding, git hygiene, no secrets in any of
  the 7 commits. It independently confirmed three things that had only been
  asserted: (a) it downloaded the actual wheel from the UC volume and
  byte-diffed all 6 modules against git — `engine.py` is byte-identical to
  `bcfeba3`, and the content hash recomputes to the filename's
  `0.1.0.post2213007160`, so the cluster provably ran the committed code;
  (b) `INSERT OVERWRITE` idempotency is empirical, not theoretical — Delta
  history shows two real runs (v15, v16) both at 840,000 rows, never doubling,
  with `VERSION AS OF 15 EXCEPT VERSION AS OF 16` returning 0 rows;
  (c) it re-ran 7 grid cells locally at their derived seeds and matched the
  stored rows to max rel 6.6e-15, with negative controls that correctly fail.

  It also caught four prose errors in the original Phase 3 writeup, all since
  corrected above — the CODE AND DATA WERE ALWAYS CORRECT:
  1. A **false causal claim** that scipy version drift caused a 1.6e-9 gap.
     Three environments spanning scipy 1.15.1 → 1.18.0 give bitwise-identical
     aggregates; the truth is stronger than what was claimed.
  2. The 17-digit figure `968027.4399933217`, cited as Phase 2's mean,
     **is not reproducible in any environment** and appears nowhere in the repo
     outside that line. It originated in a session report, not Phase 2's
     committed record, which states dollar precision ($968,027) and matches
     exactly. `math.fsum` proves ...233 is the true mean. Do not re-introduce
     the ...217 figure.
  3. `+2.49 SE` used the SE of one estimate instead of the SE of the
     difference; the correct value is +1.78 SE.
  4. The `<= 3.0e-3` ratio bound was breached by one of its own quoted numbers
     and by 18 of 63 cells; it holds only pooled across allocations.
  Plus: grid runtime is 7.4s not ~15s, and the "100% paid_search" optimum was
  extrapolation rather than measurement.

  The distributed-run commit was originally `98c7828`; its message was amended
  to correct the same four prose figures (same pattern as Phase 2's
  `85b89a7` → `bcfeba3`). Code and data were unchanged and always correct.
- **Later:** Phase 4 (optimization / efficient frontier), Phase 5 (Workflows +
  MLflow orchestration), Phase 6 (analysis & reporting).

  **WARNING FOR PHASE 4 — the frontier is currently degenerate, and it is a
  modeling artifact, not a bug.** The engine has no saturation / diminishing-
  returns curve: within a channel `clicks = spend / CPC`, so revenue is exactly
  LINEAR in spend. Bronze gives `paid_search` an expected ROAS of 3.26x against
  1.97x for the next best (+65%), an edge large enough that concentrating also
  raises the 5th percentile. What is actually MEASURED across the 21 candidates
  x 4 scenarios: `dominant_paid_search` (80/5/5/5/5, the most concentrated
  candidate in the set) simultaneously maximises expected revenue, VaR-95 AND
  CVaR-95 in all four scenarios, and the mean-VaR Pareto set is a single point
  (1 of 21) in all four. A 100% paid_search corner is an EXTRAPOLATION from the
  model's linearity, not a measurement — no such candidate exists in the grid,
  where max spend_pct is 0.80. Only mean-VARIANCE yields a real tradeoff
  (13 of 21 non-dominated under normal / recession / seasonal_peak, 12 of 21
  under platform_algo_change). Phase 4 must add a spend response curve before
  an efficient frontier means anything; that was out of scope for Phase 3.

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
