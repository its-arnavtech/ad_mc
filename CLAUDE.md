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
    CAVEAT, corrected by the Phase 4 verifier: the bitwise-stability claim
    covers MEAN and STD only, and NOT because `np.percentile` is version-
    unstable — it is stable. Feeding the CLUSTER's own saved vector to local
    numpy 2.5.1 reproduces the cluster's ...234 exactly. The real mechanism
    is that the two VECTORS differ by ~1e-15, and a percentile is a single
    ORDER STATISTIC, so it cannot average that drift away the way mean and
    std do. Quote the dollar figure $715,966, not the 17-digit one, for VaR.
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
- **Phase 4 prerequisite (spend-saturation curve):** IN PROGRESS on branch
  `phase-4-saturation-curve` (branched off `phase-3-distributed-simulation`
  at `bfc3b46`). Fixes the degenerate frontier described below BEFORE any
  optimization work starts. Diminishing returns enter at the CPC step rather
  than as a generic curve on revenue, because the mechanism is real: buying
  more volume in a channel exhausts the cheapest inventory first and drags
  your average cost-per-click up.

      effective_mean_cpc(spend) = mean_cpc_bronze * (spend/reference_spend) ** theta_channel

  with `reference_spend = $100,000` — the even-split-per-channel point Phase 2
  and Phase 3's anchor cell already validated. At that point the multiplier is
  `1.0**theta`, exactly 1.0 in IEEE754, so the curve is self-anchoring and the
  validated numbers must still reproduce BITWISE. That is the regression test,
  not a nicety. Everything downstream (lognormal CPC noise, correlated CVR
  copula, RPC draw) is untouched; only the mean fed to the CPC fit changes.
  NOT the optimization sweep — no new candidates, no frontier search yet.

  **theta, derived from bronze impression footprint** (not hand-picked per
  channel): `theta_i = 0.12 + 0.115 * ln(impressions_max / impressions_i)` →
  display 0.120, programmatic 0.167, paid_social 0.225, video 0.259,
  paid_search 0.352. The ORDERING is earned — it is a monotone function of a
  Phase 1 column, CTR ranks the channels identically, and it matches the
  economics (paid_search is intent-bounded: only so many people search a term;
  display/programmatic have near-unbounded impression supply).
  `theta_min = 0.12` and `k = 0.115` are ASSUMPTIONS, flagged as such in
  `simulation/saturation.py`.

  **ANCHOR REGRESSION PASSES EXACTLY.** Locally, `theta=None` AND
  `theta=DEFAULT` both reproduce the `bfc3b46` engine bitwise (max abs diff
  0.000e+00 on the 10,000-path vector, compared against that engine checked
  out side by side). On the cluster the anchor cell matches Phase 2 with
  delta **0.0000000000** on mean and std and 1e-10 on VaR-95. `even_split`'s
  saturation multiplier is exactly `[1.0, 1.0, 1.0, 1.0, 1.0]`, so that
  allocation's silver rows are unchanged in all four scenarios.

  **BUT THE ANCHOR TEST IS BLIND TO THETA — do not treat it as the whole
  safety net.** Because `even_split` sits exactly on the reference point,
  `theta=None`, `theta=DEFAULT_THETA`, and even an absurd
  `theta=[0.9, 0.01, 0.5, 0.7, 0.3]` all produce BITWISE IDENTICAL results
  there. So the anchor cannot detect (a) whether saturation ran at all,
  (b) whether theta has the right values or channel alignment, or (c) the
  scenario/saturation composition order. That is not hypothetical: the
  notebook initially failed to thread theta into `simulate_cell` at all, and
  the anchor test would have passed that cleanly. What actually covers those:
  `theta_matches_driver`, the cluster-reported `even_split_multiplier`, the
  v16 before/after diff, the closed-form check in
  `saturation_moved_the_rest`, and independent per-cell re-simulation with
  failing negative controls. The anchor IS sensitive to `reference_spend`
  (ref=$50k moves results 2.1e-01 relative), so that parameter is guarded.

  **THE MODEL STOPPED BEING CORNER-SEEKING — but the discrete frontier is
  still thin, and both halves matter.** The structural fix is real: the
  continuous-simplex optimum is now strictly INTERIOR, and marginal ROAS
  varies 1.29–2.10 across allocations against a constant 3.26 before. If
  Phase 4 optimises over the simplex, that is exactly the property it needs.
  But on THIS 21-point grid the mean-VaR Pareto set is 2 points — a line
  segment you can enumerate, not a curve — so do not call it a frontier yet.
  Re-ran the full 84-cell grid on the cluster
  (840,000 rows, all checks pass; 34.6s in-notebook, 66.3s job wall clock):
  - argmax mean moved `dominant_paid_search` → **`heavy_paid_search`**;
    argmax VaR-95 and CVaR-95 moved to **`tilt_display_paid_search`**. The
    mean-maximiser and the VaR-maximiser are now DIFFERENT allocations in all
    four scenarios — that is the qualitative fix.
  - mean-VaR non-dominated count went 1 → **2 of 21** in every scenario.
  - `dominant_paid_search` fell from #1 to **#5** on mean
    ($1,469,472 → $1,011,028, −31%) and to **#14 of 21** on VaR-95
    (VaR-95 $649,801, with seven allocations still below it).
  - Marginal ROAS on paid_search at `dominant_paid_search` is **1.2887** vs
    **2.0986** at even_split (−38.6%); under the linear model it was a
    constant 3.2597 everywhere. The analytic interior optimum is strictly
    interior: 23.0% display / 48.6% paid_search / 19.0% paid_social /
    7.5% programmatic / 1.8% video.

  **FOUR CAVEATS THAT MATTER FOR PHASE 4 — do not treat this as settled:**
  1. *The MAGNITUDE of k is effectively reverse-engineered, even though the
     ordering is not.* A sweep shows the corner only stops winning between
     k=0.0865 and k=0.1269, and above that the winner keeps moving:
     k=0.16 gives `tilt_display_paid_search` (a balanced 30/30/13/13/13
     tilt, arguably the LEAST degenerate point in the sweep) and only at
     k≈0.22 does display-heavy take over. The published k=0.115 sits 71%
     of the way across a window just 0.040 wide, and only **10.3% below**
     the upper boundary where the reported winner changes. Nothing in the
     repo pins k.
     Mitigating: the result is not the flattering one (the winner still puts
     60% in paid_search), and theta is genuinely measurable in the real world
     as the slope of log(avg CPC) on log(spend) — a testable assumption.
  2. *The mean-VaR frontier is THIN — 2 of 21.* Not a grid-resolution
     artifact: on ~200 random Dirichlet allocations it is 3 of 200. Root
     cause is the near-collinearity of the two objectives —
     `corr(mean, VaR-95)` is **0.9397** averaged over the 21 allocations
     (0.9752 pooled over all 84 cells, 0.9645 on the Dirichlet set; quote
     the grain). Phase 4 should expect a short arc, not
     a rich curve; **mean-VARIANCE is where the tradeoff lives** (8-9 of 21
     non-dominated) — though note saturation SHRANK that too, 13 → 8 under
     normal, 12 → 8 platform_algo_change, 13 → 9 recession, 13 → 8
     seasonal_peak. Richer than mean-VaR, but a third thinner than before.
  3. *`reference_spend = $100,000` is NOT a neutral normalisation.* At $25k
     and $50k the argmax moves to `tilt_display_paid_search` (a balanced
     tilt, not a degenerate corner); at $300k and $500k
     `dominant_paid_search` returns as argmax mean. It also has no bronze support — observed daily spends are
     $461-$4,110 per channel and the $500k budget has no stated time grain.
  4. *A sub-reference subsidy is doing real work.* Anchoring at $100k means
     channels funded BELOW that get a CPC discount. That is $48,161 (4.8%) of
     `dominant_paid_search`'s expected revenue; strip it and it scores
     $955,909 against even_split's $965,149 — it flips from winning to losing.
     That region is extrapolation beyond what bronze can identify.

  **`verifier` (2026-08-13) against live Databricks — PASS on code and data,
  PARTIAL on the original prose.** Clean: the anchor regression (it checked
  `bfc3b46`'s engine out and ran it head-to-head — bitwise on the revenue
  vector, on `revenue_by_channel`, and on the raw CPC draws, under BOTH
  `theta=None` and `theta=DEFAULT`), the theta derivation re-derived from live
  bronze, the curve's monotonicity and every guard, and git hygiene. It
  byte-diffed the wheel from the UC volume against `git show HEAD:` — all 7
  modules identical, content hash recomputes to the filename — and confirmed
  the Phase 3 wheel does NOT contain `saturation.py` while this one does. It
  re-simulated 6 live cells locally with saturation ON (max rel 3.2e-15) with
  five negative controls that all fail loudly. The $48,161 / 4.80% subsidy
  claim reproduces to the dollar, including the sign flip.

  It caught five prose errors, all since corrected above — CODE AND DATA WERE
  ALWAYS CORRECT: (1) the VaR-95 rank was #14, not #18 (no definition yields
  18); (2) a **reintroduced false causal claim** — `np.percentile` is NOT
  version-unstable; feeding the cluster's own vector to local numpy
  reproduces its value exactly, and the real mechanism is that a percentile
  is an order statistic and cannot average away ~1e-15 vector drift the way
  mean and std do (this is the same class of error the Phase 3 verifier
  caught, made one level down — watch for it); (3) k=0.16 gives a balanced
  tilt, not display-heavy, which only wins at k≈0.22; (4) `corr(mean, VaR)`
  0.960 was the Dirichlet-set number quoted as the 21-allocation number
  (0.9397); (5) the runtime was 34.6s, not ~101s. It also flagged two things
  that had gone unsaid: the anchor test's blindness to theta, and that
  saturation SHRANK the mean-variance frontier 13 → 8.

  The saturation commit was originally `5fa9753`; its message was amended to
  correct the same five figures, and the `_spearman` tie defect plus the
  closed-form check were fixed in the same amend. Third time this pattern has
  run (Phase 2 `85b89a7` → `bcfeba3`, Phase 3 `98c7828` → `bfc3b46`): the
  code and data have been right every time and the PROSE has been wrong every
  time, so weight self-reported figures accordingly and re-derive before
  quoting them.

  **CV consequence of the mean-only spec.** `std_cpc` keeps its bronze value
  while the mean rises, so the CPC coefficient of variation FALLS as spend
  grows (paid_search: 0.1264 at $25k → 0.0476 at $400k, a 2.65x swing) and the
  Jensen `exp(sigma^2)` correction shrinks with it. Worth $2,051 (0.20%) at
  `dominant_paid_search`. A CV-preserving variant is implemented behind
  `saturate_std_cpc=True` (default OFF, per spec) and is arguably more
  defensible — it matches how Phase 3's scenario multipliers already behave,
  giving the codebase one rule instead of two. Not switched without a decision.

  **`saturation_moved_the_rest` asserts a CLOSED FORM, not a rank
  correlation.** A first draft ranked allocations by raw concentration
  (`max_share`) with threshold Spearman < -0.8 and failed at -0.51 against a
  correct model — the curve does not penalise concentration per se, it
  penalises spend away from reference WEIGHTED BY theta (0.120 to 0.352).
  Proof: all five `dominant_*` allocations sit at max_share 0.80 yet their
  before/after ratios run 0.688 (paid_search) to 0.981 (display), which no
  function of max_share can order. A theta-weighted exposure variable gets
  Spearman -0.8039 — but that clears its own threshold by only 0.0039, so it
  is now DESCRIPTIVE only. What is asserted is the exact analytic ratio
  `E[rev] = spend * (exp(sigma^2)/cpc) * cvr * rpc` evaluated with and without
  saturation: it predicts every observed ratio to **max residual 0.00064**
  with **Spearman +1.0000**. That is tight because both runs share their
  seeds, so common random numbers cancel nearly all the Monte Carlo noise.

  **A tie-handling defect was fixed in the same function.** `_spearman` used
  `argsort(argsort(x))`, which breaks ties by input order. `max_share` is
  heavily tied (five allocations at exactly 0.80), so the printed contrast
  figure took **104 distinct values across 200 shuffles of identical data**,
  spanning -0.61 to -0.38 — which is why -0.49, -0.45 and -0.4403 all appeared
  in different places. Ranks are now tie-averaged and the value is a stable
  -0.5082, matching scipy. This never affected pass/fail: the asserted
  variables have no ties.
- **Phase 4 (Optimization sweep / efficient frontier):** IN PROGRESS on branch
  `phase-4-optimization` (branched off `phase-4-saturation-curve` at
  `28f0d62`). The saturation prerequisite above is what makes this meaningful
  at all — without it the answer is a corner and there is nothing to trace.
  Populates the so-far-empty `gold` schema. NOT Phase 5 (no Workflows, no
  MLflow).

  **Two decisions the user made, both departures worth knowing about:**
  1. **COMMON RANDOM NUMBERS, not Phase 3's per-cell independence.** A
     frontier is decided by PAIRWISE DOMINANCE, and with independent seeds
     every comparison carries the full SE of a difference (~$2,600-$3,000)
     against gaps that turned out to be much smaller than the ~$18,500
     assumed at design time: realized adjacent gaps along the `normal`
     mean-VaR frontier are $9,961 / $2,714 / $6,777 / $5,971 / **$323**. So
     noise would flip dominance between close allocations. Sharing the draws within a scenario
     makes comparisons paired and cancels the common noise. This deliberately
     contradicts Phase 3's seeding note, which warned that identical streams
     across cells would invalidate cross-allocation comparison; that warning
     is about treating individual cell estimates as independent, and is the
     opposite of what pairing does for DIFFERENCES. MEASURED at 300
     replicates per arm — and the gain is NOT one number, because CRN cancels
     COMMON noise and therefore grows as two allocations get more similar,
     which is exactly the regime a frontier lives in:
     distant pair $647 vs $2,638 (**16.6x**, CI [13.2, 20.9]); near-frontier
     $561 vs $2,756 (**24.1x**); nearly identical $45 vs $2,974 (**4,370x**).
     The independent arm sits at $2,600-$3,000 regardless of the pair — it
     gets no benefit from similarity. No detectable bias (max |t| = 0.68).
     An earlier 60-replicate reading of $2,397 / 9.2x / $121 was a single
     noisy realization and understated the gain; do not requote it. Verified mechanically
     too: at a shared seed two DIFFERENT allocations draw bitwise-identical
     CVR and RPC matrices, and a zero-spend allocation still draws the same
     stream (the idle channel keeps consuming its own lognormal block, so
     dropping a channel does not shift the draws of the ones after it).
  2. **Two-stage search:** broad structured + Dirichlet coverage of the
     simplex, then local refinement around the stage-1 Pareto set, so
     resolution is spent where the frontier actually is.

  **A gap the optimizer forced:** `saturation_multiplier` raised on zero
  spend, which blocked the search from reaching the simplex boundary even
  though dropping a channel is a legitimate allocation. The correct limit is
  revenue ∝ `spend^(1-theta) → 0`, so zero spend must give zero revenue for
  that channel, not an error and not an infinity. Negative spend still raises.

  **Scale note, and why Spark finally earns its keep here.** Phase 3's 84
  cells ran in 7.4s single-threaded, so distribution was a demonstration. The
  sweep is 500-800 candidates x 4 scenarios x 10,000 paths = 20-32 MILLION
  paths, which is genuinely worth distributing. Output grain is ONE AGGREGATE
  ROW per (allocation, scenario) — mean/std/VaR/CVaR/ROAS computed inside the
  UDF — not path level, which would be tens of millions of rows with no
  consumer.

  **RUN COMPLETE. 971 candidates x 4 scenarios x 10,000 paths = 38,840,000
  paths**, 292s single-threaded locally (571 stage-1 + 400 stage-2). Gold is
  populated and independently re-queried: `allocation_sweep_results` 3,884
  rows, `efficient_frontier` 512, `frontier_recommendations` 36. Checks pass:
  no null metrics, CVaR-95 <= VaR-95 everywhere, every candidate spends the
  full budget, no orphan frontier rows.

  **THE THIN-FRONTIER PREDICTION WAS RIGHT, AND SAMPLING HARDER MADE IT
  WORSE, NOT BETTER.** Non-dominated counts of 971:

      mean vs VaR-95   6 / 5 / 4 / 5   (normal / platform / recession / seasonal)
      mean vs CVaR-95  6 / 5 / 6 / 7
      mean vs std    111 / 112 / 119 / 126

  **Only mean-VARIANCE is a curve worth the name** (~11-13%). Report the
  mean-VaR result as a short arc; do not dress it up.

  **DO NOT ARGUE THIS FROM THE PROPORTION — that inference is invalid, and the
  Phase 4 verifier disproved it.** An earlier draft said the mean-VaR set being
  0.62% of 971 versus 9.5% of 21 "settles" that the collinearity is structural.
  It settles nothing: the expected size of a 2-objective Pareto front grows
  like `ln(n) + gamma`, so the PROPORTION falls mechanically with n whatever
  the structure (the null predicts 17.4% at n=21 and 0.77% at n=971, so the
  observed drop is LESS dramatic than chance). Subsampling 21 of the 971
  candidates 2,000 times gives a median front size of **2**, with
  P(size = 2) = 48.1% — Phase 3's "2 of 21" is the MODAL outcome of drawing 21
  from Phase 4's own set, so the two results never disagreed.
  Permuting `var_95` to destroy the correlation raises the front only from
  6 to ~7.2 (against `H_971` = 7.46), so **near-collinearity explains only
  about 17% of the thinness**; the rest is the generic log(n) behaviour of any
  two-objective front. The defensible statement rests on the COUNT: 2 -> 6 as
  n went 21 -> 971 is what theory predicts, and mean-std at 111 is the one
  that sits far ABOVE the null, which is why only it is a real curve.

  **Stage 2 earned its place, measurably.** Mean-variance non-dominated went
  44 (stage 1 alone) -> 111 (both stages), and two of the three mean-VaR
  recommendations are stage-2 points -- one perturbation, one blend, and 94
  of the 111 mean-variance non-dominated points are stage-2. Stage 2's
  contribution is frontier DENSITY, not the revenue record: the best expected
  revenue found, $1,054,860 (`dir1_110`), is a **stage-1** Dirichlet candidate
  and owes nothing to refinement. It still beats the Phase 3 grid's best of
  $1,035,656 (`heavy_paid_search`) by 1.85%, so broad sampling alone reached
  allocations the fixed 21-candidate set could not.

  **A stage-2 capping defect was found and fixed.** `out[:max_candidates]`
  truncated a list with all perturbations appended before all blends, so at
  >= 25 stage-1 frontier points it kept 400 perturbations and **zero blends**
  -- deleting precisely the move that densifies a thin frontier, which is the
  one thing this model needs. (At 20 points it cut blends 180 -> 80.) The cap
  is now per-family round-robin. Blends survive and land ON the frontier.

  **A FRONTIER CAN BE NON-DEGENERATE AND STILL NOT ORDERED.** `n_efficient`
  only asks whether the frontier has an interior; it does not ask whether the
  points are far enough apart for their ORDER to be real. Realized adjacent
  gaps on the `normal` mean-VaR frontier are
  $9,961 / $2,714 / $6,777 / $5,971 / **$323**, and that last pair was measured
  directly: its own CRN difference SD is **$322**, so the gap is 1.0x the noise
  and ranks 5 and 6 are NOT ordered by this sweep. `recommend_from_frontier`
  now emits `nearest_neighbour_gap` and `ordering_unresolved` (gap < 2x the
  measured CRN noise, default $650 — the conservative end, since over-flagging
  is the safe direction). **6 of 36 picks are flagged**, including exactly the
  mean-VaR `min_risk` pick. The flag does not change which point is chosen; it
  stops the rule implying a precision the data does not have.

  **`verifier` (2026-08-14) against live Databricks — PASS on code and data,
  PARTIAL on the original prose.** Clean: zero-spend and its CRN coupling, the
  anchor regression (bitwise 0.000e+00 vs `28f0d62` under both theta modes),
  gold contents, the stage-2 fix, determinism, git hygiene, no secrets in any
  commit. It went well past the brief: it **recomputed the entire Pareto
  frontier with its own brute-force code** and matched all 12
  (scenario, objective-pair) sets by MEMBERSHIP, not just count; re-simulated
  9 gold cells at their CRN seeds to max rel 2.1e-16 with seven negative
  controls; and confirmed no Phase 4 write touched silver — `v16 EXCEPT v17`
  is exactly 800,000 rows, i.e. precisely the 40,000 `even_split` rows
  unchanged, which independently re-proves the anchor property. It also found
  the stage-2 defect was WORSE than reported: the real run had **53** stage-1
  Pareto points, not 25, so the old code would have kept 400 perturbations and
  zero blends.

  It caught six prose errors, all corrected above — CODE AND DATA WERE ALWAYS
  CORRECT: (1) an **invalid inference** arguing thinness from the falling
  PROPORTION, disproved by subsampling; (2) the CRN trio $2,397 / 9.2x / $121
  was one noisy 60-replicate realization and UNDERSTATED the gain (the cause is
  real: `independent_seed` hashes the allocation-id string, so that arm's SD
  depends on naming); (3) the subsidy headline is composition-driven — 111 of
  112 frontier points are the mean-variance frontier, and the two risk-floor
  frontiers lean on the extrapolation LESS than average; (4) $56, not "$60";
  (5) `dir1_110` is a stage-1 candidate, so the revenue record does not belong
  under "stage 2 earned its place"; (6) the design-time "$18,500 between
  frontier points" is contradicted by realized gaps of $323-$9,961.
  Fourth phase running where the code was right and the self-reported figures
  were wrong — re-derive before quoting.

  **THE SUBSIDY EXPOSURE GOT WORSE, AS PREDICTED.** The saturation curve
  discounts CPC below `reference_spend`, a region bronze cannot identify, and
  a simplex search deliberately visits it. Measured on `normal`: mean subsidy
  share is **4.58% across all 971 candidates and 5.61% on the 112 frontier
  points** (max 11.19%), and the thinnest funded channel on a frontier point
  is **0.000560x reference = $56**, extrapolating the power law ~3 orders of
  magnitude below anything observed.
  **BUT SPLIT THAT BY OBJECTIVE PAIR BEFORE REPEATING IT.** 111 of those 112
  points ARE the mean-variance frontier, so the headline is really a statement
  about that one frontier (5.64%). Broken out, the mean-VaR frontier averages
  **1.90%** and mean-CVaR **2.05%** — both BELOW the 4.58% all-candidate
  average. So "the frontier leans on the extrapolation more than average" is
  true of the VARIANCE frontier and **false of both risk-floor frontiers**,
  which lean on it less. Phase 6 should not present frontier allocations without this
  caveat, and a spend floor is the obvious mitigation.
- **Later:** Phase 5 (Workflows +
  MLflow orchestration), Phase 6 (analysis & reporting).

  **HISTORICAL — this describes the LINEAR model, FIXED by the saturation
  curve above. Kept because it explains why that curve exists, and because
  Delta `VERSION AS OF 16` of `simulated_allocation_outcomes` still holds this
  pre-saturation grid. The frontier was degenerate then, as a modeling
  artifact rather than a bug.** The engine has no saturation / diminishing-
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
