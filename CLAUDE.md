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

## Project complete (2026-08-19)

All six phases plus the two mid-phase modeling fixes are merged into `main` by
fast-forward on 2026-08-19, via PR
[#4](https://github.com/its-arnavtech/ad_mc/pull/4) (auto-closed as MERGED when
main was fast-forwarded past its head). Every phase kept its committed SHA:

| phase | tip commit | what it delivered |
|---|---|---|
| 1 — Data Foundation | `39c750c` | Bronze Delta tables: channel history, assumptions, revenue correlation |
| 2 — Simulation Engine | `43c9163` | Single-node correlated Monte Carlo, Jensen-validated |
| CVR correlation fix | `bcfeba3` | Dedicated `channel_cvr_correlation_matrix`, engine rewired to it |
| 3 — Distributed Batch | `bfc3b46` | Wheel + `applyInPandas` on serverless, 840,000 silver rows |
| 4 prerequisite — Saturation | `28f0d62` | Spend-elasticity curve anchored at $100k, opt-in; anchor bitwise |
| 4 — Optimization Sweep | `9df9c1c` | 971 candidates × 4 scenarios × 10,000 paths = 38.84M paths, gold populated |
| 4 follow-up — Spend Floor | `a289016` | p5 daily-spend floor on the CPC curve, `extrapolation_floor_applied` flag |
| 5 — Orchestration | `accf127` | Persisted Workflow job `94493651519110`, one MLflow run per job run |
| 6 — Analysis & Reporting | `3ee1b9c` | Live-gold-backed HTML report + 4th gold table + regression tests + CI |

Merge tip on `main`: **`3ee1b9c`**, identical to the tip of
`phase-6-analysis-reporting`. This CLAUDE.md summary sits one commit further on
top of the merge, so a reader today sees `main` at that summary commit rather
than at `3ee1b9c` — nothing before it has been rewritten.

Post-merge verification (2026-08-19, independent verifier pass):
- `main` fast-forwarded from `1ad64d1` to `3ee1b9c`; PR #4 auto-closed as MERGED with merge commit = `3ee1b9c`
- Topology: `main`'s history is fully linear, every commit has exactly one parent — no synthetic merge commit
- 13 pytest contracts pass against `main` HEAD (recomputed in a worktree pinned to the summary commit)
- GitHub Actions "Python checks" ran and succeeded on the merge push (run `32303819281`, 46s) AND on the summary-commit push (run `32304081142`, 43s) — both green
- Live gold spot-check against Databricks matches every headline figure the report shows (best normal `$1,054,690.6952`; frontier sizes 6/7/8/9, 8/8/9/11, 107/109/114/123; 11/16 recs flagged; 180 contribution rows)
- Full-history secret scan across all 51 files reachable from main: zero hits across databricks / AWS / GitHub / Slack tokens, PEM headers, and inline literal assignments
- Every one of the eight phase branches is reachable from `main` and therefore safe to delete; branches left in place per instruction

**Eight branches** hold the phase record — Phase 4 is split into a saturation
prerequisite, an optimization sweep, and a spend-floor follow-up:
`phase-1-data-foundation`, `phase-2-simulation-engine`,
`phase-3-distributed-simulation`, `phase-4-saturation-curve`,
`phase-4-optimization`, `phase-4-spend-floor`, `phase-5-orchestration`,
`phase-6-analysis-reporting`. All are ancestors of `main` and can be deleted
safely whenever the owner chooses.

Everything below this line is the phase-by-phase build log kept during
development — retained so the reasoning trail (verifier findings, prose
corrections, motivated-reasoning caveats) is not lost. The tables above are
the compact story; this is the receipts.

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
- **Phase 3 (Distributed Batch Simulation):** COMPLETE on branch
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
- **Phase 4 prerequisite (spend-saturation curve):** COMPLETE on branch
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
- **Phase 4 (Optimization sweep / efficient frontier):** COMPLETE on branch
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
- **Phase 4 follow-up (spend floor on the CPC curve):** on branch
  `phase-4-spend-floor` (branched off `phase-4-optimization` at `9df9c1c`).
  Bounds the saturation curve at the low end so it stops extrapolating CPC
  below anything bronze has observed. NOT Phase 5.

  **Floors are p5 of each channel's observed DAILY spend** over 730 days:
  display $311, programmatic $552, paid_social $834, video $1,300,
  paid_search $2,755. p5 sits above 37 of the 730 daily observations — far
  enough into the tail to be the honest edge of the evidence, but not the
  single minimum, which is one day in two years and would make the bound
  hostage to an outlier. p1/p5/p10 clip **17.20%/17.82%/18.33%** of the 971
  candidates, a spread of 1.1 points, so the choice governs how far clipped
  candidates move rather than how many. p5 is an ASSUMPTION, same class as
  `theta_min` and `k`. (An earlier draft said 28.7%/29.4%/30.0%; those counted
  zero-spend channels as clipped, which the rule explicitly does not.)

  **The INPUT is clipped, not the output.** Below the floor the curve is
  evaluated AT the floor. Two consequences worth stating:
  - `clicks = actual_spend / drawn_CPC` still uses REAL spend, so a clipped
    channel buys what its budget affords at a defensible price — it is not
    credited with the floor's budget.
  - **Zero spend is NOT clipped.** A channel at zero is unfunded, not
    underfunded; lifting it to the floor would invent a channel the
    allocation never bought. Only strictly positive sub-floor spend clips.

  **Anchor regression is BITWISE in all four combinations** (theta on/off x
  floor on/off), max abs diff 0.000e+00 vs `9df9c1c`, and the anchor's floor
  mask is all-False. The largest floor is 0.0276x reference.
  **But the anchor is BLIND to the floor**, exactly as it is blind to theta:
  every floor is <= 0.028x reference, so at the anchor `floor=None`,
  `floor=BRONZE` and `floor=[1,1,1,1,1]` are indistinguishable. It only becomes
  sensitive above the reference. What actually covers the floor is the per-cell
  re-simulation, the independent flag reconciliation, and the unclipped-vs-
  clipped split below.

  **THE FIX BOUNDS THE PRICING, NOT THE SEARCH — read this before claiming it
  removed the extrapolation.** Measured, `normal`, 971 candidates:
  - Allocations with no clipped channel changed by **exactly $0.00**. The
    floor is surgical.
  - Clipped allocations lose 0% to 1.69% of expected revenue (mean −0.48%).
  - **The thin points are still on the frontier.** Pre-floor the thinnest
    funded channel on a mean-std frontier point was $56 (0.00056x ref);
    post-floor it is **$1** (0.00001x ref). The floor stops such a channel
    being PRICED at an ungrounded CPC; it does nothing to stop the optimizer
    PROPOSING it. Bounding the search itself would be a separate change — a
    constraint in the candidate generator or a penalty term.
  - **The "0% to 1.69%" figure UNDERSELLS what was removed — it is small only
    because thin channels hold tiny budgets.** Per channel the singularity was
    enormous: at $1 in paid_search the unbounded curve implies a marginal ROAS
    of **3,905x** against 3.26x at reference, and at $56 it implies 98x. The
    distortion is amplified by the mean-only `std_cpc` spec, which leaves the
    fitted sigma at 1.744 and the Jensen factor `exp(sigma^2)` at **20.9** at
    $1, versus 1.08 at the floor. The floor caps that channel at 12.4x. So the
    fix removes an unbounded per-channel singularity; the aggregate revenue
    effect is small because the singularity sits on tiny budgets.
  - What REMAINS ungrounded: even at the floor a channel is priced 2.0x-3.5x
    cheaper than its bronze mean CPC, and the whole interval between the floor
    and the reference is the interval bronze cannot identify.
  - So the honest claim is: the numbers on the frontier are now defensible,
    and where they still rest on the edge of the data the row says so.

  **RECOMMENDATION CHANGES ARE ALMOST ENTIRELY INDIRECT — through the adaptive
  search, not by re-ranking a fixed set.** 17 of 36 recommendations changed.
  Holding the candidate set FIXED at the 799 common candidates the repricing
  alone gives mean-VaR 6 -> 6 (all kept), mean-CVaR 5 -> 5 (all kept), mean-std
  76 -> 77 (75 kept) — so re-ranking a fixed set moves almost nothing. What
  actually moved the answer is that the repricing shifted the stage-1 Pareto
  union 53 -> 51, which changed the points stage 2 blends around, so 172 of 971
  candidates differ between runs and 64% of frontier membership turned over.
  **Do NOT call the search path a separate cause or a confounder** — an earlier
  draft did, and that treats a downstream consequence of the fix as if it were
  exogenous. The floor is the only input that changed.
  Best expected revenue is `dir1_110` both before and after
  ($1,054,860 -> $1,054,691, itself a clipped allocation).

  **Gold carries the caveat, same pattern as `ordering_unresolved`:**
  `extrapolation_floor_applied` on all three tables plus `n_channels_floored`
  on the sweep. Live: 692 of 3,884 sweep rows flagged (173 candidates x 4
  scenarios), 50 of 519 frontier rows, **16 of 36 recommendations**.
  `ordering_unresolved` moved 6 -> 11 of 36. That is NOT "unrelated to the
  floor" (an earlier draft said so and it is wrong — the floor is the only
  changed input). Decomposed: on the fixed 799 common candidates the repricing
  moves it 5 -> 1, i.e. DOWN; the rise to 11 comes entirely from the different
  stage-2 blend set, which the floor caused. Same indirect mechanism as the
  recommendation changes above. Phase 6 must respect both flags.

  **`verifier` (2026-08-15) against live Databricks — PASS on data, PARTIAL on
  the prose.** It re-ran the entire two-stage sweep TWICE itself (floor on and
  off) and reproduced live gold bitwise on all 3,884 cells, all 12 frontier
  sets by membership, and all 36 recommendations. It independently recomputed
  the flags from its own bronze-derived floors: 0 mismatches on every row, with
  five negative controls that discriminate. CRN determinism re-verified on 12
  live cells including 6 clipped ones (max rel 0.000e+00) with 7 failing
  negative controls. Silver confirmed untouched, and the floor is a NO-OP on
  the whole Phase 3 grid (its minimum channel spend is $25,000, above every
  floor), so Phase 3's guarantee is intact.

  It found four real defects, all since fixed: (1) `check_spend_floor_snapshot`
  was never called AND its 1e-9 default tolerance made it raise on correct data,
  since the frozen floors are deliberately rounded to the dollar (7.4e-04) — the
  tolerance is now 1e-3 and `load_phase4_gold` calls it; (2) the p1/p5/p10 clip
  figures were wrong (see above); (3) `analytic_expected_revenue_weights`,
  `subsidy_decomposition`, `marginal_roas` and `analytic_interior_optimum` were
  floor-UNAWARE, so the repo's own analytic cross-checks disagreed with the
  engine by +0.41% mean on exactly the clipped group — now floor-aware, and the
  clipped-group bias is -0.00094 against the unclipped group's -0.00095, i.e.
  gone; `marginal_roas` is now a central difference on the analytic revenue so
  it cannot drift again, and it still reproduces the prerequisite's published
  values exactly; (4) `SweepInputs(spend_floor=np.zeros(5))` was ACCEPTED and
  silently reverted to unbounded behaviour — degenerate floors now raise. Plus:
  the new gold columns carried no COMMENT because `ALTER TABLE ADD COLUMNS`
  omitted them and no repo script performed the migration; `08_create_gold_
  frontier.py` now re-applies every column comment idempotently on each run.

  **OPEN ISSUE AGAINST THE SATURATION PREREQUISITE — do not act on it without a
  decision, it would invalidate the whole Phase 4 sweep.** The verifier
  regressed log(CPC) on log(spend) in bronze and got slopes that are
  essentially UNIFORM: display 0.131, programmatic 0.123, video 0.123,
  paid_social 0.102, paid_search **0.105** (se ~0.012, all t > 8). Published
  theta runs 0.120 to **0.352**, and paid_search — the channel whose high theta
  breaks the corner — measures the LOWEST slope of the five. Separately, every
  channel's entire observed daily spend range sits below **0.07x reference**,
  so the curve evaluated where bronze actually measured CPC predicts one-half
  to one-third of the CPC observed there. `reference_spend = $100,000` is
  therefore not merely unsupported at the daily grain, it is contradicted by
  it. CLAUDE.md claims theta is "genuinely measurable as the slope of log(avg
  CPC) on log(spend) — a testable assumption"; run against this project's own
  bronze, that test does not corroborate the published values. Caveat on the
  caveat: daily spend varies only ~2-3x within a channel, which may be too
  little leverage to identify a per-channel elasticity, so this is arguably
  "unconfirmed" rather than "refuted". Either way the ordering story is not
  currently supported by data, and Phase 6 must not present theta as measured.

- **Phase 5 (Orchestration & Tracking):** COMPLETE on branch
  `phase-5-orchestration` (branched off `phase-4-spend-floor` at `a289016`).
  Wires the ALREADY-VALIDATED pipeline into a persisted Databricks Workflow and
  instruments it with MLflow. NO modeling changes: simulation math, theta, the
  spend floor and the CRN scheme are untouched, and the open theta-vs-bronze
  question stays open for Phase 6.

  **Two structural facts that shape the DAG, from inventorying the repo:**
  1. The Phase 4 sweep runs LOCALLY today (`load_phase4_gold.py`, single
     threaded, writing to gold over the SQL API). A Workflow task runs ON
     Databricks, so the sweep has to move there. Phase 3 already built the
     machinery for exactly that — a wheel built from the repo's own modules,
     named in a serverless environment spec — so Phase 5 reuses it rather than
     inventing a second mechanism.
  2. The search is ADAPTIVE, so this cannot be one flat parallel job: stage 2
     refines around stage 1's computed Pareto union. That forces a real
     dependency edge, and it forces intermediate state to cross a task
     boundary (separate processes), which a UC volume carries.

  Phase 5 also needs a PERSISTED job definition, which is the first time this
  project creates one — Phases 3 and 4 deliberately used transient
  `jobs.submit` runs to stay inside the "no Workflows before Phase 5" rule.

  **DEPLOYED AND RUN. Job `94493651519110`** ("ad_mc_poc -- frontier pipeline
  (Phase 5)"), 5 tasks, serverless job compute. Runs: `529952200505846` FAILED,
  then `407499289143226` and `322037088744573` both SUCCESS in ~240s.
  DAG: `bronze_refresh -> stage1_simulate -> stage2_generate -> stage2_simulate`,
  with `gold_aggregate` fanning in from BOTH `stage1_simulate` and
  `stage2_simulate` (it merges the two stages). Stage-1 sweep 41.6s for 2,284
  cells, stage 2 33.5s for 1,600 — against 292s single-threaded locally in
  Phase 4, so `applyInPandas` chunking (~32 cells per chunk, 72/50 chunks) is
  doing real work here.

  **Run 1's failure is worth keeping.** `TypeError: Object of type PlanMetrics
  is not JSON serializable`, thrown by `pdf.to_parquet()` AFTER all 22.8M paths
  had been simulated. `DataFrame.attrs` is written into parquet key-value
  metadata as JSON, and Databricks attaches a `PlanMetrics` object to every
  frame `toPandas()` returns. Fixed by clearing `attrs` on a copy before write.

  **Parameters** — 7 job parameters, defaults exactly the `a289016` values,
  threaded job parameter -> task `base_parameters` (`{{job.parameters.*}}`) ->
  notebook widget -> `resolve_params()` -> typed config. Widget defaults are
  EMPTY on purpose: `resolve_params` raises on an empty or still-templated
  value, so a parameter that fails to resolve fails the task instead of
  silently falling back to a notebook-local default.
  `total_budget` 500000.0 / `master_seed` 20260808 / `n_paths` 10000 /
  `stage1_dirichlet_total` 450 / `stage2_cap` 400 / `catalog` / `mlflow_experiment`.
  **Naming departure, deliberate:** there is no `stage1_candidates` parameter.
  Stage 1 is 21 Phase 3 + 1 centroid + fixed {5,3} and {5,4} lattices (35 + 70)
  + three Dirichlet blocks; only the last is a real knob, so a parameter called
  `stage1_candidates` would name something it cannot set. `stage1_dirichlet_total`
  is apportioned by largest remainder and at 450 returns exactly
  `((1.0,225),(0.3,125),(4.0,100))`, reproducing all 571 stage-1 candidates with
  ids, families and spend tuples equal under exact float `==`.

  **Three refactors, all reported, none touching modeling.** `git diff` shows
  `engine.py`, `saturation.py`, `cell.py`, `seeding.py`, `sweep_seeding.py`,
  `scenarios.py`, `frontier.py`, `allocations.py`, `config.py` byte-unchanged.
  1. `run_phase3_distributed.build_wheel()` gained `modules`/`required` kwargs
     and `upload_notebook()` gained `source`/`base_name` — NECESSARY because
     Phase 5 needs `frontier.py` + `sweep_seeding.py` on executors and five
     notebooks, and a second wheel builder would be the wrong answer. The
     builder's default-output equality with HEAD was self-reported and has NOT
     been independently re-verified; what has been verified is the deployed
     wheel's CONTENTS (below).
  2. `simulation/gold_assembly.py` (new), with `load_phase4_gold.py` delegating
     to it — NECESSARY because the Workflow must build identical gold in a
     different process and two copies would drift. Rebuilding all three frames
     from the Phase 4 cache matches live gold bitwise on every value.
  3. `simulation/bronze_sql.py` (new), with `03_build_assumptions.py` importing
     it — NECESSARY so `bronze_refresh` runs the SAME SQL via `spark.sql`
     rather than a retyped copy. All three statements byte-identical to HEAD's.
  Plus one addition beyond the brief: `gold_assembly.canonical_cell_order`,
  because Spark returns cells in task-completion order while the frontier
  helpers break exact ties by row position.

  **MLflow** — experiment `/Users/its.arnavk.here@gmail.com/ad_mc_poc`
  (id `1983482982160806`), ONE run per Workflow run, created in `bronze_refresh`
  and passed forward via `dbutils.jobs.taskValues`, resumed by later tasks.
  Run `376e1a6f3ad74f1082e8b7df763bc55c`: 36 params, 51 metrics, FINISHED.
  theta and spend_floor logged PER CHANNEL from `saturation` (0.12/0.352/0.225/
  0.167/0.259 and 311/2755/834/552/1300), 12 `frontier_size__*` metrics,
  `total_paths_simulated` 38,840,000, `stage1_pareto_union` 51, flag counts AND
  percentages, `wall_clock_s__*` for all five tasks, and three real artifacts
  (two frontier PNGs, `frontier_recommendations.csv` 36 x 14 carrying both flags).

  **REPRODUCTION vs `a289016`: every DECISION identical, values drift 1-2 ulp.**
  Reference captured before the run and re-recovered via Delta time travel
  (sweep v78 / frontier v14 / recs v8).
  - Identical: 3,884 / 519 / 36 rows; every `allocation_id`, `scenario_id`,
    `family`, `objective_pair`, `recommendation`, `seed`; `rank_by_return`
    519/519; **all three boolean flags, 0 mismatches** (692 / 50 / 16 floored,
    11 of 36 `ordering_unresolved`); frontier and recommendation key sets;
    `stage1_pareto_union` 51; and best-normal `dir1_110` at
    1054690.6952344836 — bitwise equal.
  - Different: 3,880 of 3,884 rows differ in at least one float, largest
    absolute difference anywhere **$8.4e-09**.
  - The split is the mechanism, not noise: AVERAGING columns are bitwise on
    ~63-67% of rows (mean_revenue 64.9%, max rel 5.2e-16) while ORDER
    STATISTICS are bitwise on only ~25-32% (var_95 24.8%, max rel 2.4e-15).
    Averaging cancels ~1e-15 path drift; a percentile cannot.
  - Cause is the ENVIRONMENT, not the Phase 5 code path. 12 live cells
    re-simulated locally at their CRN seeds match the `a289016` reference
    **60/60 bitwise** — the reference was produced by this interpreter — while
    matching the cluster's gold on only **25 of 60** (an earlier draft said
    30/60; that figure is not reproducible and the exact count depends on
    which cells you sample, so quote the structure, not the number).
    **THE VERSION ATTRIBUTION WAS TESTED, NOT ASSUMED — and unlike Phase 3 and
    Phase 4, it SURVIVES the test.** Two LOCAL environments differing only in
    package versions (py3.13.14 / numpy 2.5.1 / scipy 1.18.0 versus
    py3.12.13 / numpy 1.26.4 / scipy 1.17.1) disagree on **10 of 60** values
    on the same machine, same code, same seeds. So package versions
    demonstrably do move these numbers. That is the exact test the earlier two
    causal claims FAILED (three environments there gave bitwise-identical
    aggregates, which is what disproved them). Executors are py3.12.3 /
    numpy 2.1.3 / scipy 1.15.1. Still **no specific special function is named
    as the mechanism** — that remains untested.
  - Determinism: runs 2 and 3 are bitwise identical on all three tables
    (54,376 / 3,114 / 252 float values), and the four intermediate parquet
    files are sha256-identical between runs.

  **THIS REFINES A PHASE 3 CLAIM.** Phase 3 states aggregates are "bitwise
  identical across environments" — that was measured on ONE anchor cell.
  Across 3,884 cells `mean_revenue` is bitwise identical on only 64.9%; the
  rest differ by <= 2 ulp. The Phase 3 statement is true of its sample, not of
  the population. Do not generalise it.

  **VERIFICATION (2026-08-16), recomputed against live Databricks.** The
  independent pass was cut short by an account limit, so the remaining items
  were re-derived directly rather than re-read. PASS: the drift table
  reproduces exactly (3,880 of 3,884 rows differ; averaging columns bitwise on
  63.3-67.4%, order statistics on 24.8-32.2%; max abs anywhere 8.382e-09);
  decisions are 100% identical to the reference (0 mismatches on flags, seeds,
  family/stage, frontier key sets, ranks, and all 36 recommendation picks);
  runs 2 and 3 are bitwise deterministic (0 rows differing, max abs 0.0,
  `EXCEPT` empty both directions); all gold column comments survived
  `INSERT OVERWRITE`; the three job-compute bronze rebuilds are identical, and
  `gold_assembly` rebuilds the reference at **19,420 / 19,420 floats bitwise**;
  stage 1 reproduces 571 and stage 2 reproduces 400 under exact float `==`
  with a Pareto union of 51.

  **WHEEL: "byte-identical" was IMPRECISE.** All 12 modules in the deployed
  wheel match `b2e76e5`, but four (`cell.py`, `engine.py`, `frontier.py`,
  `saturation.py`) are NOT raw-byte identical — each differs by exactly its
  line count, because the wheel is built from the CRLF working tree while
  `git show` emits LF. All four are **AST-identical**. State it as "identical
  after line-ending normalisation", not "byte-identical".

  Not independently re-verified: `build_wheel()`'s default-output equality, and
  `bronze_sql`'s statements as STRINGS (their OUTPUT is verified identical, via
  the three matching bronze rebuilds and a local re-simulation from live
  post-refresh bronze that reproduces the reference 60/60 bitwise — which is
  impossible if the refresh had altered bronze).

  **Two gaps, both flagged rather than papered over:**
  1. ~~A FAILED task leaves its MLflow run in status RUNNING forever.~~
     **CLOSED.** This was not merely untidy: the stuck run carried 9 metrics
     and ALL 9 keys overlap the ones a successful run logs (`bronze__*`,
     `wall_clock_s__bronze_refresh`), so any aggregate over the experiment that
     did not filter on status silently mixed a partial run into the numbers.
     Run 1's `60fb32bb3c1e4b019e72ccd9a5bfe6c2` has been terminated as FAILED
     (the experiment now reads FAILED / FINISHED / FINISHED, zero RUNNING), and
     `phase5_tasks.mark_run_failed` plus a `try/except BaseException` in all
     five notebooks makes it self-healing. `mark_run_failed` swallows its own
     errors on purpose — it runs inside an exception handler, and masking the
     real failure with a bookkeeping error would be strictly worse.
     **The safeguard was PROVEN, not just shipped:** a throwaway MLflow run was
     created in RUNNING, a probe notebook failed exactly as a guarded task
     would, the task failed as intended (the re-raise is preserved) and the run
     ended **FAILED**. The throwaway run was then deleted.
  2. `load_phase4_gold.py` was NOT executed end to end after being refactored
     to delegate to `gold_assembly`, because doing so would overwrite the
     Workflow's gold. Its frame building is proven bitwise-identical in
     process and its column lists proven `==` to the originals, but the script
     itself has not been run since the change. Left as-is deliberately: the
     verifier's recommendation, and the risk is now small because the delegated
     logic is proven to rebuild the reference at 19,420/19,420 floats bitwise —
     what remains unexercised is only the script's own orchestration, which is
     otherwise unchanged from `a289016`.
  Also: `gold_aggregate` asserts the gold tables exist rather than creating
  them, so DDL stays owned by `08_create_gold_frontier.py` and its column
  comments are never rewritten. On a fresh workspace that script runs first.
  `spark.sql.adaptive.coalescePartitions.enabled` is REFUSED on serverless
  (`CONFIG_NOT_AVAILABLE`); `shuffle.partitions` is accepted.
- **Phase 6 (Analysis & Reporting):** on branch `phase-6-analysis-reporting`
  (branched off `phase-5-orchestration` at `accf127`).
  `reporting/build_report.py` queries four gold
  tables at render time and emits a self-contained HTML report for a budget
  decision-maker; `reporting/report_template.py` holds the markup so numbers
  and prose cannot drift into each other.

  **Approved channel-attribution extension (2026-08-18):** the original three
  gold tables were allocation-grain, so a strict channel contribution analysis
  was impossible from their persisted schema. With explicit user approval,
  `databricks/09_create_channel_contributions.py` adds the fourth gold table,
  `recommendation_channel_contributions`, and
  `simulation/load_channel_contributions.py` reruns only the 24 unique
  allocation/scenario cells referenced by the 36 recommendations. It uses each
  cell's original seed and path count plus exact candidate spends from matching
  Phase 5 Workflow run 322037088744573. It rejects any aggregate
  mean/std/VaR/CVaR/ROAS mismatch before writing and does not alter the existing
  three gold tables. Expected contributions are channel path means; risk uses
  Euler component volatility, Cov(channel,total)/Std(total). Both channel sums
  are reconciled to authoritative gold. The report itself remains query-only:
  no candidate generation, RNG or simulation is called during rendering.

  **LIVE GOLD DOES NOT MATCH THE FIGURES IN THE PHASE 6 BRIEF.** The brief said
  the mean-VaR/CVaR frontiers are "6 of 971 (~0.62%)". Live gold, in
  seasonal/normal/recession/platform order: mean-VaR **9 / 7 / 8 / 6** and
  mean-CVaR **11 / 8 / 9 / 8**; 0.62% is only platform_algo_change. (An earlier
  draft listed 7/6/8/9 and 8/8/9/11 against that label — right values, wrong
  order.)
  Mean-variance is **107-123 of 971 (11.02-12.67%)**, which does match the
  brief's "~11-13%". The brief quoted the pre-spend-floor Phase 4 values; gold
  now holds the Phase 5 run. The report uses the live numbers.

  **Findings the report leads with, all verified against live gold:**
  - Only the mean-VARIANCE frontier is a real curve. The thin risk-floor
    frontiers are framed as **expected sampling behaviour** (Pareto front size
    grows like ln(n)) and the report explicitly says this is NOT evidence of
    collinearity, because that argument was tested and failed.
  - **The lowest-volatility pick loses money in both adverse scenarios**:
    `ref_dir0p3_079_c100_02` returns 0.83x under platform_algo_change and 0.90x
    in recession, against the aggressive pick's 1.60x and 1.74x. Least variable
    is not least likely to lose money — the report makes that its own callout.
  - Revenue-seeking and risk-reducing weights differ. For `dir1_110`, the
    report now shows each channel's exact spend, expected-revenue contribution,
    Euler component volatility, and corresponding shares. The component risks
    sum exactly to total portfolio standard deviation; negative components, if
    present, are diversification benefits rather than impossible negative risk.
    This is model-based attribution, not causal incrementality evidence.
  - The floored allocation makes the caveat concrete: video is funded at only
    $1,128. Attribution can quantify its simulated contribution under the model,
    but does not create empirical evidence below the historical spend floor.
  - Flags surfaced inline next to the numbers, never footnoted: 11 of 36 recs
    `ordering_unresolved`, 16 of 36 `extrapolation_floor_applied`, 0 both;
    50 of 519 frontier rows and 692 of 3,884 sweep rows floored. Flagged
    frontier points are drawn as outlined markers, not dropped.
  - Limitations cover all three required items plus scenario severity: the
    theta elasticities are ASSUMPTIONS the bronze regression does not
    corroborate (and largely cannot, for lack of spend variation), with
    incrementality/spend-variation testing named as what would settle it; both
    flags explained with counts; and Spark proven but not necessary at this
    scale (~7s single-threaded for the Phase 3 grid).

  **`verifier` (2026-08-17) — PASS on every figure, PARTIAL elsewhere; all
  findings since fixed.** It re-rendered from live gold into a scratch path and
  got a **byte-identical** file, recomputed all 12 Pareto sets by membership
  with its own brute force, and re-implemented the closed form from its
  docstring (max abs diff 0.0). **Zero numeric errors** — a first for this
  project. It also strengthened the frontier framing: a permutation test gives
  mean-VaR 7 actual vs 7.16 permuted despite corr +0.9725, while mean-std has
  LOWER correlation (0.7533) and a front of 109, so correlation does not drive
  front size and the ln(n) framing is right rather than merely defensible.

  It caught, and this has been fixed: (1) "orchestrated as a scheduled
  workflow" was an OVERCLAIM — job 94493651519110 has schedule/trigger/
  continuous all None and every run is ONE_TIME; (2) the footer disclosed the
  closed-form revenue but NOT the reconstructed spend, while claiming
  everything came from gold; (3) "no RNG" was false; (4) the sensitivity table
  is close to the normal column times three constants, so "steadiness and
  safety are different things" implied an allocation-specific fragility the
  model cannot produce — losing money is determined mainly by base-case ROAS,
  and exactly 103 of 971 fall below the platform-change break-even threshold;
  (5) the diversification mechanism is not supported
  by the correlation matrix alone, since display is ALSO the lowest-variance
  channel on its own; (6) the theta caveat over-softened ("cannot identify an
  elasticity at all") when every measured slope has t = 8.2-10.3 — the real
  limit is extrapolation, the observed range sitting below 0.07x the reference;
  (7) three of four recommendations had no channel split and were unactionable;
  (8) the budget had no time grain stated; (9) the `ordering_unresolved` chip
  never appeared beside a recommendation, since all four shown happened to be
  false — a fifth pick that carries it is now displayed; (10) two totals for
  one allocation went unreconciled (closed form vs simulated, 0.11%).

  **2026-08-18 strict-completion follow-up.** A fresh audit
  against the literal Phase 6 prompt found two scope gaps despite the earlier
  numeric verifier pass: the HTML plotted only the four primary mean-variance
  views while merely describing mean-VaR and mean-CVaR, and the channel table
  quantified revenue contribution but described risk only qualitatively. The
  source now renders all 12 scenario/objective views (mean-variance remains
  primary) and preserves both caveat types as plot outlines. The first fresh
  verifier pass independently recomputed all 12 Pareto memberships with zero
  differences and found two additional defects: the old HTML's claimed
  recession-factor spread `<0.002` is actually 0.0104639190, and regenerated
  candidate ids did not prove the spend vectors. The correction now derives
  the factor range live (0.8192-0.8297), reads exact spend vectors from Phase 5
  run 322037088744573, validates its result Parquets against live gold, removes
  analytic per-channel revenue, and renders return/risk associations that are
  labelled descriptive, compositional and non-causal. Live re-render succeeded
  at 3,884 / 519 / 36 rows with 12 SVGs and reproduced byte-for-byte on a
  second run.

  The independent verifier PASSed the engineering and every then-available
  datum: exact 971 candidate / 3,884 result keys, max absolute gold-vs-Parquet
  difference 0.0 for mean/std/VaR/CVaR/ROAS, all 12 plot counts and memberships,
  all recommendation/sensitivity figures and flags, exact `dir1_110` spend, and
  all five return/risk correlations. It found no candidate RNG, simulation,
  frontier regeneration or analytic channel-revenue call in reporting.
  It correctly left literal prompt compliance PARTIAL at that point because no
  persisted output contained per-channel revenue or component risk.

  **Resolved with explicit approval on 2026-08-18:** GitHub issue #5 records
  that schema gap. The approved targeted loader now persists 180 rows (36
  recommendations x 5 channels) in the fourth gold table after rerunning only
  24 unique cells. Before reconciliation, maximum absolute differences from
  existing gold were 2.33e-10 mean, 5.82e-11 std, 3.49e-10 VaR, 2.33e-10 CVaR,
  and 4.44e-16 ROAS. The loader independently verifies persisted row/key counts
  and that spend, spend share, expected revenue, revenue share, component
  volatility and risk share all reconcile for every recommendation. The report
  now queries this table and presents literal channel-level expected-revenue and
  risk contribution for the top recommendation; the former proxy is removed.

  **2026-08-18 repository audit follow-up.** GitHub issues #1-#3 record three
  confirmed repository-level defects found after Phase 6: the public README
  stopped at Phase 2 and miscounted the bronze tables; the report hard-coded
  the default catalog, Workflow id, $500,000 budget and 10,000 paths despite
  the documented parameterized Workflow; and no automated offline regression
  or CI suite existed. The fixes parameterize deployment identifiers through
  CLI options, validate budget/path count from live gold, compute candidate
  shares from each persisted row total, refresh the README through Phase 6,
  and add GitHub Actions plus nine credential-free pytest contracts. Local
  verification after the approved attribution extension: all Python sources
  compile, 13/13 tests pass, live gold remains 3,884 / 519 / 36 plus 180
  channel-contribution rows, the candidate artifact still matches, every
  contribution group reconciles, and two consecutive renders are byte-identical
  at SHA-256
  `BDF3B00D5E2A1FEF82DCBBA25F9C216E1610E9D9A62F7305AA4E8F655A00A94E`.

  **Two small residual defects the earlier verifier had flagged are now
  closed.** The rendered HTML now carries `<meta charset="utf-8">` and the one
  remaining raw em-dash in the source (the "Balanced" tier's caption) is
  entified, so the file opens correctly from disk under a cp1252 default rather
  than mojibaking on Windows. Live re-render is byte-identical to the disk copy
  and non-ASCII byte count is zero. The predecessor SHA
  `19A758EE91AA8D6F4B3CCE62ED03FC4E4BF4661DD601256CF236932CEB5D5223` is what
  was measured before this pair of cosmetic fixes; no gold data changed.

  Published as a Claude artifact for viewing — NOT to Databricks; no HTML is
  written to any UC volume, the workspace tree or MLflow. NOT MERGED — the standing rule is that
  the merge is the user's call, and it has been put to them rather than assumed.
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
