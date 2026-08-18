# ad_mc

Ad revenue optimization with Monte Carlo simulation.

Advertising channels are treated like assets in a financial portfolio: channel
ROAS is the "return", cross-channel correlation is the "risk", and the eventual
goal is an efficient frontier of budget allocations. Built on Databricks
(Delta Lake, Workflows, MLflow).

The implementation is complete through Phase 6 on phase-specific branches.
Those branches have intentionally not been merged into `main`; review
`CLAUDE.md` for the verified run history, exact caveats, and current branch
status.

## Repo layout

```
data_generation/   synthetic history generator (no Spark -- plain CSV out)
data/raw/          generated CSV lands here (gitignored; reproducible by seed)
data/validation/   validation artifacts, e.g. correlation heatmap (gitignored)
databricks/        manually-run scripts for the Databricks side
simulation/        Monte Carlo engine, distributed sweep, frontier, Workflow code
reporting/         live-gold Phase 6 report builder and generated HTML report
tests/             offline deterministic regression tests
.github/workflows/ automated compile and offline pytest checks
```

## Phase 1 -- Data foundation (bronze layer)

Produces four Delta tables in `ad_mc_poc.bronze`:

| table | grain | contents |
|---|---|---|
| `channel_performance_history` | date x channel | date, channel_id, impressions, clicks, conversions, spend, revenue |
| `channel_assumptions` | channel | mean/std of CTR, CPC, CVR, revenue-per-conversion + distribution_type |
| `channel_correlation_matrix` | channel pair | Pearson correlation of daily **revenue** -- diagnostic only |
| `channel_cvr_correlation_matrix` | channel pair | Pearson correlation of daily **CVR** -- drives the simulation |

### Two correlation matrices, and why

Revenue = clicks x CVR x revenue-per-conversion, so a revenue correlation
bundles CTR, CVR and revenue-per-conversion co-movement together. The
simulation's Cholesky step correlates **CVR only**, so it needs correlation
measured on CVR. Both are kept: the CVR matrix feeds the engine, the revenue
matrix stays as a business-level diagnostic. Switch between them with
`CORRELATION_SOURCE` in `simulation/config.py`.

Measured on this dataset the revenue matrix **understates** CVR correlation
(mean off-diagonal 0.371 vs 0.406, higher on 18 of 20 pairs), because the
large independent per-channel revenue-per-conversion noise dilutes the shared
macro signal that CVR carries cleanly.

Days with zero clicks leave CVR undefined; those channel-days are excluded
pairwise rather than imputed. On the current 730-day dataset there are **none**,
so every pair uses all 730 observations.

### 1. Generate the data locally

```bash
pip install -r requirements.txt
python data_generation/generate_channel_data.py --start-date 2024-01-01 --days 730 --seed 42
```

Writes `data/raw/channel_performance_history.csv` (3,650 rows = 730 days x 5
channels). The seed and explicit start date make this byte-for-byte
reproducible; without `--start-date` the window floats relative to today.

### 2. Configure Databricks credentials

Credentials are read from the Databricks unified auth chain. **Nothing in this
repo ever contains a token.** Use either the CLI credential store
(`~/.databrickscfg`) or environment variables:

```
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_TOKEN=<personal-access-token>
DATABRICKS_WAREHOUSE_ID=<optional; auto-detected if omitted>
```

### 3. Run the bronze pipeline

Each script is standalone, idempotent, and manually run -- nothing is scheduled.

```bash
python databricks/00_check_auth.py         # verify auth; reports UC vs Hive metastore
python databricks/01_setup_catalog.py      # catalog + bronze/silver/gold schemas + landing volume
python databricks/02_load_bronze.py        # CSV -> ad_mc_poc.bronze.channel_performance_history
python databricks/03_build_assumptions.py  # derive the two assumption tables
python databricks/04_validate_bronze.py    # row counts, nulls, ranges, correlation heatmap
```

`04_validate_bronze.py` exits non-zero if any check fails, so it can gate later
phases.

### How SQL reaches the workspace

The scripts use the Databricks SDK's **Statement Execution API** against a SQL
warehouse. That avoids `databricks-connect` and any pin to a specific cluster or
DBR version. If the workspace has no SQL warehouse, these steps need porting to
a notebook on an all-purpose cluster instead.

### Assumptions baked into Phase 1

Documented at the top of `databricks/03_build_assumptions.py`, briefly:

- Correlation uses the **full 730-day history**, not a rolling window.
- Correlation is **Pearson on daily revenue**, pairing channels by date; the
  full symmetric matrix (including the 1.0 diagonal) is stored, which is the
  form a Cholesky decomposition wants later.
- `channel_name` is title-cased from `channel_id` -- the CSV has no name column.
- `distribution_type` describes the **revenue-per-conversion** marginal and is
  chosen from observed skewness (> 0.5 -> `lognormal`), not hardcoded.
- Daily ratios are averaged across days (mean-of-ratios), so each day counts
  equally; `std_*` are sample standard deviations.

## Phase 2 -- Single-node simulation engine

Proves the correlated Monte Carlo logic by hand, on one node, before Phase 3
distributes it. No Spark, no Workflows, no MLflow. Output is local only
(printed stats + a saved histogram) -- persisting results starts in Phase 3.

```bash
python simulation/run_phase2_simulation.py
```

```
simulation/config.py                  budget, allocation strategy, paths, seed, scenario
simulation/engine.py                  pure math -- no I/O, deterministic given a seed
simulation/data_access.py             reads the bronze assumption tables
simulation/run_phase2_simulation.py   runs one simulation and validates it
```

### The CPC gap

Phase 1's `channel_assumptions` carried CTR, which maps impressions -> clicks
-- the historical generator's direction. Simulating *forward* from a dollar
allocation needs the opposite: cost per click. `mean_cpc` / `std_cpc` were
added as **columns on `channel_assumptions`** rather than a companion table,
because they are the same grain (one row per channel) and the same kind of
thing (a distribution parameter the simulation draws from); a separate table
would force a join for no benefit. Derived from daily `spend / clicks` with
the same mean-of-ratios / sample-stddev treatment as the existing columns.
CTR stays in the table as a historical diagnostic and is unused in Phase 2.

### Model

Per path, per channel: `clicks = spend / CPC`, `conversions = clicks * CVR`,
`revenue = conversions * revenue_per_conversion`.

| quantity | marginal | fitted by | correlated? |
|---|---|---|---|
| CVR | Beta | method of moments | **yes** -- Gaussian copula via Cholesky |
| CPC | LogNormal | method of moments | no |
| revenue per conversion | LogNormal | method of moments | no |

### Simplifications, stated explicitly

- **Only CVR is correlated.** CPC and revenue-per-conversion are drawn
  independently across channels. Correlating them is a reasonable extension,
  deliberately not built.
- **A Gaussian copula transmits rank correlation**, so the Pearson correlation
  of the resulting Beta draws is slightly attenuated relative to the input.
  Small and expected -- not a Cholesky failure.
- **`distribution_type` still describes only revenue-per-conversion.** The row
  now carries three distributions but one type column. Phase 2's families come
  from the spec, not that column, so it stays descriptive-only.

## Phase 3 -- Distributed simulation

Phase 3 wraps the validated engine in a Spark `applyInPandas` batch over 21
allocation probes, four scenarios, and the configured path count. In this
workspace Python UDFs run on transient serverless job compute; the retained
classic-cluster helper is not usable here.

```bash
python databricks/06_create_phase3_tables.py
python databricks/07_load_phase3_reference.py
python simulation/run_phase3_distributed.py
```

The run populates `silver.simulated_allocation_outcomes`; it does not create a
persisted Workflow or MLflow run. Those belong to Phase 5.

## Phase 4 -- Saturation, optimization, and gold

Phase 4 adds the documented spend-saturation curve and evidence floor, then
runs a two-stage candidate sweep. It writes three allocation-grain gold tables:

- `gold.allocation_sweep_results`
- `gold.efficient_frontier`
- `gold.frontier_recommendations`

```bash
python databricks/08_create_gold_frontier.py
python simulation/load_phase4_gold.py
```

The theta values controlling diminishing returns are elicited assumptions, not
measurements. The bronze history does not cover enough spend variation to
validate their published channel ordering; this limitation is carried into the
Phase 6 report rather than hidden.

## Phase 5 -- Databricks Workflow and MLflow

Phase 5 deploys the already-validated pipeline as a five-task, parameterized
Databricks Workflow and records one MLflow run per Workflow run.

```bash
python simulation/run_phase5_workflow.py --deploy-only
python simulation/run_phase5_workflow.py --run-only
```

The Workflow is persisted but deliberately has no schedule or continuous
trigger. A plain invocation deploys and runs it; use the explicit modes above
when only one action is intended.

## Phase 6 -- Analysis and report

Phase 6 reads the verified gold results and the exact candidate Parquets from
the matching successful Phase 5 run. It does not regenerate candidates or run
Monte Carlo paths.

```bash
python reporting/build_report.py
python reporting/build_report.py --help
```

The generated self-contained report is
`reporting/budget_allocation_report.html`. Catalog, Phase 5 job id, output path,
budget, and path-count handling are explicit: deployment identifiers are CLI
options, while budget and path count are validated from live gold.

Gold is allocation-grain and does not contain per-channel revenue or component
risk. The report therefore presents exact channel spend plus descriptive links
between channel share and gold-level return/risk, and says plainly that this is
not causal contribution attribution.

## Offline checks

The deterministic contracts can be checked without Databricks credentials:

```bash
pip install -r requirements-dev.txt
python -m compileall -q data_generation databricks simulation reporting tests
python -m pytest -q
```

GitHub Actions runs the same compile and pytest checks on pushes and pull
requests. Live-table verification remains a separate read-only Databricks step
because CI intentionally has no workspace credentials.
