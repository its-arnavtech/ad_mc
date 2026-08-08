# ad_mc

Ad revenue optimization with Monte Carlo simulation.

Advertising channels are treated like assets in a financial portfolio: channel
ROAS is the "return", cross-channel correlation is the "risk", and the eventual
goal is an efficient frontier of budget allocations. Built on Databricks
(Delta Lake, Workflows, MLflow).

## Repo layout

```
data_generation/   synthetic history generator (no Spark -- plain CSV out)
data/raw/          generated CSV lands here (gitignored; reproducible by seed)
data/validation/   validation artifacts, e.g. correlation heatmap (gitignored)
databricks/        manually-run scripts for the Databricks side
```

## Phase 1 -- Data foundation (bronze layer)

Produces three Delta tables in `ad_mc_poc.bronze`:

| table | grain | contents |
|---|---|---|
| `channel_performance_history` | date x channel | date, channel_id, impressions, clicks, conversions, spend, revenue |
| `channel_assumptions` | channel | mean/std of CTR, CVR, revenue-per-conversion + distribution_type |
| `channel_correlation_matrix` | channel pair | Pearson correlation of daily revenue |

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

## Later phases (not started)

Simulation engine, Workflows orchestration, and MLflow instrumentation are out
of scope until the bronze layer is signed off.
