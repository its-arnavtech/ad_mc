"""Create the Phase 6 channel-attribution gold table.

The existing three gold tables are allocation-grain. This fourth table stores
the approved, independently reconcilable channel decomposition for the 36
reported recommendation rows without altering any existing gold result.
"""

from __future__ import annotations

import sys

from _common import get_client, print_table, resolve_warehouse_id, sql

TABLE = "ad_mc_poc.gold.recommendation_channel_contributions"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    scenario_id                     STRING NOT NULL COMMENT 'Recommendation scenario',
    objective_pair                  STRING NOT NULL COMMENT 'Frontier objective pair',
    recommendation                  STRING NOT NULL COMMENT 'max_return | min_risk | balanced',
    allocation_id                   STRING NOT NULL COMMENT 'Recommended allocation id',
    channel_id                      STRING NOT NULL COMMENT 'Channel receiving the contribution',
    phase5_workflow_run_id          BIGINT NOT NULL COMMENT 'Successful Phase 5 run whose exact candidate artifact supplied spend',
    n_paths                         INT NOT NULL COMMENT 'Original gold cell path count reused by the targeted attribution rerun',
    seed                            STRING NOT NULL COMMENT 'Original unsigned 64-bit gold cell seed',
    total_spend                     DOUBLE NOT NULL,
    channel_spend                   DOUBLE NOT NULL,
    spend_share                     DOUBLE NOT NULL,
    raw_mean_revenue_contribution   DOUBLE NOT NULL COMMENT 'Channel path mean before exact reconciliation to allocation gold',
    mean_revenue_contribution       DOUBLE NOT NULL COMMENT 'Channel expected revenue scaled so channel sum equals allocation gold mean',
    revenue_share                   DOUBLE NOT NULL COMMENT 'Channel share of rerun expected revenue; sums to 1 per recommendation',
    covariance_with_total_revenue   DOUBLE NOT NULL COMMENT 'Sample covariance of channel path revenue with total path revenue',
    raw_std_revenue_component       DOUBLE NOT NULL COMMENT 'Euler component volatility: covariance(channel,total)/rerun total std',
    std_revenue_component           DOUBLE NOT NULL COMMENT 'Euler component scaled so channel sum equals allocation gold std',
    std_risk_share                  DOUBLE NOT NULL COMMENT 'Raw Euler component divided by rerun total std; sums to 1 and may be negative',
    mean_reconciliation_factor      DOUBLE NOT NULL COMMENT 'gold mean / targeted-rerun mean',
    std_reconciliation_factor       DOUBLE NOT NULL COMMENT 'gold std / targeted-rerun std',
    attribution_method              STRING NOT NULL COMMENT 'Versioned mathematical definition of the contribution columns'
)
USING DELTA
COMMENT 'Phase 6 channel attribution for every reported recommendation. Targeted reruns reuse original seeds and paths; contributions reconcile to existing allocation-level gold.'
"""


def main() -> None:
    w = get_client()
    wid = resolve_warehouse_id(w)
    sql(w, wid, DDL)
    cols, rows = sql(w, wid, f"DESCRIBE TABLE {TABLE}")
    real = [r for r in rows if r[0] and not str(r[0]).startswith("#")]
    print_table(cols, real, max_rows=30)
    if len(real) != 21:
        raise RuntimeError(f"{TABLE} has {len(real)} columns; expected 21")
    print(f"ready: {TABLE}")


if __name__ == "__main__":
    sys.exit(main())
