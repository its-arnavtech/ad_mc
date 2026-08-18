"""Build and persist channel attribution for every gold recommendation.

This is the explicitly approved exception to Phase 6's original no-recompute
rule. It reruns only the unique cells referenced by frontier_recommendations,
using their original seed, path count, scenario, exact Phase 5 candidate spend,
and current verified model inputs. Nothing in the existing three gold tables is
rewritten.

Risk attribution uses Euler component standard deviation:

    component_i = Cov(channel_revenue_i, total_revenue) / Std(total_revenue)

The raw components sum to the rerun total standard deviation. Both mean and
standard-deviation components are then scaled by the tiny rerun-to-gold ratio
so their persisted sums reconcile exactly to the authoritative gold aggregate.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "databricks"))
sys.path.insert(0, str(REPO / "reporting"))
sys.path.insert(0, str(REPO / "simulation"))

from _common import get_client, resolve_warehouse_id, sql  # noqa: E402
import engine  # noqa: E402
import frontier as F  # noqa: E402
import load_phase4_gold as P4  # noqa: E402
import build_report as R  # noqa: E402
from phase5_artifacts import load_persisted_candidates  # noqa: E402
from scenarios import get_scenario  # noqa: E402

DEFAULT_CATALOG = "ad_mc_poc"
DEFAULT_JOB_ID = 94493651519110
METHOD = "euler_std_v1: cov(channel,total)/std(total); reconciled_to_gold"

COLUMNS = [
    "scenario_id", "objective_pair", "recommendation", "allocation_id", "channel_id",
    "phase5_workflow_run_id", "n_paths", "seed", "total_spend", "channel_spend",
    "spend_share", "raw_mean_revenue_contribution", "mean_revenue_contribution",
    "revenue_share", "covariance_with_total_revenue", "raw_std_revenue_component",
    "std_revenue_component", "std_risk_share", "mean_reconciliation_factor",
    "std_reconciliation_factor", "attribution_method",
]


def _lit(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"refusing to persist non-finite value {value!r}")
        return repr(float(value))
    return "'" + str(value).replace("'", "''") + "'"


def _insert_overwrite(w, wid, table: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to overwrite the contribution table with zero rows")
    values = ",\n".join(
        "(" + ", ".join(_lit(row[c]) for c in COLUMNS) + ")" for row in rows
    )
    sql(w, wid, f"INSERT OVERWRITE {table} ({', '.join(COLUMNS)}) VALUES\n{values}")


def _simulate_cell(allocation: dict[str, float], scenario_id: str, inputs: F.SweepInputs,
                   n_paths: int, seed: int) -> engine.SimulationResult:
    scenario = get_scenario(scenario_id)
    return engine.simulate(
        channels=list(inputs.channels), allocation=allocation,
        mean_cvr=inputs.mean_cvr * scenario.cvr_multiplier,
        std_cvr=inputs.std_cvr * scenario.cvr_multiplier,
        mean_cpc=inputs.mean_cpc * scenario.cpc_multiplier,
        std_cpc=inputs.std_cpc * scenario.cpc_multiplier,
        mean_rpc=inputs.mean_rpc * scenario.revenue_multiplier,
        std_rpc=inputs.std_rpc * scenario.revenue_multiplier,
        correlation=inputs.correlation, n_paths=n_paths, seed=seed,
        theta=inputs.theta, reference_spend=inputs.reference_spend,
        saturate_std_cpc=inputs.saturate_std_cpc, spend_floor=inputs.spend_floor,
    )


def euler_std_components(revenue_by_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Cov(channel,total) and Euler volatility components.

    With sample covariance and sample standard deviation, the returned
    components sum to the standard deviation of row-wise total revenue (within
    floating-point precision). Negative values are valid diversification
    contributions.
    """
    values = np.asarray(revenue_by_channel, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("revenue_by_channel must be a 2-D array with >=2 paths")
    if not np.isfinite(values).all():
        raise ValueError("revenue_by_channel contains a non-finite value")
    total = values.sum(axis=1)
    total_std = float(total.std(ddof=1))
    if not np.isfinite(total_std) or total_std <= 0.0:
        raise ValueError("total simulated revenue must have positive volatility")
    centered_channels = values - values.mean(axis=0)
    centered_total = total - total.mean()
    covariances = np.sum(centered_channels * centered_total[:, None], axis=0) / (len(total) - 1)
    return covariances, covariances / total_std


def build_rows(sweep: pd.DataFrame, recs: pd.DataFrame, candidates: pd.DataFrame,
               workflow_run_id: int, inputs: F.SweepInputs) -> tuple[list[dict], dict]:
    spend_cols = [f"spend_{channel}" for channel in inputs.channels]
    candidate_lookup = candidates.set_index("allocation_id")
    gold = sweep.set_index(["allocation_id", "scenario_id"])
    cache: dict[tuple[str, str], tuple[engine.SimulationResult, dict, float, float]] = {}
    max_abs = {key: 0.0 for key in ("mean_revenue", "std_revenue", "var_95", "cvar_95", "expected_roas")}
    rows: list[dict] = []

    for rec in recs.itertuples(index=False):
        key = (str(rec.allocation_id), str(rec.scenario_id))
        gold_row = gold.loc[key]
        if key not in cache:
            candidate = candidate_lookup.loc[key[0]]
            allocation = {
                channel: float(candidate[f"spend_{channel}"])
                for channel in inputs.channels
            }
            n_paths = int(gold_row.n_paths)
            seed = int(str(gold_row.seed))
            result = _simulate_cell(allocation, key[1], inputs, n_paths, seed)
            summary = F.summarize_revenue(
                key[0], key[1], result.total_revenue, float(gold_row.total_spend), seed)
            for metric in max_abs:
                diff = abs(float(summary[metric]) - float(gold_row[metric]))
                max_abs[metric] = max(max_abs[metric], diff)
                if not np.isclose(summary[metric], float(gold_row[metric]), rtol=1e-11, atol=1e-6):
                    raise RuntimeError(
                        f"targeted rerun does not reconcile for {key} {metric}: "
                        f"rerun={summary[metric]!r}, gold={gold_row[metric]!r}")
            raw_mean_total = float(result.revenue_by_channel.mean(axis=0).sum())
            raw_std_total = float(result.total_revenue.std(ddof=1))
            mean_factor = float(gold_row.mean_revenue) / raw_mean_total
            std_factor = float(gold_row.std_revenue) / raw_std_total
            cache[key] = (result, allocation, mean_factor, std_factor)

        result, allocation, mean_factor, std_factor = cache[key]
        total = result.total_revenue
        total_std = float(total.std(ddof=1))
        channel_means = result.revenue_by_channel.mean(axis=0)
        covariances, raw_components = euler_std_components(result.revenue_by_channel)
        total_spend = float(sum(allocation.values()))

        for i, channel in enumerate(inputs.channels):
            rows.append({
                "scenario_id": key[1],
                "objective_pair": str(rec.objective_pair),
                "recommendation": str(rec.recommendation),
                "allocation_id": key[0],
                "channel_id": channel,
                "phase5_workflow_run_id": workflow_run_id,
                "n_paths": len(total),
                "seed": str(gold_row.seed),
                "total_spend": total_spend,
                "channel_spend": allocation[channel],
                "spend_share": allocation[channel] / total_spend,
                "raw_mean_revenue_contribution": float(channel_means[i]),
                "mean_revenue_contribution": float(channel_means[i] * mean_factor),
                "revenue_share": float(channel_means[i] / channel_means.sum()),
                "covariance_with_total_revenue": float(covariances[i]),
                "raw_std_revenue_component": float(raw_components[i]),
                "std_revenue_component": float(raw_components[i] * std_factor),
                "std_risk_share": float(raw_components[i] / total_std),
                "mean_reconciliation_factor": mean_factor,
                "std_reconciliation_factor": std_factor,
                "attribution_method": METHOD,
            })

    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["scenario_id", "objective_pair", "recommendation"], sort=False)
    if not (grouped.size() == len(inputs.channels)).all():
        raise AssertionError("every recommendation must have exactly one row per channel")
    for keys, group in grouped:
        rec = recs.set_index(["scenario_id", "objective_pair", "recommendation"]).loc[keys]
        if not np.isclose(group.mean_revenue_contribution.sum(), rec.mean_revenue,
                          rtol=1e-12, atol=1e-6):
            raise AssertionError(f"mean contribution sum does not reconcile for {keys}")
        if not np.isclose(group.std_revenue_component.sum(), rec.std_revenue,
                          rtol=1e-12, atol=1e-6):
            raise AssertionError(f"risk contribution sum does not reconcile for {keys}")
    return rows, {"unique_cells": len(cache), "max_abs_diff": max_abs}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--phase5-job-id", type=int, default=DEFAULT_JOB_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    gold = f"{args.catalog}.gold"
    table = f"{gold}.recommendation_channel_contributions"
    w = get_client()
    wid = resolve_warehouse_id(w)
    sweep, _front, recs, _existing_contributions = R.load(w, wid, gold)
    candidates, run_id = load_persisted_candidates(
        w, sweep, catalog=args.catalog, phase5_job_id=args.phase5_job_id)
    inputs, _channels = P4.build_inputs(w, wid)
    rows, diagnostics = build_rows(sweep, recs, candidates, run_id, inputs)
    print(f"built {len(rows)} rows from {diagnostics['unique_cells']} unique cells")
    print(f"max aggregate abs diff before reconciliation: {diagnostics['max_abs_diff']}")
    if args.dry_run:
        print("dry run: live table not modified")
        return

    _insert_overwrite(w, wid, table, rows)
    _cols, checks = sql(w, wid, f"""
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT CONCAT_WS('|', scenario_id, objective_pair, recommendation)) AS recs,
               COUNT(DISTINCT channel_id) AS channels,
               SUM(CASE WHEN n_paths <= 0 OR total_spend <= 0 THEN 1 ELSE 0 END) AS invalid
        FROM {table}
    """)
    got = checks[0]
    if (int(got[0]), int(got[1]), int(got[2]), int(got[3])) != (len(rows), len(recs), len(inputs.channels), 0):
        raise RuntimeError(f"live contribution verification failed: {got}")
    _cols, reconciliation = sql(w, wid, f"""
        WITH contribution_sums AS (
            SELECT scenario_id, objective_pair, recommendation, allocation_id,
                   SUM(channel_spend) AS spend_sum,
                   MAX(total_spend) AS total_spend,
                   SUM(spend_share) AS spend_share_sum,
                   SUM(mean_revenue_contribution) AS mean_sum,
                   SUM(revenue_share) AS revenue_share_sum,
                   SUM(std_revenue_component) AS std_sum,
                   SUM(std_risk_share) AS risk_share_sum
            FROM {table}
            GROUP BY scenario_id, objective_pair, recommendation, allocation_id
        )
        SELECT MAX(ABS(c.spend_sum - c.total_spend)) AS spend_diff,
               MAX(ABS(c.spend_share_sum - 1.0)) AS spend_share_diff,
               MAX(ABS(c.mean_sum - r.mean_revenue)) AS mean_diff,
               MAX(ABS(c.revenue_share_sum - 1.0)) AS revenue_share_diff,
               MAX(ABS(c.std_sum - r.std_revenue)) AS std_diff,
               MAX(ABS(c.risk_share_sum - 1.0)) AS risk_share_diff,
               COUNT(*) AS matched_recommendations
        FROM contribution_sums c
        JOIN {gold}.frontier_recommendations r
          ON c.scenario_id = r.scenario_id
         AND c.objective_pair = r.objective_pair
         AND c.recommendation = r.recommendation
         AND c.allocation_id = r.allocation_id
    """)
    recon = reconciliation[0]
    diffs = [float(v) for v in recon[:6]]
    if int(recon[6]) != len(recs) or any(diff > 1e-6 for diff in diffs):
        raise RuntimeError(f"live contribution reconciliation failed: {recon}")
    print(
        f"live verification PASS: {got[0]} rows / {got[1]} recs / {got[2]} channels; "
        f"max reconciliation diffs {diffs}")


if __name__ == "__main__":
    main()
