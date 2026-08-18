import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from reporting.build_report import contribution_breakdown, parse_args, report_run_shape
from simulation.load_channel_contributions import euler_std_components


def test_report_cli_defaults_and_overrides():
    defaults = parse_args([])
    assert defaults.catalog == "ad_mc_poc"

    output = Path("custom-report.html")
    configured = parse_args([
        "--catalog", "another_catalog",
        "--output", str(output),
    ])
    assert configured.catalog == "another_catalog"
    assert configured.output == output


def test_report_run_shape_comes_from_gold_rows():
    sweep = pd.DataFrame({
        "total_spend": [750_000.0, 750_000.0],
        "n_paths": [25_000, 25_000],
    })
    assert report_run_shape(sweep) == (750_000.0, 25_000)


@pytest.mark.parametrize(
    "column,values",
    [
        ("total_spend", [500_000.0, 750_000.0]),
        ("n_paths", [10_000, 20_000]),
    ],
)
def test_report_rejects_mixed_run_configurations(column, values):
    sweep = pd.DataFrame({
        "total_spend": [500_000.0, 500_000.0],
        "n_paths": [10_000, 10_000],
    })
    sweep[column] = values
    with pytest.raises(RuntimeError):
        report_run_shape(sweep)


def test_contribution_breakdown_selects_exact_recommendation():
    contributions = pd.DataFrame({
        "scenario_id": ["normal", "normal", "recession"],
        "objective_pair": ["mean_revenue vs std_revenue"] * 3,
        "recommendation": ["max_return"] * 3,
        "allocation_id": ["a", "a", "a"],
        "channel_id": ["search", "video", "search"],
        "channel_spend": [60.0, 40.0, 60.0],
        "spend_share": [0.6, 0.4, 0.6],
        "mean_revenue_contribution": [120.0, 80.0, 100.0],
        "revenue_share": [0.6, 0.4, 1.0],
        "std_revenue_component": [12.0, 8.0, 10.0],
        "std_risk_share": [0.6, 0.4, 1.0],
    })
    recommendation = pd.Series({
        "scenario_id": "normal",
        "objective_pair": "mean_revenue vs std_revenue",
        "recommendation": "max_return",
        "allocation_id": "a",
    })
    result = contribution_breakdown(contributions, recommendation)
    assert result.channel.tolist() == ["search", "video"]
    assert result.revenue.sum() == 200.0
    assert result.risk_component.sum() == 20.0


def test_euler_components_reconcile_to_portfolio_volatility():
    channel_paths = np.array([
        [10.0, 4.0, 7.0],
        [12.0, 3.0, 8.0],
        [9.0, 7.0, 6.0],
        [14.0, 2.0, 9.0],
        [11.0, 5.0, 5.0],
    ])
    covariance, components = euler_std_components(channel_paths)
    assert covariance.shape == (3,)
    expected_std = channel_paths.sum(axis=1).std(ddof=1)
    assert components.sum() == pytest.approx(expected_std, abs=1e-12)


def test_euler_components_reject_degenerate_total():
    with pytest.raises(ValueError, match="positive volatility"):
        euler_std_components(np.ones((4, 2)))


def test_committed_report_uses_persisted_contributions_not_proxy_links():
    report = (Path(__file__).parents[1] / "reporting" /
              "budget_allocation_report.html").read_text(encoding="utf-8")
    assert report.count("<svg ") == 12
    assert "Component volatility" in report
    assert "recommendation_channel_contributions" in report
    assert "Return link" not in report
    assert "Risk link" not in report
