import pandas as pd
import pytest
from pathlib import Path

from reporting.build_report import parse_args, report_run_shape


def test_report_cli_defaults_and_overrides():
    defaults = parse_args([])
    assert defaults.catalog == "ad_mc_poc"
    assert defaults.phase5_job_id == 94493651519110

    output = Path("custom-report.html")
    configured = parse_args([
        "--catalog", "another_catalog",
        "--phase5-job-id", "123",
        "--output", str(output),
    ])
    assert configured.catalog == "another_catalog"
    assert configured.phase5_job_id == 123
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
