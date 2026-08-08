"""
generate_channel_data.py

Standalone synthetic data generator for the Ad Revenue Monte Carlo POC
(Step 4 of the "where to start" plan). No Spark here on purpose -- this
produces a plain CSV so you can eyeball it and sanity-check it before it
ever touches Delta Lake. Loading it into the bronze `channel_performance_history`
table is Step 5.

Usage:
    python data_generation/generate_channel_data.py
    python data_generation/generate_channel_data.py --days 730 --seed 42
    python data_generation/generate_channel_data.py --start-date 2024-01-01 --days 365

By default the CSV lands in the repo's `data/raw/` directory, which is where
the bronze loader in `databricks/` expects to find it.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# repo_root/data_generation/generate_channel_data.py -> repo_root/data/raw/...
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "channel_performance_history.csv"

# ---------------------------------------------------------------------------
# Channel definitions -- tune these to change the "shape" of each channel.
#   base_impressions   : average daily impressions
#   base_ctr / base_cvr: baseline click-through / conversion rates
#   base_cpc            : average cost per click ($)
#   base_rev_per_conv   : average revenue per conversion ($)
#   macro_sensitivity    : how strongly this channel reacts to the shared
#                           demand factor below (this is what creates
#                           realistic cross-channel correlation)
# ---------------------------------------------------------------------------
CHANNELS = {
    "paid_search":  dict(base_impressions=40_000,  base_ctr=0.045, base_cvr=0.080, base_cpc=2.10, base_rev_per_conv=85, macro_sensitivity=1.0),
    "paid_social":  dict(base_impressions=120_000, base_ctr=0.012, base_cvr=0.030, base_cpc=0.85, base_rev_per_conv=55, macro_sensitivity=1.3),
    "display":      dict(base_impressions=300_000, base_ctr=0.004, base_cvr=0.015, base_cpc=0.35, base_rev_per_conv=40, macro_sensitivity=0.6),
    "video":        dict(base_impressions=90_000,  base_ctr=0.018, base_cvr=0.025, base_cpc=1.10, base_rev_per_conv=48, macro_sensitivity=0.9),
    "programmatic": dict(base_impressions=200_000, base_ctr=0.007, base_cvr=0.020, base_cpc=0.55, base_rev_per_conv=42, macro_sensitivity=1.1),
}

# Mon=0 .. Sun=6 -- B2C-ish weekly pattern, weaker on weekends
WEEKDAY_MULTIPLIER = {0: 1.05, 1: 1.08, 2: 1.10, 3: 1.07, 4: 0.95, 5: 0.75, 6: 0.80}


def simulate_macro_factor(n_days, rng, persistence=0.9, vol=0.05):
    """
    A slow-moving shared 'market demand' factor (AR(1) process) that nudges
    CTR/CVR up or down across ALL channels together. This is what creates
    genuine positive correlation between channels in the historical data --
    without it, every channel would be independent and the correlation
    matrix you compute in Step 6 would be ~0 everywhere, which isn't
    realistic and would make the Cholesky-based correlated simulation
    (Section 5.2 of the spec) pointless.
    """
    factor = np.zeros(n_days)
    for t in range(1, n_days):
        factor[t] = persistence * factor[t - 1] + rng.normal(0, vol)
    return 1.0 + np.tanh(factor)  # squashed multiplier centered at 1.0


def generate_channel_history(start_date, n_days, seed=42):
    rng = np.random.default_rng(seed)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    macro = simulate_macro_factor(n_days, rng)

    rows = []
    for channel, p in CHANNELS.items():
        idio_noise = rng.normal(1.0, 0.06, size=n_days)  # per-channel, independent of others

        for i, date in enumerate(dates):
            wd_mult = WEEKDAY_MULTIPLIER[date.weekday()]
            macro_mult = 1.0 + p["macro_sensitivity"] * (macro[i] - 1.0)

            impressions = max(0, int(
                p["base_impressions"] * wd_mult * (0.9 + 0.2 * idio_noise[i])
                * (0.95 + 0.1 * macro_mult) * rng.normal(1.0, 0.03)
            ))

            ctr = np.clip(p["base_ctr"] * macro_mult * rng.normal(1.0, 0.10), 0.0005, 0.5)
            cvr = np.clip(p["base_cvr"] * macro_mult * rng.normal(1.0, 0.12), 0.001, 0.6)

            clicks = int(round(impressions * ctr))
            conversions = int(round(clicks * cvr))

            cpc = max(0.05, p["base_cpc"] * rng.normal(1.0, 0.08))
            spend = round(clicks * cpc, 2)

            # revenue per conversion is right-skewed -- a lognormal multiplier
            # around the base value, matching the marginal distribution the
            # spec calls for in Section 5.1
            rev_per_conv = p["base_rev_per_conv"] * rng.lognormal(mean=0, sigma=0.25)
            revenue = round(conversions * rev_per_conv, 2)

            rows.append(dict(
                date=date.strftime("%Y-%m-%d"),
                channel_id=channel,
                impressions=impressions,
                clicks=clicks,
                conversions=conversions,
                spend=spend,
                revenue=revenue,
            ))

    return pd.DataFrame(rows)


def print_summary(df):
    """Quick eyeball check -- this is the Step 4/7 'does this look like real
    ad data' pass before anything touches Delta Lake."""
    print("\n=== Per-channel summary ===")
    g = df.groupby("channel_id")
    summary = pd.DataFrame({
        "avg_daily_impressions": g["impressions"].mean(),
        "avg_ctr": g.apply(lambda d: (d["clicks"] / d["impressions"]).mean()),
        "avg_cvr": g.apply(lambda d: (d["conversions"] / d["clicks"]).mean()),
        "total_spend": g["spend"].sum(),
        "total_revenue": g["revenue"].sum(),
    })
    summary["roas"] = summary["total_revenue"] / summary["total_spend"]
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary)

    print("\n=== Cross-channel revenue correlation (preview of Step 6's correlation matrix) ===")
    pivot = df.pivot(index="date", columns="channel_id", values="revenue")
    print(pivot.corr().round(2))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic channel_performance_history data.")
    parser.add_argument("--days", type=int, default=730, help="Days of history to generate (default: 730 = ~2 years)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for reproducibility")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output CSV path (default: data/raw/channel_performance_history.csv)")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD (default: --days before today)")
    args = parser.parse_args()

    start = (
        datetime.strptime(args.start_date, "%Y-%m-%d")
        if args.start_date
        else datetime.today() - timedelta(days=args.days)
    )

    df = generate_channel_history(start, args.days, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df):,} rows ({args.days} days x {len(CHANNELS)} channels) to {args.out}")
    print_summary(df)


if __name__ == "__main__":
    main()