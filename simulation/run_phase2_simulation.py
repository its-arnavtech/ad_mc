"""
Phase 2 -- run one 10,000-path correlated Monte Carlo simulation for a fixed
allocation and the "normal" scenario, then validate it.

Output is local only: printed stats plus a saved histogram. Nothing is written
back to Delta; persisting simulation output starts in Phase 3.

    python simulation/run_phase2_simulation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import (
    ALLOCATION_STRATEGY,
    N_PATHS,
    RANDOM_SEED,
    SCENARIO,
    TOTAL_BUDGET,
    build_allocation,
)
from data_access import load_assumptions, load_correlation_matrix, open_session
from engine import (
    jensen_corrected_expectation,
    naive_point_estimate,
    risk_metrics,
    simulate,
    validate_correlation_recovery,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTOGRAM_PATH = REPO_ROOT / "data" / "validation" / "phase2_revenue_distribution.png"

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}{(' -- ' + detail) if detail else ''}")


def money(x: float) -> str:
    return f"${x:,.0f}"


def main() -> None:
    print("=" * 72)
    print("PHASE 2 -- SINGLE-NODE CORRELATED MONTE CARLO")
    print("=" * 72)

    # ---- inputs -------------------------------------------------------------
    w, warehouse_id = open_session()
    assumptions = load_assumptions(w, warehouse_id)
    channels = assumptions["channel_id"].tolist()
    correlation = load_correlation_matrix(w, warehouse_id, channels)
    allocation = build_allocation(channels, ALLOCATION_STRATEGY, TOTAL_BUDGET)

    print(f"\nscenario          : {SCENARIO} (no macro multipliers applied)")
    print(f"paths             : {N_PATHS:,}")
    print(f"seed              : {RANDOM_SEED}")
    print(f"total budget      : {money(TOTAL_BUDGET)}")
    print(f"allocation        : {ALLOCATION_STRATEGY} -> "
          f"{money(TOTAL_BUDGET / len(channels))} x {len(channels)} channels")
    print(f"channels          : {channels}")
    print(f"correlation source: bronze.channel_correlation_matrix "
          f"(daily-revenue proxy, {correlation.shape[0]}x{correlation.shape[1]})")

    print("\n-- assumptions read from bronze --")
    show = assumptions[["channel_id", "mean_cpc", "std_cpc", "mean_cvr", "std_cvr",
                        "mean_revenue_per_conversion", "std_revenue_per_conversion"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    # ---- run ----------------------------------------------------------------
    result = simulate(
        channels=channels,
        allocation=allocation,
        mean_cvr=assumptions["mean_cvr"].to_numpy(),
        std_cvr=assumptions["std_cvr"].to_numpy(),
        mean_cpc=assumptions["mean_cpc"].to_numpy(),
        std_cpc=assumptions["std_cpc"].to_numpy(),
        mean_rpc=assumptions["mean_revenue_per_conversion"].to_numpy(),
        std_rpc=assumptions["std_revenue_per_conversion"].to_numpy(),
        correlation=correlation,
        n_paths=N_PATHS,
        seed=RANDOM_SEED,
    )

    print("\n-- fitted distribution parameters --")
    print(f"  {'channel':<14} {'Beta(a,b) for CVR':<28} {'LogN(mu,s) CPC':<24} LogN(mu,s) RPC")
    for c in channels:
        a, b = result.beta_params[c]
        cm, cs = result.lognormal_cpc_params[c]
        rm, rs = result.lognormal_rpc_params[c]
        print(f"  {c:<14} ({a:9.4f}, {b:11.4f})      ({cm:7.4f}, {cs:6.4f})       ({rm:6.4f}, {rs:6.4f})")

    # ---- 0. moment recovery -------------------------------------------------
    print("\n== 0. Method-of-moments recovery (does the fit reproduce the inputs?) ==")
    for label, drawn, mean_col, std_col in (
        ("CVR", result.cvr, "mean_cvr", "std_cvr"),
        ("CPC", result.cpc, "mean_cpc", "std_cpc"),
        ("RPC", result.rpc, "mean_revenue_per_conversion", "std_revenue_per_conversion"),
    ):
        target_mean = assumptions[mean_col].to_numpy()
        target_std = assumptions[std_col].to_numpy()
        rel_mean = np.abs(drawn.mean(axis=0) - target_mean) / target_mean
        rel_std = np.abs(drawn.std(axis=0, ddof=1) - target_std) / target_std
        print(f"  {label}: max relative error  mean={rel_mean.max():.4%}  std={rel_std.max():.4%}")
        check(f"{label} sample moments match assumptions within 5%",
              rel_mean.max() < 0.05 and rel_std.max() < 0.05,
              f"mean {rel_mean.max():.2%}, std {rel_std.max():.2%}")

    # ---- 1. plausibility ----------------------------------------------------
    print("\n== 1. Plausibility vs naive point estimate ==")
    naive_by_channel = naive_point_estimate(
        channels, allocation,
        assumptions["mean_cvr"].to_numpy(),
        assumptions["mean_cpc"].to_numpy(),
        assumptions["mean_revenue_per_conversion"].to_numpy(),
    )
    naive_total = naive_by_channel.sum()

    cpc_sigmas = np.array([result.lognormal_cpc_params[c][1] for c in channels])
    analytic_by_channel = jensen_corrected_expectation(naive_by_channel, cpc_sigmas)
    analytic_total = analytic_by_channel.sum()

    sim_mean = result.total_revenue.mean()
    sem = result.total_revenue.std(ddof=1) / np.sqrt(result.n_paths)

    print(f"  naive point estimate (means straight through) : {money(naive_total)}")
    print(f"  analytic E[revenue] (Jensen-corrected for 1/CPC): {money(analytic_total)}")
    print(f"  simulated mean                                 : {money(sim_mean)}")
    print(f"  Monte Carlo standard error of the mean         : {money(sem)}")
    print(f"  simulated / naive                              : {sim_mean / naive_total:.4f}")
    print(f"  simulated / analytic                           : {sim_mean / analytic_total:.4f}")
    print(f"  simulated - analytic, in standard errors       : "
          f"{(sim_mean - analytic_total) / sem:+.2f} SE")

    check("simulated mean within 10% of naive point estimate",
          abs(sim_mean / naive_total - 1.0) < 0.10,
          f"ratio {sim_mean / naive_total:.4f}")
    check("simulated mean within 4 SE of the analytic expectation",
          abs(sim_mean - analytic_total) < 4 * sem,
          f"{(sim_mean - analytic_total) / sem:+.2f} SE")

    print("\n  per-channel revenue contribution:")
    print(f"    {'channel':<14} {'spend':>10} {'naive':>14} {'simulated mean':>16} {'share':>8}")
    sim_by_channel = result.revenue_by_channel.mean(axis=0)
    for i, c in enumerate(channels):
        print(f"    {c:<14} {money(result.allocation[i]):>10} {money(naive_by_channel[i]):>14} "
              f"{money(sim_by_channel[i]):>16} {sim_by_channel[i] / sim_by_channel.sum():>7.1%}")

    # ---- 2. correlation recovery -------------------------------------------
    print("\n== 2. Correlation recovery (did Cholesky actually do anything?) ==")
    rec = validate_correlation_recovery(result.cvr, result.correlation_input)
    print(f"  max abs diff (whole matrix)          : {rec['max_abs_diff']:.6f}")
    print(f"  max abs diff (off-diagonal only)     : {rec['max_abs_diff_offdiag']:.6f}")
    print(f"  mean abs diff (off-diagonal)         : {rec['mean_abs_diff_offdiag']:.6f}")
    print(f"  input off-diagonal mean              : {rec['input_offdiag_mean']:.6f}")
    print(f"  empirical off-diagonal mean          : {rec['empirical_offdiag_mean']:.6f}")
    print(f"  independence baseline (if Cholesky   : {rec['independence_baseline']:.6f}")
    print(f"    had been skipped, diff would be ~this)")

    print("\n  empirical CVR correlation (simulated):")
    _print_matrix(rec["empirical"], channels)
    print("\n  input correlation (bronze):")
    _print_matrix(result.correlation_input, channels)

    check("empirical correlation recovers input within 0.05",
          rec["max_abs_diff_offdiag"] < 0.05,
          f"max off-diagonal diff {rec['max_abs_diff_offdiag']:.4f}")
    check("recovered correlation is far from the independence baseline",
          rec["max_abs_diff_offdiag"] < rec["independence_baseline"] / 4,
          f"{rec['max_abs_diff_offdiag']:.4f} vs baseline {rec['independence_baseline']:.4f}")

    # ---- 3. distribution ----------------------------------------------------
    print("\n== 3. Revenue distribution ==")
    m = risk_metrics(result.total_revenue, confidence=0.95)
    spend_total = result.allocation.sum()
    print(f"  expected revenue        : {money(m['mean'])}")
    print(f"  standard deviation      : {money(m['std'])}  (CV {m['std'] / m['mean']:.2%})")
    print(f"  median                  : {money(m['median'])}")
    print(f"  min / max               : {money(m['min'])} / {money(m['max'])}")
    print(f"  VaR-95  (5th pctile)    : {money(m['var'])}")
    print(f"  CVaR-95 (mean worst 5%) : {money(m['cvar'])}  over {m['tail_n']:,} paths")
    print(f"  expected ROAS           : {m['mean'] / spend_total:.3f}x on {money(spend_total)} spend")
    print(f"  ROAS at VaR-95          : {m['var'] / spend_total:.3f}x")
    print(f"  paths below break-even  : {int((result.total_revenue < spend_total).sum()):,} "
          f"of {result.n_paths:,} ({(result.total_revenue < spend_total).mean():.2%})")

    print("\n  percentiles:")
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"    p{p:<3} {money(float(np.percentile(result.total_revenue, p)))}")

    check("VaR-95 below the mean", m["var"] < m["mean"])
    check("CVaR-95 below VaR-95", m["cvar"] < m["var"])
    check("all simulated revenue strictly positive", bool((result.total_revenue > 0).all()))

    # ---- histogram ----------------------------------------------------------
    try:
        save_histogram(result.total_revenue, m, spend_total)
        print(f"\n  histogram written to {HISTOGRAM_PATH}")
    except ImportError:
        print("\n  histogram skipped: matplotlib not installed")

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 72)
    if FAILURES:
        print(f"SIMULATION VALIDATION FAILED -- {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("SIMULATION VALIDATION PASSED -- all checks green.")


def _print_matrix(matrix: np.ndarray, channels: list[str]) -> None:
    short = [c[:12] for c in channels]
    print("    " + " " * 14 + "".join(f"{s:>13}" for s in short))
    for i, c in enumerate(short):
        print(f"    {c:<14}" + "".join(f"{matrix[i, j]:>13.4f}" for j in range(len(channels))))


def save_histogram(total_revenue: np.ndarray, m: dict, spend_total: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(total_revenue, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)

    tail = total_revenue[total_revenue <= m["var"]]
    ax.hist(tail, bins=20, color="#C44E52", edgecolor="white", linewidth=0.4,
            label=f"worst 5% (CVaR-95 = ${m['cvar']:,.0f})")

    ax.axvline(m["mean"], color="black", linestyle="-", linewidth=2,
               label=f"expected = ${m['mean']:,.0f}")
    ax.axvline(m["var"], color="#C44E52", linestyle="--", linewidth=2,
               label=f"VaR-95 = ${m['var']:,.0f}")
    ax.axvline(spend_total, color="#55A868", linestyle=":", linewidth=2,
               label=f"break-even spend = ${spend_total:,.0f}")

    ax.set_xlabel("Total revenue across all channels ($)")
    ax.set_ylabel("Paths")
    ax.set_title(f"Simulated total revenue -- {len(total_revenue):,} paths, "
                 f"even ${spend_total / 5:,.0f}/channel split, scenario '{SCENARIO}'")
    ax.xaxis.set_major_formatter(lambda v, _: f"${v/1e6:.1f}M")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    HISTOGRAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HISTOGRAM_PATH, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
