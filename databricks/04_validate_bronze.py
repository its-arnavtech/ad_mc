"""
Step 6 -- sanity-check the bronze layer and print the actual numbers.

Checks:
  * row counts for all three tables (expected: 730 days x 5 channels)
  * null scan on every column of every table
  * CTR / CVR plausibility -- must sit strictly inside (0, 1)
  * spend / revenue non-negative, no duplicate (date, channel_id) keys
  * correlation matrix well-formedness: complete NxN, diagonal == 1,
    symmetric, every coefficient within [-1, 1]
  * correlation heatmap written to data/validation/correlation_heatmap.png

Exits non-zero if any check fails, so it can gate later phases.

    python databricks/04_validate_bronze.py
"""

from pathlib import Path

from _common import (
    REPO_ROOT,
    TBL_ASSUMPTIONS,
    TBL_CORRELATION,
    TBL_HISTORY,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)

HEATMAP_PATH = REPO_ROOT / "data" / "validation" / "correlation_heatmap.png"

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}{(' -- ' + detail) if detail else ''}")


def scalar(w, wid, statement):
    _, rows = sql(w, wid, statement)
    return rows[0] if rows else []


def main() -> None:
    w = get_client()
    wid = resolve_warehouse_id(w)

    # ---- 1. row counts ------------------------------------------------------
    print("\n== 1. Row counts ==")
    hist = scalar(
        w, wid,
        f"SELECT COUNT(*), COUNT(DISTINCT channel_id), COUNT(DISTINCT date), MIN(date), MAX(date) FROM {TBL_HISTORY}",
    )
    n_rows, n_channels, n_days, first_date, last_date = (
        int(hist[0]), int(hist[1]), int(hist[2]), hist[3], hist[4],
    )
    print(f"  {TBL_HISTORY}: {n_rows:,} rows | {n_channels} channels | {n_days} days | {first_date} -> {last_date}")

    n_assume = int(scalar(w, wid, f"SELECT COUNT(*) FROM {TBL_ASSUMPTIONS}")[0])
    n_corr = int(scalar(w, wid, f"SELECT COUNT(*) FROM {TBL_CORRELATION}")[0])
    print(f"  {TBL_ASSUMPTIONS}: {n_assume} rows")
    print(f"  {TBL_CORRELATION}: {n_corr} rows")

    check("history row count == days x channels", n_rows == n_days * n_channels,
          f"{n_rows} vs {n_days}x{n_channels}={n_days * n_channels}")
    check("one assumption row per channel", n_assume == n_channels, f"{n_assume} vs {n_channels}")
    check("correlation matrix is complete NxN", n_corr == n_channels ** 2,
          f"{n_corr} vs {n_channels}^2={n_channels ** 2}")

    dupes = int(scalar(w, wid, f"""
        SELECT COUNT(*) FROM (
            SELECT date, channel_id FROM {TBL_HISTORY}
            GROUP BY date, channel_id HAVING COUNT(*) > 1
        )""")[0])
    check("no duplicate (date, channel_id) keys", dupes == 0, f"{dupes} duplicate keys")

    # ---- 2. null scan -------------------------------------------------------
    print("\n== 2. Null scan ==")
    table_cols = {
        TBL_HISTORY: ["date", "channel_id", "impressions", "clicks", "conversions", "spend", "revenue"],
        TBL_ASSUMPTIONS: ["channel_id", "channel_name", "mean_ctr", "std_ctr", "mean_cpc", "std_cpc",
                          "mean_cvr", "std_cvr", "mean_revenue_per_conversion",
                          "std_revenue_per_conversion", "distribution_type"],
        TBL_CORRELATION: ["channel_id_a", "channel_id_b", "correlation_coefficient"],
    }
    for table, columns in table_cols.items():
        expr = ", ".join(f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in columns)
        counts = [int(v) for v in scalar(w, wid, f"SELECT {expr} FROM {table}")]
        offenders = {c: n for c, n in zip(columns, counts) if n > 0}
        check(f"{table.split('.')[-1]}: no nulls", not offenders, str(offenders) if offenders else "")

    # ---- 3. CTR / CVR plausibility -----------------------------------------
    print("\n== 3. CTR / CVR / value ranges ==")
    cols, rows = sql(w, wid, f"""
        WITH daily AS (
            SELECT channel_id,
                   clicks / impressions      AS ctr,
                   conversions / NULLIF(clicks, 0) AS cvr
            FROM {TBL_HISTORY}
            WHERE impressions > 0
        )
        SELECT channel_id,
               ROUND(MIN(ctr), 5) AS min_ctr, ROUND(MAX(ctr), 5) AS max_ctr,
               ROUND(MIN(cvr), 5) AS min_cvr, ROUND(MAX(cvr), 5) AS max_cvr
        FROM daily GROUP BY channel_id ORDER BY channel_id
    """)
    print_table(cols, rows)

    bad = int(scalar(w, wid, f"""
        SELECT COUNT(*) FROM {TBL_HISTORY}
        WHERE impressions <= 0 OR clicks <= 0 OR conversions <= 0
           OR clicks > impressions OR conversions > clicks
    """)[0])
    check("every row has 0 < conversions <= clicks <= impressions", bad == 0, f"{bad} violating rows")

    neg = int(scalar(w, wid, f"SELECT COUNT(*) FROM {TBL_HISTORY} WHERE spend < 0 OR revenue < 0")[0])
    check("spend and revenue non-negative", neg == 0, f"{neg} negative rows")

    out_of_range = int(scalar(w, wid, f"""
        SELECT COUNT(*) FROM {TBL_ASSUMPTIONS}
        WHERE mean_ctr <= 0 OR mean_ctr >= 1 OR mean_cvr <= 0 OR mean_cvr >= 1
           OR std_ctr < 0 OR std_cvr < 0
           OR mean_cpc <= 0 OR std_cpc < 0
           OR mean_revenue_per_conversion <= 0 OR std_revenue_per_conversion < 0
    """)[0])
    check("assumption means/stds inside plausible bounds", out_of_range == 0, f"{out_of_range} bad rows")

    # Beta method-of-moments needs var < mean*(1-mean); if this fails the
    # Phase 2 CVR fit is impossible, so catch it here rather than at sim time.
    beta_infeasible = int(scalar(w, wid, f"""
        SELECT COUNT(*) FROM {TBL_ASSUMPTIONS}
        WHERE POWER(std_cvr, 2) >= mean_cvr * (1 - mean_cvr)
    """)[0])
    check("CVR admits a Beta method-of-moments fit", beta_infeasible == 0,
          f"{beta_infeasible} channels with var >= mean*(1-mean)")

    # ---- 4. correlation matrix well-formedness -----------------------------
    print("\n== 4. Correlation matrix ==")
    diag_bad = int(scalar(w, wid, f"""
        SELECT COUNT(*) FROM {TBL_CORRELATION}
        WHERE channel_id_a = channel_id_b AND ABS(correlation_coefficient - 1.0) > 1e-9
    """)[0])
    check("diagonal == 1.0", diag_bad == 0, f"{diag_bad} bad diagonal entries")

    range_bad = int(scalar(w, wid, f"""
        SELECT COUNT(*) FROM {TBL_CORRELATION}
        WHERE correlation_coefficient < -1.0 OR correlation_coefficient > 1.0
    """)[0])
    check("all coefficients within [-1, 1]", range_bad == 0, f"{range_bad} out of range")

    asym = int(scalar(w, wid, f"""
        SELECT COUNT(*) FROM {TBL_CORRELATION} x
        JOIN {TBL_CORRELATION} y
          ON x.channel_id_a = y.channel_id_b AND x.channel_id_b = y.channel_id_a
        WHERE ABS(x.correlation_coefficient - y.correlation_coefficient) > 1e-9
    """)[0])
    check("matrix is symmetric", asym == 0, f"{asym} asymmetric pairs")

    cols, corr_rows = sql(w, wid, f"""
        SELECT channel_id_a, channel_id_b, ROUND(correlation_coefficient, 4)
        FROM {TBL_CORRELATION} ORDER BY channel_id_a, channel_id_b
    """)

    offdiag = [float(r[2]) for r in corr_rows if r[0] != r[1]]
    if offdiag:
        print(f"\n  off-diagonal correlations: min={min(offdiag):.4f}  "
              f"max={max(offdiag):.4f}  mean={sum(offdiag) / len(offdiag):.4f}")

    # ---- 5. heatmap ---------------------------------------------------------
    print("\n== 5. Correlation heatmap ==")
    try:
        render_heatmap(corr_rows)
        print(f"  written to {HEATMAP_PATH}")
    except ImportError:
        print("  skipped: matplotlib not installed (pip install matplotlib)")

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"VALIDATION FAILED -- {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("VALIDATION PASSED -- all checks green.")


def render_heatmap(corr_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = sorted({r[0] for r in corr_rows})
    idx = {c: i for i, c in enumerate(channels)}
    n = len(channels)
    matrix = [[0.0] * n for _ in range(n)]
    for a, b, v in corr_rows:
        matrix[idx[a]][idx[b]] = float(v)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), channels, rotation=45, ha="right")
    ax.set_yticks(range(n), channels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                    color="white" if abs(matrix[i][j]) > 0.6 else "black", fontsize=9)
    ax.set_title("Daily revenue correlation across channels\n(bronze.channel_correlation_matrix)")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    Path(HEATMAP_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HEATMAP_PATH, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
