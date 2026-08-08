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
    TBL_CVR_CORRELATION,
    TBL_HISTORY,
    get_client,
    print_table,
    resolve_warehouse_id,
    sql,
)

HEATMAP_PATH = REPO_ROOT / "data" / "validation" / "correlation_heatmap.png"

CORR_COLUMNS = ["channel_id_a", "channel_id_b", "correlation_coefficient"]

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
    n_cvr_corr = int(scalar(w, wid, f"SELECT COUNT(*) FROM {TBL_CVR_CORRELATION}")[0])
    print(f"  {TBL_ASSUMPTIONS}: {n_assume} rows")
    print(f"  {TBL_CORRELATION}: {n_corr} rows")
    print(f"  {TBL_CVR_CORRELATION}: {n_cvr_corr} rows")

    check("history row count == days x channels", n_rows == n_days * n_channels,
          f"{n_rows} vs {n_days}x{n_channels}={n_days * n_channels}")
    check("one assumption row per channel", n_assume == n_channels, f"{n_assume} vs {n_channels}")
    check("revenue correlation matrix is complete NxN", n_corr == n_channels ** 2,
          f"{n_corr} vs {n_channels}^2={n_channels ** 2}")
    check("CVR correlation matrix is complete NxN", n_cvr_corr == n_channels ** 2,
          f"{n_cvr_corr} vs {n_channels}^2={n_channels ** 2}")

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
        TBL_CORRELATION: CORR_COLUMNS,
        TBL_CVR_CORRELATION: CORR_COLUMNS,
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
    matrices = {}
    for label, table in (("revenue-based", TBL_CORRELATION), ("CVR-based", TBL_CVR_CORRELATION)):
        print(f"\n== 4. Correlation matrix -- {label} ({table.split('.')[-1]}) ==")
        short = table.split(".")[-1]

        diag_bad = int(scalar(w, wid, f"""
            SELECT COUNT(*) FROM {table}
            WHERE channel_id_a = channel_id_b AND ABS(correlation_coefficient - 1.0) > 1e-9
        """)[0])
        check(f"{short}: diagonal == 1.0", diag_bad == 0, f"{diag_bad} bad diagonal entries")

        range_bad = int(scalar(w, wid, f"""
            SELECT COUNT(*) FROM {table}
            WHERE correlation_coefficient < -1.0 OR correlation_coefficient > 1.0
        """)[0])
        check(f"{short}: all coefficients within [-1, 1]", range_bad == 0,
              f"{range_bad} out of range")

        asym = int(scalar(w, wid, f"""
            SELECT COUNT(*) FROM {table} x
            JOIN {table} y
              ON x.channel_id_a = y.channel_id_b AND x.channel_id_b = y.channel_id_a
            WHERE ABS(x.correlation_coefficient - y.correlation_coefficient) > 1e-9
        """)[0])
        check(f"{short}: matrix is symmetric", asym == 0, f"{asym} asymmetric pairs")

        _, corr_rows = sql(w, wid, f"""
            SELECT channel_id_a, channel_id_b, correlation_coefficient
            FROM {table} ORDER BY channel_id_a, channel_id_b
        """)
        matrices[label] = corr_rows

        offdiag = [float(r[2]) for r in corr_rows if r[0] != r[1]]
        print(f"  off-diagonal: min={min(offdiag):.4f}  max={max(offdiag):.4f}  "
              f"mean={sum(offdiag) / len(offdiag):.4f}")
        check(f"{short}: off-diagonal not trivially zero",
              all(abs(v) > 1e-6 for v in offdiag) and sum(offdiag) / len(offdiag) > 0.01)

        # Positive definiteness. The simulation Choleskys this matrix, and
        # pairwise-deleted correlations are not guaranteed PD, so check rather
        # than discover it at simulation time.
        channels_sorted, matrix = _dense(corr_rows)
        eigenvalues = _eigenvalues(matrix)
        if eigenvalues is None:
            print("  eigenvalue check skipped: numpy not installed")
        else:
            print(f"  eigenvalues: {', '.join(f'{e:.4f}' for e in eigenvalues)}")
            check(f"{short}: positive definite (Cholesky-able)", min(eigenvalues) > 1e-10,
                  f"min eigenvalue {min(eigenvalues):.3e}")

    # ---- 4b. the two matrices side by side ---------------------------------
    print("\n== 4b. CVR-based vs revenue-based (off-diagonal pairs) ==")
    cols, rows = sql(w, wid, f"""
        SELECT c.channel_id_a, c.channel_id_b,
               ROUND(r.correlation_coefficient, 4) AS revenue_based,
               ROUND(c.correlation_coefficient, 4) AS cvr_based,
               ROUND(c.correlation_coefficient - r.correlation_coefficient, 4) AS delta
        FROM {TBL_CVR_CORRELATION} c
        JOIN {TBL_CORRELATION} r
          ON c.channel_id_a = r.channel_id_a AND c.channel_id_b = r.channel_id_b
        WHERE c.channel_id_a < c.channel_id_b
        ORDER BY c.channel_id_a, c.channel_id_b
    """)
    print_table(cols, rows)

    # ---- 5. heatmaps --------------------------------------------------------
    print("\n== 5. Correlation heatmaps ==")
    try:
        render_heatmaps(matrices)
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


def _dense(corr_rows):
    """Long-form rows -> (ordered channels, dense NxN list-of-lists)."""
    channels = sorted({r[0] for r in corr_rows})
    idx = {c: i for i, c in enumerate(channels)}
    n = len(channels)
    matrix = [[0.0] * n for _ in range(n)]
    for a, b, v in corr_rows:
        matrix[idx[a]][idx[b]] = float(v)
    return channels, matrix


def _eigenvalues(matrix):
    try:
        import numpy as np
    except ImportError:
        return None
    return sorted(np.linalg.eigvalsh(np.array(matrix, dtype=float)).tolist())


def render_heatmaps(matrices: dict) -> None:
    """One figure, both matrices side by side, plus their difference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rev_channels, rev = _dense(matrices["revenue-based"])
    cvr_channels, cvr = _dense(matrices["CVR-based"])
    if rev_channels != cvr_channels:
        raise ValueError("the two correlation matrices disagree on channel set")
    channels, n = rev_channels, len(rev_channels)
    delta = [[cvr[i][j] - rev[i][j] for j in range(n)] for i in range(n)]

    panels = [
        ("Daily REVENUE correlation\n(channel_correlation_matrix -- diagnostic)", rev, "RdYlBu_r", -1, 1),
        ("Daily CVR correlation\n(channel_cvr_correlation_matrix -- drives the sim)", cvr, "RdYlBu_r", -1, 1),
        ("Difference (CVR - revenue)", delta, "PuOr_r", -0.2, 0.2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (title, matrix, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n), channels, rotation=45, ha="right")
        ax.set_yticks(range(n), channels)
        span = max(abs(vmin), abs(vmax))
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                        color="white" if abs(matrix[i][j]) > 0.6 * span else "black", fontsize=9)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(HEATMAP_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HEATMAP_PATH, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
