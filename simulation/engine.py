"""
Single-node correlated Monte Carlo engine for ad-channel revenue.

Pure functions only -- no I/O, no Databricks, no Spark. Everything here is
deterministic given a seed, so the validation in run_phase2_simulation.py can
make exact claims about it.

Model, per path, per channel:

    clicks       = allocated_spend / CPC
    conversions  = clicks * CVR
    revenue      = conversions * revenue_per_conversion

with marginals:

    CVR    ~ Beta(a, b)        method-of-moments from mean_cvr / std_cvr
    CPC    ~ LogNormal(mu, s)  method-of-moments from mean_cpc / std_cpc
    RPC    ~ LogNormal(mu, s)  method-of-moments from mean/std_revenue_per_conversion

CORRELATION -- and its limits, stated plainly:

  * Only CVR is correlated across channels, via a Gaussian copula: standard
    normals are mixed through the Cholesky factor of the input correlation
    matrix, pushed through the normal CDF to uniforms, then inverted through
    each channel's Beta marginal.
  * The input matrix is the correlation of daily *revenue*, used here as a
    proxy for CVR co-movement. Revenue correlation bundles CVR, CPC and
    revenue-per-conversion co-movement together, so attributing all of it to
    CVR overstates CVR correlation somewhat. Deriving a dedicated CVR
    correlation matrix in bronze is the clean fix; noted, not built.
  * CPC and revenue-per-conversion are drawn INDEPENDENTLY across channels.
    Correlating them too is a reasonable extension, deliberately out of scope.
  * A Gaussian copula transmits rank correlation, so the Pearson correlation
    of the resulting Beta draws is slightly ATTENUATED relative to the input.
    That gap is expected and small for near-symmetric marginals -- it is not
    evidence the Cholesky step failed. See validate_correlation_recovery().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --- Method-of-moments parameter fitting ------------------------------------

def fit_beta_moments(mean: float, std: float) -> tuple[float, float]:
    """Beta(a, b) matching a target mean and standard deviation.

    Requires 0 < mean < 1 and var < mean*(1-mean); outside that range no Beta
    exists with those moments and we raise rather than silently clamp.
    """
    if not 0.0 < mean < 1.0:
        raise ValueError(f"Beta mean must be in (0,1), got {mean}")
    var = std ** 2
    max_var = mean * (1.0 - mean)
    if not 0.0 < var < max_var:
        raise ValueError(
            f"Beta moment fit impossible: var={var:.6g} must be in (0, {max_var:.6g}) "
            f"for mean={mean:.6g}"
        )
    common = max_var / var - 1.0
    return mean * common, (1.0 - mean) * common


def fit_lognormal_moments(mean: float, std: float) -> tuple[float, float]:
    """LogNormal underlying-normal (mu, sigma) matching a target mean and std.

    If X = exp(N(mu, sigma^2)) then E[X] = exp(mu + sigma^2/2) and
    Var[X] = (exp(sigma^2) - 1) * exp(2mu + sigma^2). Inverting those gives:
    """
    if mean <= 0:
        raise ValueError(f"LogNormal mean must be > 0, got {mean}")
    if std < 0:
        raise ValueError(f"LogNormal std must be >= 0, got {std}")
    sigma_sq = np.log1p((std / mean) ** 2)
    mu = np.log(mean) - sigma_sq / 2.0
    return float(mu), float(np.sqrt(sigma_sq))


# --- Correlation plumbing ---------------------------------------------------

def cholesky_factor(corr: np.ndarray) -> np.ndarray:
    """Lower-triangular L with L @ L.T == corr.

    Raises with a useful message if the matrix isn't positive definite, which
    is the realistic failure mode for an empirical correlation matrix built
    from few observations or from pairwise-deleted data.
    """
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"correlation matrix must be square, got {corr.shape}")
    if not np.allclose(corr, corr.T, atol=1e-9):
        raise ValueError("correlation matrix is not symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-9):
        raise ValueError("correlation matrix diagonal is not all 1.0")
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        eigenvalues = np.linalg.eigvalsh(corr)
        raise ValueError(
            f"correlation matrix is not positive definite "
            f"(min eigenvalue {eigenvalues.min():.6g}); a nearest-PD repair would "
            f"be needed before Cholesky"
        ) from exc


def correlated_uniforms(chol: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """(n_paths, k) uniforms carrying the correlation structure in `chol`.

    Z ~ N(0, I) -> Y = Z @ L.T has covariance L L.T == corr, and because the
    input matrix has a unit diagonal each Y column is standard normal, so
    Phi(Y) is marginally uniform.
    """
    k = chol.shape[0]
    z = rng.standard_normal((n_paths, k))
    y = z @ chol.T
    return stats.norm.cdf(y)


# --- Simulation -------------------------------------------------------------

@dataclass
class SimulationResult:
    channels: list[str]
    allocation: np.ndarray          # (k,)   dollars per channel
    cvr: np.ndarray                 # (n, k) simulated conversion rates
    cpc: np.ndarray                 # (n, k) simulated cost per click
    rpc: np.ndarray                 # (n, k) simulated revenue per conversion
    clicks: np.ndarray              # (n, k)
    conversions: np.ndarray         # (n, k)
    revenue_by_channel: np.ndarray  # (n, k)
    total_revenue: np.ndarray       # (n,)
    correlation_input: np.ndarray   # (k, k) the matrix fed to Cholesky
    beta_params: dict[str, tuple[float, float]]
    lognormal_cpc_params: dict[str, tuple[float, float]]
    lognormal_rpc_params: dict[str, tuple[float, float]]

    @property
    def n_paths(self) -> int:
        return self.total_revenue.shape[0]


def simulate(
    channels: list[str],
    allocation: dict[str, float],
    mean_cvr: np.ndarray,
    std_cvr: np.ndarray,
    mean_cpc: np.ndarray,
    std_cpc: np.ndarray,
    mean_rpc: np.ndarray,
    std_rpc: np.ndarray,
    correlation: np.ndarray,
    n_paths: int,
    seed: int,
) -> SimulationResult:
    """Run `n_paths` correlated paths for a fixed allocation.

    All array arguments are indexed consistently with `channels`; the caller
    is responsible for that alignment and run_phase2_simulation.py asserts it.
    """
    k = len(channels)
    for name, arr in (("mean_cvr", mean_cvr), ("std_cvr", std_cvr), ("mean_cpc", mean_cpc),
                      ("std_cpc", std_cpc), ("mean_rpc", mean_rpc), ("std_rpc", std_rpc)):
        if len(arr) != k:
            raise ValueError(f"{name} has length {len(arr)}, expected {k}")
    if correlation.shape != (k, k):
        raise ValueError(f"correlation shape {correlation.shape}, expected {(k, k)}")

    rng = np.random.default_rng(seed)
    spend = np.array([allocation[c] for c in channels], dtype=float)

    # 1. correlated CVR through a Gaussian copula
    chol = cholesky_factor(correlation)
    uniforms = correlated_uniforms(chol, n_paths, rng)

    beta_params: dict[str, tuple[float, float]] = {}
    cvr = np.empty((n_paths, k))
    for i, channel in enumerate(channels):
        a, b = fit_beta_moments(float(mean_cvr[i]), float(std_cvr[i]))
        beta_params[channel] = (a, b)
        cvr[:, i] = stats.beta.ppf(uniforms[:, i], a, b)

    # 2. CPC -- independent lognormal per channel
    cpc_params: dict[str, tuple[float, float]] = {}
    cpc = np.empty((n_paths, k))
    for i, channel in enumerate(channels):
        mu, sigma = fit_lognormal_moments(float(mean_cpc[i]), float(std_cpc[i]))
        cpc_params[channel] = (mu, sigma)
        cpc[:, i] = rng.lognormal(mu, sigma, n_paths)

    # 3. revenue per conversion -- independent lognormal per channel
    rpc_params: dict[str, tuple[float, float]] = {}
    rpc = np.empty((n_paths, k))
    for i, channel in enumerate(channels):
        mu, sigma = fit_lognormal_moments(float(mean_rpc[i]), float(std_rpc[i]))
        rpc_params[channel] = (mu, sigma)
        rpc[:, i] = rng.lognormal(mu, sigma, n_paths)

    # 4-7. forward through the funnel
    clicks = spend / cpc
    conversions = clicks * cvr
    revenue_by_channel = conversions * rpc
    total_revenue = revenue_by_channel.sum(axis=1)

    return SimulationResult(
        channels=list(channels),
        allocation=spend,
        cvr=cvr, cpc=cpc, rpc=rpc,
        clicks=clicks, conversions=conversions,
        revenue_by_channel=revenue_by_channel,
        total_revenue=total_revenue,
        correlation_input=np.asarray(correlation, dtype=float),
        beta_params=beta_params,
        lognormal_cpc_params=cpc_params,
        lognormal_rpc_params=rpc_params,
    )


# --- Analytics --------------------------------------------------------------

def naive_point_estimate(channels, allocation, mean_cvr, mean_cpc, mean_rpc) -> np.ndarray:
    """Deterministic revenue, mean assumptions plugged straight through.

    Per-channel array; sum it for the total.
    """
    spend = np.array([allocation[c] for c in channels], dtype=float)
    return (spend / np.asarray(mean_cpc, dtype=float)) * np.asarray(mean_cvr, dtype=float) \
        * np.asarray(mean_rpc, dtype=float)


def jensen_corrected_expectation(naive_by_channel: np.ndarray, cpc_sigmas: np.ndarray) -> np.ndarray:
    """Analytic E[revenue], correcting the naive estimate for E[1/CPC] != 1/E[CPC].

    clicks = spend / CPC is convex in CPC, so the simulated mean sits ABOVE the
    naive point estimate. For CPC ~ LogNormal(mu, sigma):

        E[1/CPC] = exp(-mu + sigma^2/2),  1/E[CPC] = exp(-mu - sigma^2/2)

    so the ratio is exactly exp(sigma^2). CVR and revenue-per-conversion enter
    linearly and are independent of CPC, so they contribute no bias. This gives
    an exact target for the simulated mean -- a far sharper test than
    "same ballpark".
    """
    return np.asarray(naive_by_channel, dtype=float) * np.exp(np.asarray(cpc_sigmas, dtype=float) ** 2)


def risk_metrics(total_revenue: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    """Expected value, dispersion, VaR and CVaR at `confidence`.

    VaR-95 is the 5th percentile of the revenue distribution: a 95% chance
    revenue lands above it. CVaR-95 is the mean of the outcomes at or below
    that percentile -- the average of the worst 5% of paths.
    """
    tail_pct = (1.0 - confidence) * 100.0
    var = float(np.percentile(total_revenue, tail_pct))
    tail = total_revenue[total_revenue <= var]
    return {
        "mean": float(total_revenue.mean()),
        "std": float(total_revenue.std(ddof=1)),
        "min": float(total_revenue.min()),
        "max": float(total_revenue.max()),
        "median": float(np.median(total_revenue)),
        "var": var,
        "cvar": float(tail.mean()) if tail.size else float("nan"),
        "tail_n": int(tail.size),
    }


def validate_correlation_recovery(simulated_cvr: np.ndarray, correlation_input: np.ndarray) -> dict:
    """Empirical CVR correlation vs the matrix fed to Cholesky.

    Also reports the independence baseline: if the Cholesky step had silently
    produced independent draws, the empirical off-diagonals would sit near
    zero and max_abs_diff would be roughly the largest input off-diagonal.
    That contrast is what makes this a real test.
    """
    empirical = np.corrcoef(simulated_cvr, rowvar=False)
    diff = np.abs(empirical - correlation_input)
    k = correlation_input.shape[0]
    off = ~np.eye(k, dtype=bool)
    return {
        "empirical": empirical,
        "abs_diff": diff,
        "max_abs_diff": float(diff.max()),
        "max_abs_diff_offdiag": float(diff[off].max()),
        "mean_abs_diff_offdiag": float(diff[off].mean()),
        "empirical_offdiag_mean": float(empirical[off].mean()),
        "input_offdiag_mean": float(correlation_input[off].mean()),
        "independence_baseline": float(np.abs(correlation_input[off]).max()),
    }
