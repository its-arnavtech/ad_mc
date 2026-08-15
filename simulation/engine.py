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
    CPC    ~ LogNormal(mu, s)  method-of-moments from effective mean_cpc / std_cpc
    RPC    ~ LogNormal(mu, s)  method-of-moments from mean/std_revenue_per_conversion


PHASE 4 CHANGE -- SPEND SATURATION. THIS FILE WAS FROZEN AFTER PHASE 2
======================================================================
This module was deliberately unchanged from Phase 2 (`43c9163`) through the
whole of Phase 3 -- the distributed layer wrapped it rather than editing it, so
"distributing the engine did not change the math" could be proven bitwise. This
is the first edit since, and it is a modelling change, not a refactor.

WHAT CHANGED: `simulate()` gained an optional `theta` (plus `reference_spend`
and `saturate_std_cpc`). When `theta` is supplied, the MEAN CPC fed into the
lognormal moment fit becomes spend-dependent:

    effective_mean_cpc_i = mean_cpc_i * (spend_i / reference_spend) ** theta_i

so revenue stops being linear in spend and becomes a power law with exponent
(1 - theta_i). See `saturation.py` for the economics, the derivation of theta
from bronze impression volumes, and the assumptions involved.

WHY: without a response curve, expected revenue is linear in the allocation
weights, so its maximum is always a simplex corner and concentration costs only
variance, never mean. Phase 3 measured the consequence over 21 candidates x 4
scenarios: `dominant_paid_search` maximised expected revenue, VaR-95 AND
CVaR-95 in all four scenarios and the mean-VaR Pareto set was a single point.
An "efficient frontier" over that model is not a frontier.

WHAT DID NOT CHANGE:
  * With `theta=None` (the default) this function is behaviourally identical to
    Phase 2/3 -- the saturation block short-circuits and the original arrays are
    passed through by identity, not multiplied by 1.0.
  * Even with theta supplied, at spend == reference_spend the multiplier is
    `1.0 ** theta`, exactly 1.0 in IEEE-754, and `x * 1.0 == x` bitwise. So the
    even-$100k/channel anchor reproduces Phase 2/3 to the bit either way.
  * THE RNG DRAW ORDER AND DRAW COUNT ARE UNTOUCHED: correlated normals, then
    one CPC lognormal block per channel in `channels` order, then one RPC block
    per channel. Saturation only alters the (mu, sigma) handed to
    `rng.lognormal`; it adds no draw, removes none, and reorders none.

ZERO SPEND (PHASE 4 OPTIMIZATION)
=================================
A channel may now be allocated exactly $0. Its clicks are `0.0 / CPC == 0.0`
exactly and it contributes no revenue -- which is the true limit of the
saturation curve, since revenue is proportional to `spend**(1 - theta)`. It is
NOT an error and NOT an infinity. The idle channel still draws its own CPC and
RPC block from the stream (using placeholder moments it cannot influence
anything with), so dropping a channel does not shift any other channel's draws;
that is what lets Phase 4's optimizer compare allocations with different support
under common random numbers. NEGATIVE spend raises, in both the saturated and
the unsaturated model.

A CONSEQUENCE OF THE SPEC WORTH KNOWING (see saturation.py for the numbers):
only the MEAN cpc is spend-dependent by default. `fit_lognormal_moments`
derives sigma from std/mean, so a rising mean against bronze's fixed std means
the CPC distribution gets relatively TIGHTER as spend grows and the Jensen
correction exp(sigma^2) shrinks. `saturate_std_cpc=True` selects the
CV-preserving variant instead (std scaled by the same multiplier, sigma exactly
invariant), which matches how Phase 3's scenario multipliers behave.

CORRELATION -- and its limits, stated plainly:

  * Only CVR is correlated across channels, via a Gaussian copula: standard
    normals are mixed through the Cholesky factor of the input correlation
    matrix, pushed through the normal CDF to uniforms, then inverted through
    each channel's Beta marginal.
  * The input matrix is `bronze.channel_cvr_correlation_matrix` -- Pearson
    correlation measured directly on daily CVR, which is the quantity actually
    being correlated here.
    Phase 2 originally used the daily-*revenue* matrix as a proxy. That bundles
    CTR, CVR and revenue-per-conversion co-movement together, and empirically
    it UNDERstates CVR correlation (mean off-diagonal 0.371 vs 0.406): revenue
    carries a large independent per-channel revenue-per-conversion noise term
    that dilutes the shared macro signal, while CVR does not. The revenue
    matrix stays in bronze as a diagnostic but no longer feeds the simulation.
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

try:  # package-style import (simulation.engine / ad_mc_sim.engine)
    from .saturation import REFERENCE_SPEND, effective_cpc_moments
except ImportError:
    try:  # flat import, which is how the rest of this repo runs
        from saturation import REFERENCE_SPEND, effective_cpc_moments
    except ImportError as exc:  # pragma: no cover -- packaging mistake, not a code path
        # This is the failure mode a Spark executor will hit if the wheel is built
        # without saturation.py. Raising here with the fix named is worth the six
        # lines: the alternative is a bare ModuleNotFoundError surfacing from
        # inside a task, where it reads as "the engine is broken" rather than
        # "the package is incomplete".
        raise ImportError(
            "engine.py requires saturation.py (added in Phase 4 for the spend response "
            "curve). If this is a Spark executor, the wheel was built without it: add "
            "'saturation.py' to PACKAGE_MODULES in simulation/run_phase3_distributed.py "
            "and rebuild. The wheel's content hash will change, which is expected."
        ) from exc


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
    # --- saturation (Phase 4). All None when saturation is off, so a result
    # from the Phase 2/3 code path is distinguishable from a saturated one
    # rather than silently looking the same.
    theta: np.ndarray | None = None                  # (k,) CPC elasticity w.r.t. spend
    reference_spend: float | None = None
    saturation_multiplier: np.ndarray | None = None  # (k,) (spend/ref)**theta, 0.0 if idle
    # (k,) True where the channel was funded BELOW its evidence floor, so the
    # curve was evaluated at the floor instead of at the actual spend. Surfaced
    # rather than buried: an allocation relying on the clip is priced at the
    # edge of the data, not inside it. All False when no floor is supplied.
    spend_floor_applied: np.ndarray | None = None
    effective_mean_cpc: np.ndarray | None = None     # (k,) mean actually fitted
    effective_std_cpc: np.ndarray | None = None      # (k,) std actually fitted
    # (k,) True where spend == 0. Those channels contribute exactly zero clicks,
    # conversions and revenue; their `effective_mean_cpc` above is a PLACEHOLDER
    # (the unsaturated bronze mean) that exists only so the lognormal fit is well
    # posed and the RNG draw order is unchanged. Never read it for an idle
    # channel -- read `saturation_multiplier`, which is the true 0.0.
    idle_channel: np.ndarray | None = None

    @property
    def n_paths(self) -> int:
        return self.total_revenue.shape[0]

    @property
    def is_saturated(self) -> bool:
        return self.theta is not None


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
    theta: np.ndarray | None = None,
    reference_spend: float = REFERENCE_SPEND,
    saturate_std_cpc: bool = False,
    spend_floor: np.ndarray | None = None,
) -> SimulationResult:
    """Run `n_paths` correlated paths for a fixed allocation.

    All array arguments are indexed consistently with `channels`; the caller
    is responsible for that alignment and run_phase2_simulation.py asserts it.

    SATURATION (Phase 4, opt-in):

    `theta=None` -- the default -- is exactly the Phase 2/3 engine. Nothing in
    the saturation block executes and the caller's `mean_cpc` / `std_cpc` arrays
    reach `fit_lognormal_moments` by identity.

    `theta` as a (k,) array of CPC elasticities makes the fitted mean CPC
    spend-dependent, `mean_cpc_i * (spend_i/reference_spend) ** theta_i`, which
    turns revenue from linear in spend into a power law with exponent
    (1 - theta_i). `saturate_std_cpc=True` additionally scales std_cpc by the
    same multiplier, preserving the coefficient of variation (and therefore
    sigma, and therefore the Jensen correction) exactly; the default False
    implements the spec as written, mean only. See `saturation.py`.

    ORDER OF COMPOSITION WITH SCENARIO MULTIPLIERS. `cell.py` scales the bronze
    moments by the scenario's `cpc_multiplier` BEFORE calling this function, so
    the effective mean is `(bronze_mean * cpc_mult) * (spend/ref)**theta`. Both
    factors are multiplicative on the mean, so the two orderings agree in exact
    arithmetic; in float64 they can differ in the last ulp, so the order is
    fixed here (scenario first, saturation second) rather than left to chance.
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

    # Spend must be finite and non-negative in BOTH models. Zero is allowed and
    # means "this channel is not funded": clicks = 0/CPC = 0 exactly, so the
    # channel contributes nothing and nothing downstream can go NaN or infinite.
    # Negative spend has no interpretation and is always a caller bug, so it
    # raises here rather than silently producing negative revenue.
    if not np.all(np.isfinite(spend)):
        raise ValueError(f"allocation must be finite, got {dict(zip(channels, spend))}")
    if np.any(spend < 0.0):
        raise ValueError(
            f"allocation has negative spend: "
            f"{ {c: s for c, s in zip(channels, spend) if s < 0.0} }. Zero is allowed "
            f"(the channel simply earns nothing); negative is not."
        )

    # 0. spend saturation -- the ONLY thing that changes here is the mean (and
    #    optionally the std) handed to the CPC moment fit in step 2. No RNG call
    #    happens in this block, so the draw order below is unaffected.
    if theta is None:
        sat_mult = None
        idle = None
        floored = np.zeros(k, dtype=bool)
        eff_mean_cpc = mean_cpc          # by identity: not `* 1.0`
        eff_std_cpc = std_cpc
    else:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (k,):
            raise ValueError(f"theta shape {theta.shape}, expected {(k,)}")
        if spend_floor is not None:
            spend_floor = np.asarray(spend_floor, dtype=float)
            if spend_floor.shape != (k,):
                raise ValueError(
                    f"spend_floor shape {spend_floor.shape}, expected {(k,)}"
                )
        eff_mean_cpc, eff_std_cpc, sat_mult, idle, floored = effective_cpc_moments(
            mean_cpc, std_cpc, spend, theta, reference_spend, saturate_std_cpc,
            spend_floor,
        )

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
        mu, sigma = fit_lognormal_moments(float(eff_mean_cpc[i]), float(eff_std_cpc[i]))
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

    # A lognormal draw is exp(finite), so CPC is strictly positive and this can
    # only fire if something upstream is broken (a zero CPC would give inf for a
    # funded channel and NaN for an idle one -- the exact failure mode the
    # zero-spend path has to be safe against). O(n_paths), one pass, cheap.
    if not np.all(np.isfinite(total_revenue)):
        n_bad = int((~np.isfinite(total_revenue)).sum())
        raise ValueError(
            f"{n_bad} of {n_paths} paths produced non-finite revenue; CPC min was "
            f"{float(cpc.min()):.6g} (must be > 0). Allocation: "
            f"{dict(zip(channels, spend))}"
        )

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
        theta=theta,
        reference_spend=None if theta is None else float(reference_spend),
        saturation_multiplier=sat_mult,
        spend_floor_applied=floored,
        effective_mean_cpc=None if theta is None else np.asarray(eff_mean_cpc, dtype=float),
        effective_std_cpc=None if theta is None else np.asarray(eff_std_cpc, dtype=float),
        idle_channel=idle,
    )


# --- Analytics --------------------------------------------------------------

def naive_point_estimate(channels, allocation, mean_cvr, mean_cpc, mean_rpc) -> np.ndarray:
    """Deterministic revenue, mean assumptions plugged straight through.

    Per-channel array; sum it for the total.

    UNDER SATURATION the caller must pass the EFFECTIVE mean CPC
    (`saturation.effective_mean_cpc(...)`, or `result.effective_mean_cpc`), not
    the bronze one -- otherwise this silently reports the Phase 3 linear model's
    answer and the comparison against a saturated simulation is meaningless.
    `analytic_expected_revenue` below does that bookkeeping automatically and is
    the safer choice for a validation target.
    """
    spend = np.array([allocation[c] for c in channels], dtype=float)
    return (spend / np.asarray(mean_cpc, dtype=float)) * np.asarray(mean_cvr, dtype=float) \
        * np.asarray(mean_rpc, dtype=float)


def analytic_expected_revenue(result: "SimulationResult") -> np.ndarray:
    """Exact analytic E[revenue] per channel, reconstructed from a result's own fits.

    Self-contained: every input is recovered from the fitted marginals the run
    actually used, so it is correct with or without saturation and cannot drift
    out of sync with the engine by using a stale mean.

        E[revenue_i] = spend_i * E[1/CPC_i] * E[CVR_i] * E[RPC_i]
        E[1/CPC]     = exp(-mu + sigma^2/2)      for LogNormal(mu, sigma)
        E[CVR]       = a / (a + b)               for Beta(a, b)
        E[RPC]       = exp(mu + sigma^2/2)

    CVR, CPC and RPC are mutually independent in this model (only CVR is
    correlated, and only ACROSS channels), so the expectation factorises exactly
    and this is a hard target for the simulated mean, not an approximation.
    """
    spend = np.asarray(result.allocation, dtype=float)
    out = np.empty(len(result.channels), dtype=float)
    for i, channel in enumerate(result.channels):
        a, b = result.beta_params[channel]
        cpc_mu, cpc_sigma = result.lognormal_cpc_params[channel]
        rpc_mu, rpc_sigma = result.lognormal_rpc_params[channel]
        e_inv_cpc = np.exp(-cpc_mu + cpc_sigma ** 2 / 2.0)
        e_cvr = a / (a + b)
        e_rpc = np.exp(rpc_mu + rpc_sigma ** 2 / 2.0)
        out[i] = spend[i] * e_inv_cpc * e_cvr * e_rpc
    return out


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
