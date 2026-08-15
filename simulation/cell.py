"""
One (allocation, scenario) Monte Carlo cell -- the unit of work Phase 3 distributes.

Deliberately Spark-free and Databricks-free: numpy/pandas/scipy only, no I/O.
The Spark layer's `applyInPandas` / `mapInPandas` UDF calls `simulate_cell` and
nothing else, so the modelling can be unit-tested on a laptop and the same code
path is exercised locally and on a cluster.

This module wraps `engine.simulate`; it never reimplements or adjusts its maths.
(Through Phase 3 the engine was literally frozen at its Phase 2 bytes. Phase 4
added an opt-in spend-saturation curve to it -- see `engine.py` and
`saturation.py`. This module only THREADS the new parameters through; the wrap
is still a wrap.)


SATURATION AND SCENARIO MULTIPLIERS COMPOSE MULTIPLICATIVELY
============================================================
Both act on the mean CPC. The order fixed here is SCENARIO FIRST, SATURATION
SECOND:

    effective_mean_cpc = (bronze_mean_cpc * cpc_multiplier) * (spend/ref)**theta

`_scale` applies the scenario here, on the moment vector; the engine applies the
saturation multiplier to whatever mean it receives. In exact arithmetic the two
orderings are identical -- both are the product of the same three factors -- so
the composition is commutative and there is no modelling question to settle.

In float64 it is NOT bitwise commutative, and that was measured rather than
assumed. Across all 21 allocations x 4 scenarios x 5 channels = 420 effective
means on the real bronze moments, scenario-first and saturation-first agree
bitwise on 313 of 420 (74.5%) and differ by at most 1 ulp on the rest (worst
relative difference 2.203e-16, worst absolute difference 4.4e-16 dollars on a
CPC). So the order is fixed here rather than left implicit -- but nothing
downstream can detect the difference, and the `normal` scenario, where
`cpc_multiplier` is exactly 1.0, is bitwise identical either way for all 21
allocations, which is what the Phase 2/3 anchor equivalence depends on.

The economic reading is the intended one: in a seasonal peak the auction is
hotter for everyone (the scenario factor) AND you still pay a volume premium for
buying a large position in a channel (the saturation factor). Neither cancels
the other.

`std_cpc` follows a DIFFERENT rule for the two, and that asymmetry is deliberate
and comes from the spec. The scenario multiplier scales mean and std together
(CV-preserving). Saturation scales only the mean by default, so the CPC
coefficient of variation FALLS as spend rises. `saturate_std_cpc=True` makes
saturation CV-preserving too; it is off by default because the spec says
mean-only. See `saturation.py` for how far the CV actually moves.


WHERE THE SCENARIO MULTIPLIERS ARE APPLIED, AND WHY
===================================================
The multipliers are applied to the ASSUMPTION MOMENTS -- mean and std together,
before the engine fits its marginals -- not to the drawn samples. Three reasons,
each of which was checked rather than assumed:

1. SCALING BOTH MOMENTS PRESERVES THE COEFFICIENT OF VARIATION.
   mean -> c*mean and std -> c*std leaves std/mean unchanged, so a scenario
   moves the LEVEL of a quantity without secretly also changing its relative
   dispersion. If only the mean were scaled, a recession would simultaneously
   be a "CVR is 25% lower" scenario and a "CVR is 33% more volatile relative to
   its mean" scenario, and the two effects would be inseparable in the output.

2. FOR THE LOGNORMALS (CPC and revenue-per-conversion) THIS IS EXACTLY
   EQUIVALENT TO SCALING THE DRAWS.
   Verified rather than taken on trust. `engine.fit_lognormal_moments` computes
       sigma^2 = log1p((std/mean)^2)        -> depends only on the CV, so it is
                                               INVARIANT under (c*mean, c*std)
       mu      = log(mean) - sigma^2/2      -> becomes log(c) + mu
   and numpy draws lognormal as exp(mu + sigma*Z) from the RNG stream, so with
   the same stream every draw is multiplied by exp(log c) = c, pathwise, not
   merely in distribution. Numerically confirmed on the real bronze moments at
   c in {0.82, 0.98, 1.10, 1.25}: sigma matched to within 2 ulp and the
   pathwise ratio draws_scaled/draws stayed within 1.1e-15 relative of c.
   So for CPC and RPC the choice of where to apply the multiplier is a
   free one, and moment-scaling is chosen for consistency with the CVR case.

3. FOR CVR IT IS THE ONLY CORRECT CHOICE.
   CVR is Beta. A Beta variate multiplied by c is NOT Beta: its support becomes
   (0, c), and its shape is not the moment-matched Beta of the scaled moments
   (checked on paid_search at c=1.20 -- refitting gives Beta(33.27, 320.69)
   while naive scaling of the shape parameter would suggest 40.64). For c > 1
   the support alone is disqualifying: a conversion RATE above 1 is not a
   quantity, and scaled draws admit it in principle. Moment-scaling and
   refitting keeps every CVR draw in (0,1) unconditionally, which is why the
   two lognormals are scaled the same way even though they need not be.

GUARD RAILS
===========
`fit_beta_moments` only exists for 0 < mean < 1 with var < mean*(1-mean). A
scenario multiplier can in principle push a channel out of that region, so
`_check_scaled_cvr_legal` validates it FIRST and raises naming the offending
channel, scenario and multiplier -- rather than letting the engine raise a
context-free error deep in a Spark task, and rather than silently clamping. On
the current bronze table there is enormous headroom (mean_cvr tops out at 7.8%,
so the mean bound only binds above c ~ 12.8, and var/max_var is at most 0.0023),
so the guard is defensive, not load-bearing. It matters if the assumptions are
ever refreshed with high-CVR channels.

THE `normal` SCENARIO IS AN EXACT NO-OP
=======================================
`_scale` returns the input array UNTOUCHED when the multiplier is exactly 1.0.
Multiplying float64 by 1.0 is already exact in IEEE-754 and would not change
any bit, but the early return makes the guarantee unconditional and immune to
dtype promotion questions. Nothing else here touches the RNG: `engine.simulate`
is called unmodified, so its draw order (correlated normals -> CPC lognormals
per channel -> RPC lognormals per channel) is preserved exactly. A `normal`
cell at seed 20260808 with the even split is therefore bitwise identical to the
Phase 2 run, and the local check asserts exactly that.

MODELLING SIMPLIFICATIONS CARRIED FORWARD FROM PHASE 2
======================================================
Restated here because they now propagate into 84 cells rather than one:
  * Only CVR is correlated across channels; CPC and revenue-per-conversion are
    drawn independently. Real CPC moves in common across channels (shared
    auction pressure), so cross-channel risk is UNDERSTATED.
  * The Gaussian copula transmits rank correlation, so the realised Pearson
    correlation of the Beta draws is slightly attenuated versus the input.
  * The correlation matrix is held FIXED across scenarios. Real correlations
    rise in a downturn, so the recession scenario's tail risk is understated in
    a way this model cannot show.
  * Revenue was exactly linear in spend through Phase 3 -- no saturation /
    diminishing returns curve -- so expected revenue was a linear function of
    the allocation weights and was maximised at a corner of the simplex. On the
    bronze assumptions paid_search has an expected ROAS of 3.26x per dollar
    against 1.97x for the next best channel, a 65% edge, large enough that
    concentrating into it raised the 5th percentile as well as the mean.
    Measured over the 21 Phase 3 candidates, `dominant_paid_search` maximised
    E[revenue], VaR-95 AND CVaR-95 simultaneously in all four scenarios, so a
    mean-VaR or mean-CVaR "efficient frontier" over that model collapsed to a
    single point.
    FIXED IN PHASE 4, opt-in: pass `theta` to get the saturation curve, which
    makes each channel's contribution strictly concave and moves the optimum off
    the corner. Leaving `theta=None` reproduces the linear Phase 2/3 model
    exactly, so both models remain runnable from this one code path.
  * Multipliers are uniform across channels and deterministic, contributing no
    variance of their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # package-style import (simulation.cell)
    from . import engine
    from .saturation import REFERENCE_SPEND
except ImportError:  # flat import, which is how the rest of this repo runs
    import engine
    from saturation import REFERENCE_SPEND


# Output contract, kept next to the code that produces it so the Spark layer
# can mirror it without guessing. `path_number` is 0-BASED.
CELL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "allocation_id",
    "scenario_id",
    "path_number",
    "total_simulated_revenue",
    "total_spend",
    "roas",
)

# Paste-able schema for applyInPandas / mapInPandas. path_number is int32 ->
# IntegerType; the money columns are float64 -> DoubleType. Matching these
# exactly avoids the classic Arrow "expected X, got Y" failure at collect time.
#
# The string columns' pandas dtype is version-dependent -- `object` on pandas
# <3.0 (Databricks Runtime today), the new `str` dtype on pandas >=3.0 -- which
# Arrow renders as `string` and `large_string` respectively. Both map cleanly to
# Spark STRING, so the DDL below is correct either way; it is only the empty
# frame that has to be built carefully (see `empty_cell_frame`).
CELL_OUTPUT_SPARK_SCHEMA_DDL: str = (
    "allocation_id STRING, "
    "scenario_id STRING, "
    "path_number INT, "
    "total_simulated_revenue DOUBLE, "
    "total_spend DOUBLE, "
    "roas DOUBLE"
)

_PATH_NUMBER_DTYPE = np.int32


def _build_frame(
    allocation_id: str, scenario_id: str, revenue: np.ndarray, total_spend: float
) -> pd.DataFrame:
    """The single place the output frame is constructed, so its dtypes have one origin."""
    n = revenue.shape[0]
    return pd.DataFrame({
        "allocation_id": np.repeat(allocation_id, n),
        "scenario_id": np.repeat(scenario_id, n),
        "path_number": np.arange(n, dtype=_PATH_NUMBER_DTYPE),
        "total_simulated_revenue": revenue,
        "total_spend": np.full(n, total_spend, dtype=np.float64),
        "roas": revenue / total_spend,
    })


def empty_cell_frame() -> pd.DataFrame:
    """Correctly-typed zero-row frame, for a degenerate Spark partition.

    Returning a bare `pd.DataFrame()` from an applyInPandas UDF fails schema
    validation, and hand-declaring the dtypes is fragile: on pandas 3.0 a
    populated frame's id columns come out as the `str` dtype (Arrow
    `large_string`) while `pd.Series([], dtype=object)` comes out as `object`
    (Arrow `null` when empty), which is exactly the mismatch that makes an
    empty partition fail while every populated one succeeds. So this builds a
    real one-row frame through `_build_frame` and slices it away -- the dtypes
    are then correct by construction on any pandas version.
    """
    return _build_frame("", "", np.zeros(1, dtype=np.float64), 1.0).iloc[:0].reset_index(drop=True)


def _scale(values, multiplier: float) -> np.ndarray:
    """Multiply a moment vector, with an exact no-op at multiplier == 1.0."""
    arr = np.asarray(values, dtype=float)
    if multiplier == 1.0:
        return arr
    return arr * multiplier


def _check_multiplier(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def _check_scaled_cvr_legal(
    channels: list[str],
    scaled_mean_cvr: np.ndarray,
    scaled_std_cvr: np.ndarray,
    scenario_id: str,
    cvr_multiplier: float,
) -> None:
    """Raise a contextual error if a scenario pushed CVR outside Beta's domain.

    The engine would raise anyway, but from inside a Spark task and without
    saying which channel or which scenario caused it.
    """
    for i, channel in enumerate(channels):
        mean = float(scaled_mean_cvr[i])
        std = float(scaled_std_cvr[i])
        if not 0.0 < mean < 1.0:
            raise ValueError(
                f"scenario {scenario_id!r} (cvr_multiplier={cvr_multiplier}) pushed "
                f"mean CVR for channel {channel!r} to {mean:.6g}, outside (0,1); a "
                f"conversion rate cannot leave the unit interval. Fix the scenario "
                f"multiplier rather than clamping."
            )
        var = std ** 2
        max_var = mean * (1.0 - mean)
        if not 0.0 < var < max_var:
            raise ValueError(
                f"scenario {scenario_id!r} (cvr_multiplier={cvr_multiplier}) made the "
                f"Beta moment fit impossible for channel {channel!r}: var={var:.6g} must "
                f"be in (0, {max_var:.6g}) for mean={mean:.6g}. No Beta distribution has "
                f"those moments."
            )


def simulate_cell(
    allocation_id: str,
    scenario_id: str,
    channels: list[str],
    allocation: dict[str, float],
    mean_cvr,
    std_cvr,
    mean_cpc,
    std_cpc,
    mean_rpc,
    std_rpc,
    correlation,
    cvr_multiplier: float,
    cpc_multiplier: float,
    revenue_multiplier: float,
    n_paths: int,
    seed: int,
    theta=None,
    reference_spend: float = REFERENCE_SPEND,
    saturate_std_cpc: bool = False,
    spend_floor=None,
) -> pd.DataFrame:
    """Run one (allocation, scenario) cell and return one row per path.

    All moment arrays are indexed consistently with `channels`; the caller owns
    that alignment (`data_access.load_assumptions` orders by channel_id and
    `load_correlation_matrix` reindexes the matrix to match).

    `theta=None` (default) is the linear Phase 2/3 model, unchanged. A (k,)
    array of CPC elasticities aligned to `channels` turns on the Phase 4
    saturation curve; see the module docstring for how it composes with the
    scenario multiplier, and `saturation.theta_vector` for building the array.

    Returns columns `CELL_OUTPUT_COLUMNS`, with `path_number` 0-based and in
    the engine's natural path order (path i is the i-th row of the engine's
    draw matrices, so per-path values stay joinable to any later per-channel
    breakdown of the same seed).
    """
    if not channels:
        raise ValueError("`channels` is empty")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")
    missing = [c for c in channels if c not in allocation]
    if missing:
        raise ValueError(f"allocation is missing channels {missing}")

    cvr_multiplier = _check_multiplier("cvr_multiplier", cvr_multiplier)
    cpc_multiplier = _check_multiplier("cpc_multiplier", cpc_multiplier)
    revenue_multiplier = _check_multiplier("revenue_multiplier", revenue_multiplier)

    # --- apply the scenario to the MOMENTS (see module docstring) ---
    s_mean_cvr = _scale(mean_cvr, cvr_multiplier)
    s_std_cvr = _scale(std_cvr, cvr_multiplier)
    s_mean_cpc = _scale(mean_cpc, cpc_multiplier)
    s_std_cpc = _scale(std_cpc, cpc_multiplier)
    s_mean_rpc = _scale(mean_rpc, revenue_multiplier)
    s_std_rpc = _scale(std_rpc, revenue_multiplier)

    _check_scaled_cvr_legal(channels, s_mean_cvr, s_std_cvr, scenario_id, cvr_multiplier)

    # --- engine called unmodified; saturation is threaded through, not applied here ---
    result = engine.simulate(
        channels=channels,
        allocation=allocation,
        mean_cvr=s_mean_cvr,
        std_cvr=s_std_cvr,
        mean_cpc=s_mean_cpc,
        std_cpc=s_std_cpc,
        mean_rpc=s_mean_rpc,
        std_rpc=s_std_rpc,
        correlation=np.asarray(correlation, dtype=float),
        n_paths=n_paths,
        seed=seed,
        theta=theta,
        reference_spend=reference_spend,
        saturate_std_cpc=saturate_std_cpc,
        spend_floor=spend_floor,
    )

    total_spend = float(result.allocation.sum())
    if total_spend <= 0.0:
        raise ValueError(
            f"allocation {allocation_id!r} has total spend {total_spend}; ROAS is undefined"
        )

    return _build_frame(allocation_id, scenario_id, result.total_revenue, total_spend)


def simulate_cell_from_scenario(
    allocation_id: str,
    scenario,
    channels: list[str],
    allocation: dict[str, float],
    mean_cvr,
    std_cvr,
    mean_cpc,
    std_cpc,
    mean_rpc,
    std_rpc,
    correlation,
    n_paths: int,
    seed: int,
    theta=None,
    reference_spend: float = REFERENCE_SPEND,
    saturate_std_cpc: bool = False,
    spend_floor=None,
) -> pd.DataFrame:
    """Convenience wrapper taking a `scenarios.Scenario` instead of loose floats.

    Useful on the driver / in tests. The Spark UDF should prefer `simulate_cell`
    with plain floats, since broadcasting a dataclass into executors is one more
    thing that can go wrong for no benefit.
    """
    return simulate_cell(
        allocation_id=allocation_id,
        scenario_id=scenario.scenario_id,
        channels=channels,
        allocation=allocation,
        mean_cvr=mean_cvr, std_cvr=std_cvr,
        mean_cpc=mean_cpc, std_cpc=std_cpc,
        mean_rpc=mean_rpc, std_rpc=std_rpc,
        correlation=correlation,
        cvr_multiplier=scenario.cvr_multiplier,
        cpc_multiplier=scenario.cpc_multiplier,
        revenue_multiplier=scenario.revenue_multiplier,
        n_paths=n_paths,
        seed=seed,
        theta=theta,
        reference_spend=reference_spend,
        saturate_std_cpc=saturate_std_cpc,
        spend_floor=spend_floor,
    )
