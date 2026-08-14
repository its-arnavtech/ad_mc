"""
Spend saturation (diminishing returns) for the ad-channel simulation.

WHY THIS EXISTS
===============
Through Phase 3 the model had no response curve at all. Within a channel
`clicks = spend / CPC` with CPC independent of spend, so revenue was exactly
LINEAR in spend, expected total revenue was a linear function of the weight
vector, and the maximum therefore sat at a corner of the simplex. Measured over
Phase 3's 21 candidates x 4 scenarios, `dominant_paid_search` (80/5/5/5/5)
maximised expected revenue, VaR-95 AND CVaR-95 simultaneously in all four
scenarios, and the mean-VaR Pareto set was a single point. There was nothing
for an optimizer to trade off. This module supplies the missing curvature.

WHERE THE CURVATURE IS APPLIED, AND THE ECONOMICS OF THAT CHOICE
================================================================
Diminishing returns enter at the COST side, not as a generic concave function
bolted onto revenue:

    effective_mean_cpc(spend) = mean_cpc_bronze * (spend / reference_spend) ** theta

Buying more volume in a channel exhausts the cheapest inventory first. To go
further you bid into progressively more expensive auctions and progressively
less well-matched audiences, so your AVERAGE cost per click rises with the
volume you buy. That is a statement about the supply curve of ad inventory,
which is observable and which an advertiser can actually measure, rather than
an unexplained "response curve" fitted to revenue.

It is also not a weaker assumption than a revenue-side curve -- it is the same
one written in more testable form. Substituting into the funnel:

    clicks = spend / (mean_cpc * (spend/ref)**theta * noise)
           = (ref**theta / mean_cpc) * spend**(1 - theta) / noise

so revenue is a POWER LAW in spend with exponent (1 - theta). That is the
classic constant-elasticity ad response curve. theta is the elasticity of
average CPC with respect to spend, and (1 - theta) is the elasticity of revenue
with respect to spend:

    theta = 0     no saturation, revenue linear in spend  (Phase 2/3 behaviour)
    theta = 0.35  a 1% spend increase buys 0.65% more revenue
    theta -> 1    clicks stop responding to spend at all
    theta >= 1    revenue NON-INCREASING in spend -- rejected as pathological

Because each channel's contribution `a_i * ref**theta_i * spend_i**(1-theta_i)`
is strictly concave for theta_i > 0, the sum is strictly concave and the
budget-constrained maximum moves from a corner to a unique INTERIOR point. That
is precisely the property Phase 3 was missing.

THE PARAMETERS: WHAT IS DATA AND WHAT IS ASSUMPTION
===================================================
Per-channel theta is derived as

    theta_i = THETA_MIN + K * ln(impressions_max / impressions_i)

  * `impressions_i` (average daily impressions per channel) is REAL DATA, read
    from `bronze.channel_performance_history`. It fixes the ORDERING and the
    relative SPACING of the thetas.
  * `THETA_MIN = 0.12` and `K = 0.115` are ASSUMPTIONS. They are not fitted to
    anything. They set the level and the scale of the whole curve family, and
    they are the single largest unvalidated input this module introduces. A
    real deployment would estimate theta per channel directly by regressing
    log(average CPC) on log(spend) across periods of differing spend --
    theta IS that regression slope, and it is measurable from data an
    advertiser already has. Nothing in this repo does that.
  * `REFERENCE_SPEND = 100_000.0` is also an ASSUMPTION, and a consequential
    one: see "THE REFERENCE POINT IS NOT NEUTRAL" below.

The economic argument for the ordering: a channel with a smaller impression
footprint runs out of inventory sooner, so its average cost rises faster as you
push volume into it. Paid search is intent-bounded -- only so many people search
a given term on a given day, and no amount of budget creates more of them --
and it has by far the smallest impression footprint in bronze (44.8k/day vs
display's 336.5k/day, a 7.5x gap). Display and programmatic sit on effectively
unbounded remnant/exchange supply and so saturate slowest. The log form encodes
"equal RATIOS of footprint produce equal STEPS in theta", which is the natural
functional form for a quantity spanning an order of magnitude.

CTR is the same ordering read from the other end (paid_search 4.37% vs display
0.39%): a high-CTR channel is one where the addressable audience is already
narrow and well-qualified, which is the same scarcity the impression count
shows. The two agree, so the ranking does not depend on which of the two
columns is used.

MOTIVATED REASONING -- STATED, NOT HIDDEN
=========================================
Giving paid_search the highest theta is exactly what is needed to break the
degenerate frontier, because paid_search's 3.26x expected ROAS (vs 1.97x for
the next best) is what drove all the concentration. That is convenient, and the
convenience deserves to be called out rather than glossed over. Three things
argue it is not merely reverse-engineered:

  1. The ordering is a strict function of a bronze column that was generated in
     Phase 1, long before any of this, and it is monotone in that column with no
     exceptions and no per-channel overrides. Impressions and CTR give the same
     ranking independently.
  2. It is the ordering standard practice would predict anyway. Search is the
     textbook example of a demand-CAPTURE channel with a hard intent ceiling;
     display is the textbook example of near-unlimited cheap supply.
  3. The outcome is not the tidy one a reverse-engineer would have produced.
     With these thetas the best of the 21 candidates on expected revenue is
     `heavy_paid_search` (60%), not a balanced allocation, and the unconstrained
     interior optimum still puts ~49% into paid_search. If the constants had
     been tuned to manufacture a nice frontier, they would have landed
     somewhere flattering to `even_split`, and they did not.

What IS reverse-engineered, and should be treated as such: the MAGNITUDE. K was
not fitted; 0.115 was chosen so that theta spans roughly 0.12 to 0.35 across the
observed footprint range, which is the range that makes saturation visible at
this budget. Measured over the 21 Phase 3 candidates under `normal`, K decides
the answer: at K = 0.05 the ordering claims above all still hold but
`dominant_paid_search` is STILL the expected-revenue maximiser, i.e. the corner
has not moved; it only stops winning somewhere between K = 0.08 and K = 0.115.
Push further and the result inverts -- at K = 0.22 the winner is
`heavy_display`. So: the SIGN and RANK of the effect are grounded in bronze; the
SIZE, and therefore WHICH allocation wins, is an assumption, and every
quantitative conclusion downstream inherits that.

THE REFERENCE POINT IS NOT NEUTRAL
==================================
`reference_spend` is the spend at which the curve passes through the bronze CPC
exactly. Below it the model gives a channel a DISCOUNT (effective CPC < bronze),
above it a penalty. It is not a free normalisation: it decides which allocations
are being extrapolated.

$100,000 is the even-split-per-channel point, so `even_split` sits exactly on
the anchor for every channel and reproduces Phase 2/3 unchanged. But the 21
candidates run from $25,000 (5%) to $400,000 (80%), i.e. 0.25x to 4x reference,
and at 0.25x paid_search's effective CPC drops to $1.29 from bronze's $2.10.
That 39% discount is pure extrapolation -- bronze contains no observation of
what happens to CPC at a quarter of the reference volume, because bronze has no
spend variation to identify the curve from in the first place.

It is also not a small effect, and it points in the direction that flatters
concentration: a thin 5% hedge is modelled as buying unusually cheap clicks, so
the four hedges in an 80/5/5/5/5 split subsidise the penalty on the dominant
channel. Quantified at `dominant_paid_search` (analytic E[revenue], no Jensen):
$1,004,070 as modelled, of which $48,161 (4.8%) is the sub-reference discount on
the four hedge channels. Strip that discount out -- clip the multiplier at 1.0,
penalty above reference but no bonus below -- and it scores $955,909 against
`even_split`'s $965,149, i.e. it goes from BEATING the even split to LOSING to
it. Whether extreme concentration still looks good under this model therefore
rests entirely on an unmeasured extrapolation below the anchor. Anyone reading a
frontier off this model should know that both the penalty and the subsidy are
assumed, not measured.

A DELIBERATE CONSEQUENCE OF THE SPEC: THE CPC DISTRIBUTION TIGHTENS
===================================================================
The spec fixes that ONLY the mean CPC responds to spend; `std_cpc` keeps its
bronze value. `engine.fit_lognormal_moments` derives sigma from std/mean, so a
rising mean against a fixed std means a FALLING coefficient of variation: the
CPC distribution gets relatively tighter the more you spend, and the Jensen
correction exp(sigma^2) on E[1/CPC] shrinks with it. Over the 0.25x-4x range the
21 candidates actually use, paid_search's CPC CV moves from 0.1264 to 0.0476 (a
2.65x swing) and its Jensen bonus from +1.598% to +0.227%. Display, the least
saturating channel, moves far less: CV 0.0917 to 0.0657, a 1.39x swing.

That is a second, unintended channel through which saturation penalises large
positions -- small compared with the mean effect, but real and not part of the
stated economic story. `saturate_std_cpc=True` on the engine turns on the
CV-PRESERVING variant instead, scaling std by the same multiplier, which is what
Phase 3's scenario multipliers do and is arguably the more defensible default.
It is a flag, not the default, because the default implements the spec as
written.
"""

from __future__ import annotations

import math

import numpy as np


# --- Fixed constants ---------------------------------------------------------

# Spend at which effective CPC equals the bronze mean exactly. ASSUMPTION.
# Chosen as $500,000 / 5 channels, the even-split point Phase 2 and Phase 3
# validated against, so the anchor cell is untouched by this module.
REFERENCE_SPEND: float = 100_000.0

# ASSUMPTIONS. Not fitted to anything -- see the module docstring.
THETA_MIN: float = 0.12   # elasticity of the largest-footprint channel
K: float = 0.115          # theta added per natural-log unit of footprint deficit

# Derived thetas are rounded before being frozen as project constants, so the
# published numbers are legible and cannot drift in the last bits if a bronze
# aggregate is ever recomputed with a different summation order. At 3 dp the
# rounding perturbs theta by at most 5e-4, which at 4x reference spend moves the
# CPC multiplier by 4**5e-4 = 1.0007 -- immaterial next to K being invented.
THETA_ROUNDING_DP: int = 3

# Snapshot of AVG(impressions) per channel from
# `bronze.channel_performance_history` (730 days, 2024-01-01..2025-12-30),
# read live on 2026-08-13. Frozen here so `DEFAULT_THETA` is available on a
# Spark executor without a warehouse round trip. `check_impressions_snapshot`
# re-validates it against live bronze.
BRONZE_AVG_IMPRESSIONS_PER_DAY: dict[str, float] = {
    "display": 336471.37534246576,
    "paid_search": 44791.3698630137,
    "paid_social": 134632.35205479452,
    "programmatic": 224394.48356164384,
    "video": 100898.83561643836,
}

# theta must stay in [0, 1). theta = 0 is "no saturation"; theta >= 1 makes
# clicks flat or DECREASING in spend, which is not diminishing returns, it is a
# broken model. Rejected rather than clamped.
THETA_MAX_EXCLUSIVE: float = 1.0


# --- Derivation --------------------------------------------------------------

def derive_theta(
    impressions: dict[str, float],
    theta_min: float = THETA_MIN,
    k: float = K,
    round_dp: int | None = THETA_ROUNDING_DP,
) -> dict[str, float]:
    """theta per channel from average daily impressions.

        theta_i = theta_min + k * ln(impressions_max / impressions_i)

    The largest-footprint channel gets exactly `theta_min`; every other channel
    is pushed up by the log of how much smaller its footprint is. Monotone
    decreasing in impressions by construction, so the ordering is a property of
    the data and not of the two constants -- those only set level and scale.

    `round_dp=None` returns the unrounded values (used to show the derivation).
    """
    if not impressions:
        raise ValueError("no channels supplied")
    values = list(impressions.values())
    if any((not math.isfinite(v)) or v <= 0 for v in values):
        raise ValueError(f"impressions must all be finite and > 0, got {impressions}")
    if not math.isfinite(theta_min) or theta_min < 0:
        raise ValueError(f"theta_min must be finite and >= 0, got {theta_min}")
    if not math.isfinite(k) or k < 0:
        raise ValueError(f"k must be finite and >= 0, got {k}")

    i_max = max(values)
    out: dict[str, float] = {}
    for channel, impr in impressions.items():
        theta = theta_min + k * math.log(i_max / impr)
        out[channel] = round(theta, round_dp) if round_dp is not None else theta
    validate_theta_mapping(out)
    return out


def validate_theta_mapping(theta: dict[str, float]) -> None:
    """Raise if any theta is outside [0, 1). See THETA_MAX_EXCLUSIVE."""
    for channel, value in theta.items():
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"theta for {channel!r} is not finite: {value!r}")
        if v < 0.0:
            raise ValueError(
                f"theta for {channel!r} is {v}; negative theta means INCREASING returns "
                f"to spend (average CPC falling as you buy more), which this model does "
                f"not represent"
            )
        if v >= THETA_MAX_EXCLUSIVE:
            raise ValueError(
                f"theta for {channel!r} is {v}; at theta >= 1 clicks are flat or "
                f"DECREASING in spend, so revenue is non-increasing in budget. That is "
                f"not diminishing returns, it is a broken response curve"
            )


# The project default, derived from the frozen bronze snapshot so the constants
# and the formula can never disagree with each other.
DEFAULT_THETA: dict[str, float] = derive_theta(BRONZE_AVG_IMPRESSIONS_PER_DAY)


def theta_vector(channels: list[str], theta: dict[str, float] | None = None) -> np.ndarray:
    """(k,) array aligned to `channels`. Raises on a missing channel."""
    table = DEFAULT_THETA if theta is None else theta
    missing = [c for c in channels if c not in table]
    if missing:
        raise ValueError(f"no theta for channels {missing}; have {sorted(table)}")
    return np.array([float(table[c]) for c in channels], dtype=float)


def check_impressions_snapshot(
    live: dict[str, float], rtol: float = 1e-9
) -> dict[str, float]:
    """Compare live bronze impressions against the frozen snapshot.

    Returns channel -> relative difference. Raises on a channel-set mismatch.
    Exists so the snapshot that produces `DEFAULT_THETA` is falsifiable rather
    than merely asserted.
    """
    if set(live) != set(BRONZE_AVG_IMPRESSIONS_PER_DAY):
        raise ValueError(
            f"channel set mismatch: live {sorted(live)} vs snapshot "
            f"{sorted(BRONZE_AVG_IMPRESSIONS_PER_DAY)}"
        )
    out = {}
    for c, v in live.items():
        ref = BRONZE_AVG_IMPRESSIONS_PER_DAY[c]
        out[c] = abs(float(v) - ref) / ref
    worst = max(out.values())
    if worst > rtol:
        raise AssertionError(
            f"frozen impressions snapshot is stale: worst relative difference {worst:.3e} "
            f"exceeds {rtol:.3e}. Re-derive DEFAULT_THETA from live bronze. Detail: {out}"
        )
    return out


# --- The curve ---------------------------------------------------------------

def saturation_multiplier(
    spend: np.ndarray,
    theta: np.ndarray,
    reference_spend: float = REFERENCE_SPEND,
) -> np.ndarray:
    """(spend / reference_spend) ** theta, elementwise, with the edge cases nailed down.

    Two properties this function must have, both load-bearing:

      * EXACTLY 1.0 at spend == reference_spend. `1.0 ** theta` is exactly 1.0
        in IEEE-754 for any finite theta, and `x * 1.0 == x` bitwise, so the
        even-split anchor reproduces Phase 2/3 to the bit rather than to a
        tolerance.
      * NO SILENT INFINITIES AT ZERO SPEND. For theta > 0, (0/ref)**theta == 0,
        which would make the effective CPC zero and clicks infinite. A channel
        with no budget should contribute no revenue, not undefined revenue, so
        zero or negative spend raises here instead of poisoning the arithmetic
        downstream. (None of the 21 Phase 3 candidates has a zero channel -- the
        floor is 5% = $25,000 -- so this is a guard, not a workaround.)

    Channels with theta == 0 are short-circuited to a multiplier of exactly 1.0
    and are exempt from the positive-spend requirement, since 0**0 is a
    convention argument the caller should not have to care about and an
    unsaturated channel at zero spend is perfectly well defined.
    """
    spend = np.asarray(spend, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if spend.shape != theta.shape:
        raise ValueError(f"spend shape {spend.shape} != theta shape {theta.shape}")
    if not math.isfinite(reference_spend) or reference_spend <= 0:
        raise ValueError(f"reference_spend must be finite and > 0, got {reference_spend}")
    if not np.all(np.isfinite(spend)):
        raise ValueError(f"spend must be finite, got {spend}")
    validate_theta_mapping({f"index {i}": t for i, t in enumerate(theta)})

    mult = np.ones_like(spend)
    active = theta != 0.0
    if np.any(active):
        bad = active & (spend <= 0.0)
        if np.any(bad):
            raise ValueError(
                f"channels at index {np.flatnonzero(bad).tolist()} have spend "
                f"{spend[bad].tolist()} with theta {theta[bad].tolist()}; saturation is "
                f"undefined at non-positive spend ((0/ref)**theta == 0 would give an "
                f"effective CPC of zero and infinite clicks). Either drop the channel "
                f"from the allocation or set its theta to 0."
            )
        mult[active] = (spend[active] / reference_spend) ** theta[active]
    return mult


def effective_mean_cpc(
    mean_cpc: np.ndarray,
    spend: np.ndarray,
    theta: np.ndarray,
    reference_spend: float = REFERENCE_SPEND,
) -> np.ndarray:
    """mean_cpc * (spend / reference_spend) ** theta."""
    return np.asarray(mean_cpc, dtype=float) * saturation_multiplier(
        spend, theta, reference_spend
    )


# --- Curve inspection (no simulation involved) -------------------------------

def curve_table(
    channels: list[str],
    mean_cpc: np.ndarray,
    spend_multiples: list[float],
    theta: dict[str, float] | None = None,
    reference_spend: float = REFERENCE_SPEND,
) -> list[dict]:
    """Effective CPC and clicks-per-dollar vs spend, straight from the algebra.

    Deterministic and simulation-free, so a wrong curve shows up here rather
    than being diagnosed through Monte Carlo noise.

      average clicks per dollar  = clicks/spend  = 1 / effective_mean_cpc
      marginal clicks per dollar = d(clicks)/d(spend)
                                 = (1 - theta) / effective_mean_cpc

    The marginal is exactly (1 - theta) times the average -- a fixed ratio at
    every spend level, which is the signature of a constant-elasticity curve
    and the cleanest single check that the implementation matches the intent.
    """
    th = theta_vector(channels, theta)
    mean_cpc = np.asarray(mean_cpc, dtype=float)
    rows: list[dict] = []
    for m in spend_multiples:
        spend = np.full(len(channels), m * reference_spend, dtype=float)
        eff = effective_mean_cpc(mean_cpc, spend, th, reference_spend)
        for i, channel in enumerate(channels):
            rows.append({
                "channel_id": channel,
                "spend_multiple": float(m),
                "spend": float(spend[i]),
                "theta": float(th[i]),
                "bronze_mean_cpc": float(mean_cpc[i]),
                "effective_mean_cpc": float(eff[i]),
                "cpc_multiplier": float(eff[i] / mean_cpc[i]),
                "avg_clicks_per_dollar": float(1.0 / eff[i]),
                "marginal_clicks_per_dollar": float((1.0 - th[i]) / eff[i]),
            })
    return rows


def _print_derivation() -> None:  # pragma: no cover -- documentation helper
    raw = derive_theta(BRONZE_AVG_IMPRESSIONS_PER_DAY, round_dp=None)
    i_max = max(BRONZE_AVG_IMPRESSIONS_PER_DAY.values())
    print(f"theta_i = {THETA_MIN} + {K} * ln(impressions_max / impressions_i)")
    print(f"impressions_max = {i_max:,.4f} (display)   [ASSUMPTIONS: theta_min, k]")
    print(f"\n{'channel':<14}{'avg impr/day':>16}{'ln(imax/i)':>13}{'theta raw':>12}{'theta':>9}")
    for c in sorted(BRONZE_AVG_IMPRESSIONS_PER_DAY, key=lambda c: -BRONZE_AVG_IMPRESSIONS_PER_DAY[c]):
        impr = BRONZE_AVG_IMPRESSIONS_PER_DAY[c]
        print(f"{c:<14}{impr:>16,.1f}{math.log(i_max / impr):>13.6f}"
              f"{raw[c]:>12.6f}{DEFAULT_THETA[c]:>9.3f}")


if __name__ == "__main__":  # pragma: no cover
    _print_derivation()
