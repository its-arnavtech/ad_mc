"""
Phase 3 candidate budget allocations.

WHAT THIS IS -- AND WHAT IT IS NOT
==================================
This is a fixed, ILLUSTRATIVE TEST SET of budget allocations, hand-designed to
span a range of concentration profiles so the Phase 3 batch has something
interesting to distribute over and so the silver layer has real variation in it.

It is explicitly NOT Phase 4's optimization. Nothing here searches, optimizes,
or claims to contain the efficient allocation. Phase 4 will run an actual sweep
over the simplex (or a constrained optimizer) and trace a frontier; these 21
points are a probe, not a solution. If one of these happens to look good in the
Phase 3 output, that is a fact about this arbitrary grid, not a recommendation.

THE GRID
========
For k channels the set is built generically (nothing is hardcoded to 5):

    even_split          1/k to every channel                    (the anchor)
    mild_<ch>           35% to <ch>, 65%/(k-1) to the rest       k of them
    heavy_<ch>          60% to <ch>, 40%/(k-1) to the rest       k of them
    dominant_<ch>       80% to <ch>, 20%/(k-1) to the rest       k of them
    tilt_<a>_<b>        30% each to a neighbouring pair,
                        40%/(k-2) to the rest                    k of them

With k = 5 that is 1 + 5 + 5 + 5 + 5 = 21 candidates.

The first four families form a deliberate CONCENTRATION LADDER at every channel
-- same channel, four increasing levels of concentration -- which is what makes
the output readable: you can hold the channel fixed and watch risk and return
move with concentration alone. Herfindahl index (sum of squared weights, 0.20 =
perfectly even at k=5, 1.00 = everything in one channel):

    even_split   0.2000
    mild_*       0.2281
    tilt_*       0.2333
    heavy_*      0.4000
    dominant_*   0.6500

The tilt family is a ring over adjacent channel pairs -- (c0,c1), (c1,c2), ...,
(c_{k-1},c0) -- so every channel appears in exactly two tilts. A full pairwise
set would be k*(k-1)/2 = 10 for k=5, which is more points than this probe needs;
the ring keeps the family balanced (each channel weighted equally often) at k
points instead of 10.

EXACT DOLLARS
=============
Weights are fractions, but dollars are quantized to WHOLE DOLLARS as Python
ints before being converted to float. Integers below 2^53 are exactly
representable in float64 and sum exactly, so `sum(allocation.values()) ==
total_budget` holds as strict float equality, not "within a tolerance". Cents
would not give that guarantee -- 133333.33 is not representable in binary, and
five such terms drift by ~1e-10.

Rounding leaves a residual of at most (k-1) dollars. It is absorbed into the
channel with the LARGEST nominal weight, ties broken by position in the
`channels` list (earliest wins). Largest-weight is chosen so the relative
distortion is smallest and so the adjustment can never drive a small channel to
zero or negative; the tie-break makes it deterministic. Example: tilt_* rounds
40%/3 of $500,000 to $66,667 three times ($200,001), so the residual is -$1 and
one of the two 30% channels receives $149,999 instead of $150,000. That is a
0.0007% perturbation of one line item and is documented rather than hidden.

`even_split` is produced by the same generic path, and is then asserted to be
BITWISE IDENTICAL to `config.even_split_allocation` whenever the budget divides
into whole dollars evenly (it does for this project: $500,000 / 5 = $100,000).
That assertion is the point -- even_split is the anchor cell that must reproduce
Phase 2 exactly, so its dollars must not merely be close.

MODELLING NOTE THAT MATTERS DOWNSTREAM
======================================
The Phase 2 engine has NO saturation / diminishing-returns curve: within a
channel, clicks = spend / CPC, so revenue is exactly LINEAR in spend. Expected
total revenue is therefore a linear function of the weight vector, which means
the highest-expected-revenue allocation in any candidate set is always 100% in
whichever single channel has the best expected ROAS, and concentration only
ever costs you variance, never mean. Any "optimal allocation" conclusion drawn
from this grid (or from Phase 4) is a statement about RISK ONLY until a spend
response curve is added to the model.
"""

from __future__ import annotations

import math

try:  # package-style import (simulation.allocations)
    from .config import TOTAL_BUDGET, even_split_allocation
except ImportError:  # flat import, which is how the rest of this repo runs
    from config import TOTAL_BUDGET, even_split_allocation


EVEN_SPLIT_ID = "even_split"

# Dominant-channel weight for each single-channel family. The remainder is
# split evenly across the other k-1 channels.
FAMILY_DOMINANT_WEIGHT: dict[str, float] = {
    "mild": 0.35,
    "heavy": 0.60,
    "dominant": 0.80,
}
# Weight given to EACH member of a two-channel tilt; the rest share what's left.
TILT_PAIR_WEIGHT = 0.30

FAMILY_DESCRIPTIONS: dict[str, str] = {
    "even": "Equal weight to every channel. The Phase 2 baseline and the validation anchor.",
    "mild": "One channel mildly favoured (35%), the rest equal. Smallest departure from even.",
    "heavy": "One channel clearly dominant (60%), the rest equal.",
    "dominant": "One channel overwhelmingly dominant (80%), the rest a thin hedge (5% each).",
    "tilt": "Two adjacent channels co-favoured (30% each), the rest equal.",
}


def _family_of(allocation_id: str) -> str:
    if allocation_id == EVEN_SPLIT_ID:
        return "even"
    return allocation_id.split("_", 1)[0]


def build_allocation_weights(channels: list[str]) -> dict[str, dict[str, float]]:
    """allocation_id -> {channel_id: weight}, weights summing to 1.0 in exact fractions.

    Deterministic: the same `channels` list in the same order always yields the
    same ids, the same ordering, and the same weights. Nothing random, nothing
    dependent on dict iteration order beyond the caller's channel order.
    """
    if not channels:
        raise ValueError("no channels to allocate across")
    if len(set(channels)) != len(channels):
        raise ValueError(f"duplicate channel ids: {channels}")
    k = len(channels)
    if k < 2:
        raise ValueError(f"need at least 2 channels to build a candidate set, got {k}")

    out: dict[str, dict[str, float]] = {}

    # 1. the anchor
    out[EVEN_SPLIT_ID] = {c: 1.0 / k for c in channels}

    # 2. single-channel concentration ladder
    for family, w_dom in FAMILY_DOMINANT_WEIGHT.items():
        w_rest = (1.0 - w_dom) / (k - 1)
        for target in channels:
            out[f"{family}_{target}"] = {
                c: (w_dom if c == target else w_rest) for c in channels
            }

    # 3. two-channel tilts around a ring of adjacent pairs
    if k >= 3:
        w_rest = (1.0 - 2.0 * TILT_PAIR_WEIGHT) / (k - 2)
        for i in range(k):
            a, b = channels[i], channels[(i + 1) % k]
            out[f"tilt_{a}_{b}"] = {
                c: (TILT_PAIR_WEIGHT if c in (a, b) else w_rest) for c in channels
            }

    return out


def _quantize_to_whole_dollars(
    weights: dict[str, float], channels: list[str], total_budget: float
) -> dict[str, float]:
    """Fractions -> whole-dollar floats that sum to `total_budget` exactly.

    Residual from rounding goes to the largest-weight channel (earliest in
    `channels` on a tie). See the module docstring for why.
    """
    budget_dollars = round(total_budget)
    if budget_dollars != total_budget:
        raise ValueError(
            f"total_budget {total_budget!r} is not a whole number of dollars; the exact-sum "
            f"guarantee in this module depends on integer-dollar quantization"
        )

    dollars = [round(weights[c] * total_budget) for c in channels]
    residual = budget_dollars - sum(dollars)

    absorb = max(range(len(channels)), key=lambda i: (weights[channels[i]], -i))
    dollars[absorb] += residual

    if min(dollars) <= 0:
        raise ValueError(
            f"allocation produced a non-positive channel spend {dict(zip(channels, dollars))}; "
            f"every channel must receive strictly positive budget"
        )
    return {c: float(d) for c, d in zip(channels, dollars)}


def build_allocation_candidates(
    channels: list[str], total_budget: float = TOTAL_BUDGET
) -> dict[str, dict[str, float]]:
    """allocation_id -> {channel_id: spend_dollars}.

    Deterministic and exact: every allocation's dollars sum to `total_budget`
    under strict float equality (asserted), every channel receives strictly
    positive spend, and `even_split` is bitwise identical to
    `config.even_split_allocation` whenever the budget divides evenly into
    whole dollars.
    """
    if total_budget <= 0:
        raise ValueError(f"total_budget must be > 0, got {total_budget}")

    weights = build_allocation_weights(channels)
    out = {
        alloc_id: _quantize_to_whole_dollars(w, channels, total_budget)
        for alloc_id, w in weights.items()
    }

    # --- invariants, checked every call rather than trusted ---
    for alloc_id, alloc in out.items():
        total = math.fsum(alloc[c] for c in channels)
        if total != total_budget:
            raise AssertionError(
                f"allocation {alloc_id!r} sums to {total!r}, expected exactly {total_budget!r}"
            )
        if any(v <= 0 for v in alloc.values()):
            raise AssertionError(f"allocation {alloc_id!r} has a non-positive channel: {alloc}")

    # --- the anchor must match Phase 2 bitwise ---
    per_channel = total_budget / len(channels)
    if per_channel == round(per_channel):  # budget divides into whole dollars
        reference = even_split_allocation(channels, total_budget)
        if out[EVEN_SPLIT_ID] != reference:
            raise AssertionError(
                f"{EVEN_SPLIT_ID} drifted from config.even_split_allocation: "
                f"{out[EVEN_SPLIT_ID]} vs {reference}"
            )

    return out


def allocation_percentages(
    allocation: dict[str, float], total_budget: float
) -> dict[str, float]:
    """Realized share per channel, computed from the QUANTIZED dollars.

    Deliberately derived from dollars rather than from the nominal weight, so
    spend_pct * total_budget reproduces spend exactly and the table cannot
    disagree with itself. Shares therefore reflect the whole-dollar rounding
    (e.g. 0.299998 rather than 0.30 on the tilt channel that absorbed the -$1).
    """
    return {c: v / total_budget for c, v in allocation.items()}


def concentration_hhi(allocation: dict[str, float], total_budget: float) -> float:
    """Herfindahl-Hirschman index of the allocation: sum of squared shares.

    1/k for a perfectly even split (0.20 at k=5), 1.0 for everything in one
    channel. Carried into the silver table because it is the natural x-axis for
    "what does concentration cost in risk".
    """
    return math.fsum((v / total_budget) ** 2 for v in allocation.values())


def allocation_rows(
    channels: list[str], total_budget: float = TOTAL_BUDGET
) -> list[dict]:
    """Long-form rows for `silver.allocation_candidates`, one per (allocation, channel).

    Columns, EXACTLY the five the Phase 3 spec fixes for the table:
    allocation_id, channel_id, spend_pct, spend_dollars, total_budget.

    Deliberately NOT persisted here, though `allocation_attributes()` still
    exposes them to Python callers: allocation_family, allocation_description,
    n_channels, concentration_hhi. Those are allocation-level attributes that
    would have to be denormalised onto every channel row, and all of them are
    recoverable from what IS stored -- HHI is a function of spend_pct, family
    is a prefix of allocation_id, n_channels is a row count. Widening a
    spec'd table with derivable columns is a schema change, not a convenience.

    Ordering is stable: allocations in `build_allocation_candidates` order,
    channels in the caller's `channels` order.
    """
    candidates = build_allocation_candidates(channels, total_budget)
    rows: list[dict] = []
    for alloc_id, alloc in candidates.items():
        pct = allocation_percentages(alloc, total_budget)
        for c in channels:
            rows.append({
                "allocation_id": alloc_id,
                "channel_id": c,
                "spend_pct": pct[c],
                "spend_dollars": alloc[c],
                "total_budget": float(total_budget),
            })
    return rows


def allocation_attributes(
    channels: list[str], total_budget: float = TOTAL_BUDGET
) -> list[dict]:
    """One row per allocation with the descriptive attributes, for Phase 4/6.

    Kept out of `silver.allocation_candidates` on purpose -- see
    `allocation_rows`. Nothing in Phase 3 writes this; it exists so the
    concentration ladder is available to analysis without recomputing it.
    """
    candidates = build_allocation_candidates(channels, total_budget)
    out: list[dict] = []
    for alloc_id, alloc in candidates.items():
        family = _family_of(alloc_id)
        out.append({
            "allocation_id": alloc_id,
            "allocation_family": family,
            "allocation_description": FAMILY_DESCRIPTIONS[family],
            "n_channels": len(channels),
            "concentration_hhi": concentration_hhi(alloc, total_budget),
        })
    return out


def allocation_ids(channels: list[str]) -> list[str]:
    """Just the ids, in the canonical order. Cheap enough to call for planning."""
    return list(build_allocation_weights(channels))
