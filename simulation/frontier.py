"""
Phase 4: candidate generation, per-cell aggregation, and efficient-frontier
computation for the budget-allocation sweep.

PURE AND SPARK-FREE, ON PURPOSE. numpy / pandas / scipy only, no I/O, no
Databricks, no SparkSession. The data-engineering layer distributes
`summarize_cell` (or `summarize_cell_row`) inside a UDF and writes the result;
nothing in this module knows that happened. Everything is deterministic given
the master seed, so a frontier can be reproduced exactly.

WHAT THIS MODULE IS FOR
=======================
Phase 3 evaluated 21 hand-designed allocations. That is a probe, not an
optimization. This module searches the 5-channel budget simplex properly and
computes which allocations are non-dominated on risk and return -- the efficient
frontier -- per scenario.

It only means anything because of the saturation curve added in the Phase 4
prerequisite. Without it expected revenue is LINEAR in the weight vector, its
maximum is always a simplex corner, and no amount of searching produces a
tradeoff. With it each channel's contribution is strictly concave and the
optimum is interior. See `saturation.py`.


THE OUTPUT GRAIN IS ONE ROW PER (ALLOCATION, SCENARIO)
======================================================
The full sweep is ~700 candidates x 4 scenarios x 10,000 paths = ~28 MILLION
paths. Phase 3's path-level grain would mean 28 million rows that nothing reads:
every downstream consumer -- the frontier, the recommendations, the charts --
needs only the moments and the tail statistics. So the tail metrics are computed
INSIDE the cell, while the path vector is in memory, and the vector is then
dropped. `summarize_cell` returns exactly one dict.

The cost of that choice, stated plainly: the path distributions are not
recoverable afterwards. Re-deriving any statistic that was not aggregated here
(a different VaR level, a histogram, a drawdown measure) means re-running the
cell. That is cheap and exactly reproducible -- the seed is in the output row --
but it is a re-run, not a query.


CANDIDATE GENERATION: TWO STAGES, AND WHY
=========================================
STAGE 1 -- broad coverage. Three sources, deliberately different in character:

  * The 21 Phase 3 candidates, verbatim and with their original ids. Continuity:
    every Phase 3 number stays comparable, and `even_split` remains in the set as
    the anchor that reproduces Phase 2 bitwise.
  * SIMPLEX LATTICES of degree 3 and 4 (Scheffe {5,3} and {5,4} designs): every
    allocation whose weights are multiples of 1/3, and every one whose weights
    are multiples of 1/4. That is 35 + 70 points, 100 of them distinct, and it is
    a DETERMINISTIC, exhaustive-at-its-resolution skeleton -- it contains all 5
    pure vertices, all 10 edge midpoints, all face centroids, and every
    "two channels only" and "three channels only" combination. Random sampling
    alone would hit the boundary of the simplex only by accident; the lattice
    guarantees it, which is what makes the zero-spend fix actually get exercised.
  * DIRICHLET random interior points at three concentrations. alpha = 1 is
    uniform over the simplex (unbiased fill); alpha = 0.3 concentrates near the
    faces and corners, producing near-zero and zero channels; alpha = 4
    concentrates near the centroid, where the interior optimum is expected to
    live. Mixing the three avoids the classic mistake of sampling only alpha = 1
    and then having almost no resolution anywhere near the optimum -- in 5
    dimensions, uniform points are mostly mediocre.

STAGE 2 -- local refinement around the stage-1 Pareto set. Two moves:

  * PERTURBATION clouds: Dirichlet(c * w + a0) centred on each stage-1 frontier
    point, at two concentrations (tight and loose), so the frontier is resampled
    at higher resolution exactly where it lives. `a0 > 0` lets a channel that is
    currently at zero come back, so refinement can leave the face it started on.
  * CONVEX BLENDS between pairs of stage-1 frontier points. This is the move that
    matters most here. The prerequisite measured a mean-VaR Pareto set of 2
    points out of 21 -- a line SEGMENT, not a curve -- and the segment between
    two frontier points is precisely the region no random sampler concentrates
    on. Blending fills it directly. If the frontier is genuinely thin, blends
    will show that (they will be dominated); if it is thin only because nothing
    sampled between the endpoints, blends will populate it.

Stage 2 is seeded from the stage-1 RESULT, so it is reproducible only together
with stage 1. That is a real dependency and the run should be treated as one
unit.


EXACT DOLLARS, AND ZERO IS ALLOWED
==================================
Weights are quantized to WHOLE DOLLARS, the same convention (and the same
residual-absorption rule) as `allocations.py`, so `sum(spend) == total_budget`
holds as strict float equality rather than within a tolerance.
`quantize_to_whole_dollars` here differs from the Phase 3 version in exactly one
way: it PERMITS a channel at $0 and forbids only negatives. Dropping a channel is
a legitimate allocation and the optimizer has to be able to reach the boundary.
`assert_matches_phase3_quantizer` checks the two agree on the 21 Phase 3
candidates, so "same convention" is verified rather than claimed.

A consequence worth knowing: a Dirichlet weight below $0.50 rounds to $0, so
alpha = 0.3 produces genuinely idle channels rather than merely small ones. That
is intended -- it is what puts the zero-spend path under load.


WHAT IS ASSUMED HERE, ON TOP OF EVERYTHING THE MODEL ALREADY ASSUMES
====================================================================
This module adds no new economics, but it does add search choices, and they
shape the answer:
  * The candidate set is finite. A "frontier" from it is the frontier OF THAT
    SET, not of the simplex. `analytic_interior_optimum` gives the continuous
    expected-revenue optimum as an independent reference point, so the discrete
    argmax can be checked against something that did not come out of the search.
  * Objectives are computed from ONE 10,000-path draw per (allocation, scenario)
    under common random numbers. Differences between allocations are precise;
    the LEVEL of the whole frontier carries the ~$1,700 error of a single draw
    and moves with the master seed. See `sweep_seeding.py`.
  * Pareto dominance is exact float comparison. Two allocations differing by
    $0.01 in mean are "different". With CRN that is far less abusive than it
    sounds -- the noise in a DIFFERENCE is small -- but it is still a discrete
    verdict on a continuous quantity, and `frontier_robustness` re-runs the
    dominance test on an independent replicate to show how much the membership
    actually moves.
  * Everything inherits the saturation curve's k and reference_spend, both
    ASSUMPTIONS with no bronze support. The prerequisite's caveats about them
    apply in full to every allocation named here.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # package-style import (simulation.frontier / ad_mc_sim.frontier)
    from . import cell, engine
    from .allocations import build_allocation_candidates
    from .config import TOTAL_BUDGET
    from .saturation import REFERENCE_SPEND, theta_vector
    from .scenarios import SCENARIOS, Scenario
    from .seeding import MASTER_SEED, stable_hash_int
    from .sweep_seeding import crn_seed
except ImportError:  # flat import, which is how the rest of this repo runs
    import cell
    import engine
    from allocations import build_allocation_candidates
    from config import TOTAL_BUDGET
    from saturation import REFERENCE_SPEND, theta_vector
    from scenarios import SCENARIOS, Scenario
    from seeding import MASTER_SEED, stable_hash_int
    from sweep_seeding import crn_seed


# =============================================================================
# Output contract
# =============================================================================

SWEEP_OUTPUT_COLUMNS: tuple[str, ...] = (
    "allocation_id",
    "scenario_id",
    "n_paths",
    "seed",
    "total_spend",
    "mean_revenue",
    "std_revenue",
    "se_mean_revenue",
    "min_revenue",
    "max_revenue",
    "median_revenue",
    "var_95",
    "cvar_95",
    "expected_roas",
    "prob_below_breakeven",
)

# Paste-able for applyInPandas / mapInPandas. `seed` is a 64-bit unsigned value
# that does not fit Spark's signed BIGINT in general, so it is carried as a
# STRING -- silently wrapping it to a negative BIGINT would make the row
# unusable for reproducing the cell, which is the only reason it is stored.
SWEEP_OUTPUT_SPARK_SCHEMA_DDL: str = (
    "allocation_id STRING, "
    "scenario_id STRING, "
    "n_paths INT, "
    "seed STRING, "
    "total_spend DOUBLE, "
    "mean_revenue DOUBLE, "
    "std_revenue DOUBLE, "
    "se_mean_revenue DOUBLE, "
    "min_revenue DOUBLE, "
    "max_revenue DOUBLE, "
    "median_revenue DOUBLE, "
    "var_95 DOUBLE, "
    "cvar_95 DOUBLE, "
    "expected_roas DOUBLE, "
    "prob_below_breakeven DOUBLE"
)

VAR_CONFIDENCE: float = 0.95


# =============================================================================
# Inputs
# =============================================================================

@dataclass(frozen=True)
class SweepInputs:
    """Everything a cell needs that is NOT the allocation or the scenario.

    Built once on the driver from `data_access` (this module does no I/O) and
    broadcast. Frozen so a UDF cannot mutate a broadcast copy and desynchronise
    executors from each other.
    """
    channels: tuple[str, ...]
    mean_cvr: np.ndarray
    std_cvr: np.ndarray
    mean_cpc: np.ndarray
    std_cpc: np.ndarray
    mean_rpc: np.ndarray
    std_rpc: np.ndarray
    correlation: np.ndarray
    theta: np.ndarray | None
    reference_spend: float = REFERENCE_SPEND
    saturate_std_cpc: bool = False
    total_budget: float = TOTAL_BUDGET
    n_paths: int = 10_000

    def __post_init__(self) -> None:
        k = len(self.channels)
        if k < 2:
            raise ValueError(f"need at least 2 channels, got {self.channels}")
        for name in ("mean_cvr", "std_cvr", "mean_cpc", "std_cpc", "mean_rpc", "std_rpc"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (k,):
                raise ValueError(f"{name} has shape {arr.shape}, expected {(k,)}")
        if np.asarray(self.correlation).shape != (k, k):
            raise ValueError(f"correlation has shape {np.shape(self.correlation)}, expected {(k, k)}")
        if self.theta is not None and np.asarray(self.theta).shape != (k,):
            raise ValueError(f"theta has shape {np.shape(self.theta)}, expected {(k,)}")
        if self.n_paths <= 0:
            raise ValueError(f"n_paths must be > 0, got {self.n_paths}")

    @property
    def k(self) -> int:
        return len(self.channels)

    def with_n_paths(self, n_paths: int) -> "SweepInputs":
        return SweepInputs(**{**self.__dict__, "n_paths": n_paths})


# =============================================================================
# Candidates
# =============================================================================

@dataclass(frozen=True)
class Candidate:
    """One budget allocation, in exact whole dollars aligned to `channels`."""
    allocation_id: str
    family: str
    stage: int
    spend: tuple[float, ...]
    parents: tuple[str, ...] = ()

    def as_dict(self, channels: Sequence[str]) -> dict[str, float]:
        return {c: s for c, s in zip(channels, self.spend)}

    def shares(self, total_budget: float) -> tuple[float, ...]:
        return tuple(s / total_budget for s in self.spend)

    def hhi(self, total_budget: float) -> float:
        return math.fsum((s / total_budget) ** 2 for s in self.spend)

    @property
    def n_funded(self) -> int:
        return sum(1 for s in self.spend if s > 0.0)


def quantize_to_whole_dollars(
    weights: Sequence[float], total_budget: float = TOTAL_BUDGET
) -> tuple[float, ...]:
    """Weights -> whole-dollar floats summing to `total_budget` EXACTLY.

    Same convention as `allocations._quantize_to_whole_dollars` (round each
    channel, absorb the residual into the largest-weight channel, earliest index
    on a tie) with ONE deliberate difference: $0 for a channel is allowed. Only
    negatives raise. `assert_matches_phase3_quantizer` verifies the two agree
    wherever both are defined.

    Integers below 2^53 are exact in float64 and sum exactly, so the returned
    tuple sums to `total_budget` under `==`, not under `isclose`.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError(f"weights must be a non-empty 1-D sequence, got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"weights must be finite, got {w}")
    if np.any(w < 0.0):
        raise ValueError(f"weights must be non-negative, got {w}")
    total_w = float(w.sum())
    if total_w <= 0.0:
        raise ValueError("weights sum to zero; there is no allocation to build")

    budget_dollars = round(total_budget)
    if budget_dollars != total_budget:
        raise ValueError(
            f"total_budget {total_budget!r} is not a whole number of dollars; the exact-sum "
            f"guarantee depends on integer-dollar quantization"
        )
    w = w / total_w

    dollars = [round(float(wi) * total_budget) for wi in w]
    residual = budget_dollars - sum(dollars)
    absorb = max(range(len(w)), key=lambda i: (w[i], -i))
    dollars[absorb] += residual

    if min(dollars) < 0:
        raise ValueError(
            f"quantization produced negative spend {dollars} from weights {w.tolist()}"
        )
    out = tuple(float(d) for d in dollars)
    if math.fsum(out) != float(total_budget):
        raise AssertionError(f"quantized spend {out} does not sum to {total_budget}")
    return out


def simplex_lattice(k: int, degree: int) -> list[tuple[float, ...]]:
    """Scheffe {k, degree} simplex-lattice weights: all multiples of 1/degree.

    Every composition of `degree` identical units into `k` ordered parts, i.e.
    C(degree + k - 1, k - 1) points. Deterministic, in lexicographic order.
    Includes the pure vertices (all `degree` units in one part), so this is what
    drives the search onto the boundary of the simplex.
    """
    if k < 2 or degree < 1:
        raise ValueError(f"need k >= 2 and degree >= 1, got k={k}, degree={degree}")
    out: list[tuple[float, ...]] = []
    for cut in itertools.combinations(range(degree + k - 1), k - 1):
        counts, prev = [], -1
        for c in cut:
            counts.append(c - prev - 1)
            prev = c
        counts.append(degree + k - 2 - prev)
        out.append(tuple(c / degree for c in counts))
    return out


def _dedupe(candidates: Iterable[Candidate]) -> list[Candidate]:
    """First occurrence of each distinct spend vector wins.

    Ordering therefore decides which ID survives a collision, and the callers
    put the Phase 3 candidates first on purpose so `even_split` keeps its name
    rather than becoming `lat3_017`.
    """
    seen: dict[tuple[float, ...], Candidate] = {}
    for cand in candidates:
        if cand.spend not in seen:
            seen[cand.spend] = cand
    return list(seen.values())


@dataclass(frozen=True)
class Stage1Config:
    """Counts for the broad stage.

    Defaults give 121 structured + 450 random = 571 distinct candidates before
    stage 2, which is the low end of the 500-800 target and leaves room for
    refinement. The structured / random split is roughly 1:4: the lattice
    guarantees coverage of the boundary and the low-dimensional faces (which
    random sampling essentially never visits), while the Dirichlet points give
    the interior a density no affordable lattice could.
    """
    include_phase3: bool = True
    lattice_degrees: tuple[int, ...] = (3, 4)
    include_centroid: bool = True
    # (alpha, count). alpha < 1 pushes mass to the faces/corners (zero channels);
    # alpha = 1 is uniform on the simplex; alpha > 1 concentrates near the centroid.
    dirichlet: tuple[tuple[float, int], ...] = ((1.0, 225), (0.3, 125), (4.0, 100))


def _dirichlet_rng(master_seed: int, label: str, index: int) -> np.random.Generator:
    """Generator for candidate GENERATION, domain-separated from the simulation.

    Candidate generation must never share a stream with the Monte Carlo draws --
    otherwise the allocations being tested would be correlated with the shocks
    they are tested against, which is a subtle way to make a sweep lie.
    """
    return np.random.default_rng(
        np.random.SeedSequence(
            entropy=[int(master_seed), stable_hash_int("phase4-candidate-generation"),
                     stable_hash_int(label), int(index)]
        )
    )


def generate_stage1(
    channels: Sequence[str],
    total_budget: float = TOTAL_BUDGET,
    config: Stage1Config = Stage1Config(),
    master_seed: int = MASTER_SEED,
) -> list[Candidate]:
    """Broad coverage of the simplex. Deterministic given `master_seed`."""
    channels = list(channels)
    k = len(channels)
    out: list[Candidate] = []

    if config.include_phase3:
        for alloc_id, alloc in build_allocation_candidates(channels, total_budget).items():
            out.append(Candidate(
                allocation_id=alloc_id, family="phase3", stage=1,
                spend=tuple(float(alloc[c]) for c in channels),
            ))

    if config.include_centroid:
        out.append(Candidate(
            allocation_id="centroid", family="structured", stage=1,
            spend=quantize_to_whole_dollars([1.0] * k, total_budget),
        ))

    for degree in config.lattice_degrees:
        for i, w in enumerate(simplex_lattice(k, degree)):
            out.append(Candidate(
                allocation_id=f"lat{degree}_{i:03d}", family=f"lattice{degree}", stage=1,
                spend=quantize_to_whole_dollars(w, total_budget),
            ))

    for j, (alpha, count) in enumerate(config.dirichlet):
        if count <= 0:
            continue
        rng = _dirichlet_rng(master_seed, f"dirichlet-alpha-{alpha}", j)
        draws = rng.dirichlet(np.full(k, float(alpha)), size=int(count))
        tag = f"{alpha:g}".replace(".", "p")
        for i, w in enumerate(draws):
            out.append(Candidate(
                allocation_id=f"dir{tag}_{i:03d}", family=f"dirichlet_a{tag}", stage=1,
                spend=quantize_to_whole_dollars(w, total_budget),
            ))

    return _dedupe(out)


@dataclass(frozen=True)
class Stage2Config:
    """Refinement around the stage-1 Pareto set.

    `perturb_concentrations` are Dirichlet concentrations c in
    `alpha = c * w + a0`: the component standard deviation is about
    sqrt(w(1-w)/(c+1)), so c = 400 explores roughly +/- 2.5 percentage points
    around a 50% weight and c = 100 roughly +/- 5. Two rings rather than one
    because a single scale either misses fine structure or never leaves the
    immediate neighbourhood.

    `a0` is what lets refinement leave a face: with a0 = 0 a channel sitting at
    exactly zero has alpha = 0, which is not a valid Dirichlet parameter, and the
    channel could never come back.

    `blend_fractions` densify the SEGMENT between frontier points. On this model
    that is the highest-value move available -- the prerequisite found a 2-point
    mean-VaR Pareto set, and nothing else in the design samples between two
    specific points.
    """
    perturb_concentrations: tuple[float, ...] = (400.0, 100.0)
    n_perturbations_per_point: int = 8   # per concentration, so 16 per frontier point
    a0: float = 0.15
    blend_fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    max_blend_pairs: int = 45            # C(10,2); above that, deterministically subsampled
    max_candidates: int = 400


def generate_stage2(
    channels: Sequence[str],
    seed_points: Sequence[Candidate],
    total_budget: float = TOTAL_BUDGET,
    config: Stage2Config = Stage2Config(),
    master_seed: int = MASTER_SEED,
) -> list[Candidate]:
    """Local candidates around `seed_points` (the stage-1 Pareto set).

    Deterministic given `master_seed` AND the seed points, which come from the
    stage-1 result -- so stage 2 is reproducible only together with stage 1.
    """
    channels = list(channels)
    k = len(channels)
    if not seed_points:
        return []
    out: list[Candidate] = []

    # 1. perturbation clouds
    for p_idx, parent in enumerate(sorted(seed_points, key=lambda c: c.allocation_id)):
        w = np.asarray(parent.shares(total_budget), dtype=float)
        for c_idx, conc in enumerate(config.perturb_concentrations):
            rng = _dirichlet_rng(master_seed, f"refine-{parent.allocation_id}-{conc:g}",
                                 p_idx * 97 + c_idx)
            alpha = conc * w + config.a0
            draws = rng.dirichlet(alpha, size=config.n_perturbations_per_point)
            for i, wd in enumerate(draws):
                out.append(Candidate(
                    allocation_id=f"ref_{parent.allocation_id}_c{conc:g}_{i:02d}",
                    family="refine", stage=2,
                    spend=quantize_to_whole_dollars(wd, total_budget),
                    parents=(parent.allocation_id,),
                ))

    # 2. convex blends between frontier points
    ordered = sorted(seed_points, key=lambda c: c.allocation_id)
    pairs = list(itertools.combinations(range(len(ordered)), 2))
    if len(pairs) > config.max_blend_pairs:
        rng = _dirichlet_rng(master_seed, "blend-pair-subsample", 0)
        keep = rng.choice(len(pairs), size=config.max_blend_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(keep.tolist())]
    for i, j in pairs:
        a, b = ordered[i], ordered[j]
        wa = np.asarray(a.shares(total_budget), dtype=float)
        wb = np.asarray(b.shares(total_budget), dtype=float)
        for t in config.blend_fractions:
            out.append(Candidate(
                allocation_id=f"blend_{a.allocation_id}__{b.allocation_id}_t{t:g}",
                family="blend", stage=2,
                spend=quantize_to_whole_dollars((1.0 - t) * wa + t * wb, total_budget),
                parents=(a.allocation_id, b.allocation_id),
            ))

    out = _dedupe(out)
    return _cap_preserving_families(out, config.max_candidates)


def _cap_preserving_families(candidates: list[Candidate], cap: int) -> list[Candidate]:
    """Trim to `cap` without letting one family starve the other.

    A plain `out[:cap]` is WRONG here and was the original bug. Perturbations
    are appended before blends, so a flat truncation kept every perturbation and
    dropped every blend -- deleting precisely the move this stage exists for,
    since blends are the only thing that samples BETWEEN two frontier points and
    the prerequisite left us a 2-point mean-VaR set to densify.

    So each family gets an equal quota; whatever a smaller family does not use
    is handed to the others rather than wasted. Order within a family is
    preserved, so the result stays deterministic.
    """
    if cap <= 0 or len(candidates) <= cap:
        return candidates

    by_family: dict[str, list[Candidate]] = {}
    for cand in candidates:
        by_family.setdefault(cand.family, []).append(cand)

    quotas = {fam: 0 for fam in by_family}
    remaining = cap
    # Round-robin one slot at a time: exact, and self-balancing when families
    # have very different sizes (no leftover from integer division).
    while remaining > 0:
        progressed = False
        for fam, members in by_family.items():
            if remaining == 0:
                break
            if quotas[fam] < len(members):
                quotas[fam] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # every family exhausted
            break

    kept_ids = set()
    for fam, members in by_family.items():
        kept_ids.update(id(m) for m in members[:quotas[fam]])
    return [c for c in candidates if id(c) in kept_ids]


def candidate_frame(
    candidates: Sequence[Candidate], channels: Sequence[str], total_budget: float = TOTAL_BUDGET
) -> pd.DataFrame:
    """Long-form (allocation, channel) rows -- the shape `silver.allocation_candidates` uses."""
    rows = []
    for cand in candidates:
        for c, s in zip(channels, cand.spend):
            rows.append({
                "allocation_id": cand.allocation_id,
                "channel_id": c,
                "spend_pct": s / total_budget,
                "spend_dollars": s,
                "total_budget": float(total_budget),
                "allocation_family": cand.family,
                "search_stage": cand.stage,
            })
    return pd.DataFrame(rows)


def assert_candidate_set_valid(
    candidates: Sequence[Candidate], channels: Sequence[str], total_budget: float = TOTAL_BUDGET
) -> dict:
    """Hard invariants on a candidate set. Raises on the first violation.

    Exact sum to budget (float `==`, not a tolerance), no negatives, zeros
    allowed, unique ids, unique spend vectors.
    """
    k = len(channels)
    seen_ids: set[str] = set()
    seen_spend: dict[tuple[float, ...], str] = {}
    n_with_zero = 0
    for cand in candidates:
        if len(cand.spend) != k:
            raise AssertionError(f"{cand.allocation_id}: {len(cand.spend)} channels, expected {k}")
        total = math.fsum(cand.spend)
        if total != float(total_budget):
            raise AssertionError(
                f"{cand.allocation_id}: spend sums to {total!r}, expected exactly {total_budget!r}"
            )
        if any(s < 0.0 for s in cand.spend):
            raise AssertionError(f"{cand.allocation_id}: negative spend {cand.spend}")
        if any(s != float(int(s)) for s in cand.spend):
            raise AssertionError(f"{cand.allocation_id}: non-integer dollars {cand.spend}")
        if cand.allocation_id in seen_ids:
            raise AssertionError(f"duplicate allocation_id {cand.allocation_id!r}")
        seen_ids.add(cand.allocation_id)
        if cand.spend in seen_spend:
            raise AssertionError(
                f"{cand.allocation_id} duplicates the spend vector of {seen_spend[cand.spend]}"
            )
        seen_spend[cand.spend] = cand.allocation_id
        if any(s == 0.0 for s in cand.spend):
            n_with_zero += 1
    return {
        "n_candidates": len(candidates),
        "n_with_a_zero_channel": n_with_zero,
        "n_fully_funded": len(candidates) - n_with_zero,
        "min_channel_spend": min((min(c.spend) for c in candidates), default=None),
        "max_channel_spend": max((max(c.spend) for c in candidates), default=None),
    }


def assert_matches_phase3_quantizer(
    channels: Sequence[str], total_budget: float = TOTAL_BUDGET
) -> int:
    """Prove this module's quantizer agrees with `allocations.py` on the Phase 3 set.

    Both round to whole dollars and absorb the residual into the largest-weight
    channel; the only intended difference is that this one tolerates a zero. If
    they ever diverge, the Phase 3 candidates inside the Phase 4 sweep would stop
    being the same allocations Phase 3 measured, and every continuity comparison
    would be quietly wrong. Returns the number of allocations checked.
    """
    try:
        from .allocations import build_allocation_weights
    except ImportError:
        from allocations import build_allocation_weights
    channels = list(channels)
    reference = build_allocation_candidates(channels, total_budget)
    weights = build_allocation_weights(channels)
    for alloc_id, w in weights.items():
        mine = quantize_to_whole_dollars([w[c] for c in channels], total_budget)
        theirs = tuple(float(reference[alloc_id][c]) for c in channels)
        if mine != theirs:
            raise AssertionError(
                f"quantizer disagreement on {alloc_id!r}: frontier {mine} vs allocations {theirs}"
            )
    return len(weights)


# =============================================================================
# The per-cell unit of work
# =============================================================================

def summarize_cell(
    allocation_id: str,
    scenario_id: str,
    allocation: dict[str, float],
    inputs: SweepInputs,
    scenario: Scenario,
    seed: int,
    var_confidence: float = VAR_CONFIDENCE,
) -> dict:
    """Run ONE (allocation, scenario) cell and return ONE aggregate row.

    This is the function the Spark UDF should call. It wraps `cell.simulate_cell`
    -- which is itself a wrapper over the frozen engine -- so the scenario
    multipliers, the saturation composition order and the Beta-domain guards all
    stay in exactly one place. Nothing here re-implements any of that.

    The path vector is aggregated and DROPPED. See the module docstring for what
    that costs.

    `prob_below_breakeven` is P(revenue < total_spend), i.e. P(ROAS < 1) -- the
    probability the campaign destroys money outright, which is a far more legible
    risk statement to a marketing audience than a percentile in dollars, and it
    needs the path vector to compute, so it has to happen here.
    """
    df = cell.simulate_cell(
        allocation_id=allocation_id,
        scenario_id=scenario_id,
        channels=list(inputs.channels),
        allocation=allocation,
        mean_cvr=inputs.mean_cvr, std_cvr=inputs.std_cvr,
        mean_cpc=inputs.mean_cpc, std_cpc=inputs.std_cpc,
        mean_rpc=inputs.mean_rpc, std_rpc=inputs.std_rpc,
        correlation=inputs.correlation,
        cvr_multiplier=scenario.cvr_multiplier,
        cpc_multiplier=scenario.cpc_multiplier,
        revenue_multiplier=scenario.revenue_multiplier,
        n_paths=inputs.n_paths,
        seed=seed,
        theta=inputs.theta,
        reference_spend=inputs.reference_spend,
        saturate_std_cpc=inputs.saturate_std_cpc,
    )
    revenue = df["total_simulated_revenue"].to_numpy(dtype=float)
    total_spend = float(df["total_spend"].iloc[0])
    return summarize_revenue(
        allocation_id, scenario_id, revenue, total_spend, int(seed), var_confidence
    )


def summarize_revenue(
    allocation_id: str,
    scenario_id: str,
    revenue: np.ndarray,
    total_spend: float,
    seed: int,
    var_confidence: float = VAR_CONFIDENCE,
) -> dict:
    """The aggregation itself, split out so it can be tested on a known vector.

    VaR-95 is the 5th percentile of revenue (95% chance of landing above it);
    CVaR-95 is the mean of the paths at or below it. Same definitions as
    `engine.risk_metrics`, recomputed here rather than imported so this function
    has no dependency on a result object -- and cross-checked against it in
    validation.
    """
    revenue = np.asarray(revenue, dtype=float)
    if revenue.ndim != 1 or revenue.size == 0:
        raise ValueError(f"revenue must be a non-empty 1-D array, got shape {revenue.shape}")
    if not np.all(np.isfinite(revenue)):
        raise ValueError(f"revenue contains non-finite values for {allocation_id}/{scenario_id}")
    if total_spend <= 0.0:
        raise ValueError(f"total_spend must be > 0, got {total_spend}")

    n = int(revenue.size)
    tail_pct = (1.0 - var_confidence) * 100.0
    var = float(np.percentile(revenue, tail_pct))
    tail = revenue[revenue <= var]
    std = float(revenue.std(ddof=1))
    mean = float(revenue.mean())
    return {
        "allocation_id": allocation_id,
        "scenario_id": scenario_id,
        "n_paths": n,
        "seed": int(seed),
        "total_spend": float(total_spend),
        "mean_revenue": mean,
        "std_revenue": std,
        "se_mean_revenue": std / math.sqrt(n),
        "min_revenue": float(revenue.min()),
        "max_revenue": float(revenue.max()),
        "median_revenue": float(np.median(revenue)),
        "var_95": var,
        "cvar_95": float(tail.mean()) if tail.size else float("nan"),
        "expected_roas": mean / float(total_spend),
        "prob_below_breakeven": float(np.mean(revenue < total_spend)),
    }


def summarize_cell_row(row: dict, inputs: SweepInputs, scenarios_by_id: dict[str, Scenario]) -> dict:
    """Dict-in / dict-out wrapper, the shape an `applyInPandas` UDF wants.

    `row` needs: allocation_id, scenario_id, seed, and one `spend_<channel>` key
    per channel (or a `spend` sequence aligned to `inputs.channels`).
    """
    channels = list(inputs.channels)
    if "spend" in row:
        allocation = {c: float(s) for c, s in zip(channels, row["spend"])}
    else:
        allocation = {c: float(row[f"spend_{c}"]) for c in channels}
    return summarize_cell(
        allocation_id=str(row["allocation_id"]),
        scenario_id=str(row["scenario_id"]),
        allocation=allocation,
        inputs=inputs,
        scenario=scenarios_by_id[str(row["scenario_id"])],
        seed=int(row["seed"]),
    )


def run_sweep(
    candidates: Sequence[Candidate],
    inputs: SweepInputs,
    scenarios: Sequence[Scenario] = SCENARIOS,
    master_seed: int = MASTER_SEED,
    replicate: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Every (candidate, scenario) cell, single-threaded. One row per cell.

    This is the reference implementation the Spark version must reproduce; it is
    not how the full sweep is meant to be run. The loop is deliberately trivial
    -- the seed depends only on the SCENARIO (common random numbers, see
    `sweep_seeding.py`) so cells are embarrassingly parallel with no shared
    state.
    """
    seeds = {s.scenario_id: crn_seed(s.scenario_id, master_seed, replicate) for s in scenarios}
    rows: list[dict] = []
    total = len(candidates) * len(scenarios)
    for scenario in scenarios:
        seed = seeds[scenario.scenario_id]
        for cand in candidates:
            rows.append(summarize_cell(
                allocation_id=cand.allocation_id,
                scenario_id=scenario.scenario_id,
                allocation=cand.as_dict(inputs.channels),
                inputs=inputs,
                scenario=scenario,
                seed=seed,
            ))
            if progress is not None and len(rows) % 100 == 0:
                progress(len(rows), total)
    return pd.DataFrame(rows, columns=list(SWEEP_OUTPUT_COLUMNS))


# =============================================================================
# Pareto / frontier
# =============================================================================

# The three objective pairs, each (return_key, risk_key, higher_risk_is_better).
# `std` is a cost, so lower is better; VaR-95 and CVaR-95 are dollar floors, so
# higher is better.
OBJECTIVE_PAIRS: tuple[tuple[str, str, bool], ...] = (
    ("mean_revenue", "var_95", True),
    ("mean_revenue", "cvar_95", True),
    ("mean_revenue", "std_revenue", False),
)


def non_dominated_mask(objectives: np.ndarray, maximize: Sequence[bool]) -> np.ndarray:
    """Boolean mask of the Pareto-optimal rows of an (n, m) objective matrix.

    A point is DOMINATED if some other point is at least as good on every
    objective and strictly better on at least one. Exact float comparison, so
    two identical rows are mutually non-dominated and BOTH survive -- which is
    correct (they are the same allocation as far as these objectives can tell)
    and is why the counts below can exceed what a picture suggests.
    """
    obj = np.asarray(objectives, dtype=float)
    if obj.ndim != 2:
        raise ValueError(f"objectives must be 2-D, got shape {obj.shape}")
    if obj.shape[1] != len(maximize):
        raise ValueError(f"{obj.shape[1]} objectives but {len(maximize)} directions")
    if not np.all(np.isfinite(obj)):
        raise ValueError("objectives contain non-finite values")
    sign = np.where(np.asarray(maximize, dtype=bool), 1.0, -1.0)
    signed = obj * sign

    n = signed.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        ge = np.all(signed >= signed[i], axis=1)
        gt = np.any(signed > signed[i], axis=1)
        if np.any(ge & gt):
            keep[i] = False
    return keep


def frontier_for_pair(
    df: pd.DataFrame, return_key: str, risk_key: str, higher_risk_is_better: bool
) -> pd.DataFrame:
    """`df` (one scenario) with an `is_efficient` column for one objective pair."""
    out = df.copy()
    mask = non_dominated_mask(
        out[[return_key, risk_key]].to_numpy(dtype=float),
        maximize=(True, higher_risk_is_better),
    )
    out["is_efficient"] = mask
    return out


def frontier_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated count per (scenario, objective pair). The headline table."""
    rows = []
    for scenario_id, sub in df.groupby("scenario_id", sort=True):
        for ret, risk, higher_better in OBJECTIVE_PAIRS:
            mask = non_dominated_mask(
                sub[[ret, risk]].to_numpy(dtype=float), maximize=(True, higher_better)
            )
            rows.append({
                "scenario_id": scenario_id,
                "objective_pair": f"{ret} vs {risk}",
                "n_candidates": int(len(sub)),
                "n_non_dominated": int(mask.sum()),
                "pct_non_dominated": 100.0 * mask.sum() / len(sub),
            })
    return pd.DataFrame(rows)


def pareto_union_ids(
    df: pd.DataFrame, objective_pairs: Sequence[tuple[str, str, bool]] = OBJECTIVE_PAIRS
) -> list[str]:
    """Allocation ids non-dominated in ANY (scenario, objective pair). Sorted.

    This is the seed set for stage 2. The union rather than one pair's frontier,
    because the three objective pairs disagree about what "efficient" means (a
    variance-efficient allocation need not be VaR-efficient) and because
    refinement is cheap relative to being wrong about where to spend it. Across
    scenarios for the same reason: an allocation that is only efficient in a
    recession is exactly the kind of point a risk-tolerance frontier should have
    resolution around.
    """
    ids: set[str] = set()
    for _, sub in df.groupby("scenario_id", sort=True):
        for ret, risk, higher_better in objective_pairs:
            mask = non_dominated_mask(
                sub[[ret, risk]].to_numpy(dtype=float), maximize=(True, higher_better)
            )
            ids.update(sub.loc[mask, "allocation_id"].tolist())
    return sorted(ids)


def _knee_index(ret_norm: np.ndarray, risk_norm: np.ndarray) -> int:
    """Index of the knee: max distance above the chord joining the two extremes.

    With both coordinates min-max normalised to [0,1] and oriented so higher is
    better, the chord between the two extreme frontier points is the line
    `x + y = 1`, so the perpendicular distance is `(x + y - 1) / sqrt(2)` and
    maximising the distance is maximising `x + y`. Ties break toward higher
    return, which makes the choice deterministic.
    """
    score = ret_norm + risk_norm
    best = float(score.max())
    tied = np.flatnonzero(score >= best - 1e-12)
    return int(tied[np.argmax(ret_norm[tied])])


# Standard deviation of the CRN difference between two allocations' mean
# revenue, MEASURED on this model at 10,000 paths. It is not one number: CRN
# cancels COMMON noise, so the gain grows as two allocations get more similar.
# Over 300 replicates -- distant pair $647, near-frontier pair $561, nearly
# identical pair $45; and the tightest adjacent pair on the actual normal
# mean-VaR frontier came out at $322 over 200 replicates. The default below is
# the conservative (largest) end, because over-stating the noise flags too many
# picks as unresolved, which is the safe direction to err in.
CRN_DIFF_SD_DEFAULT: float = 650.0


def recommend_from_frontier(
    df: pd.DataFrame, return_key: str, risk_key: str, higher_risk_is_better: bool,
    noise_scale: float = CRN_DIFF_SD_DEFAULT,
) -> pd.DataFrame:
    """Three recommendations per (scenario, objective pair), by a fixed rule.

    THE RULE, stated so it can be argued with rather than trusted:

      max_return  argmax(return) over the efficient set. Ties -> better risk.
                  For an aggressive advertiser with no downside constraint.
      min_risk    the best risk value over the efficient set. Ties -> higher
                  return. For a floor-constrained advertiser.
      balanced    the KNEE of the efficient set: normalise both objectives to
                  [0,1] across the frontier, orient both so higher is better, and
                  take the point furthest above the chord joining the two
                  extremes. That is the point where you stop buying much return
                  per unit of risk given up. Ties -> higher return.

    Nothing is hand-picked, and no weights are invented: the knee is a property
    of the frontier's own curvature. It is only meaningful when the frontier HAS
    curvature -- with fewer than 3 efficient points there is no interior to be
    the knee of, and `balanced_is_degenerate` flags that instead of quietly
    returning an endpoint as though it were a considered choice.

    RESOLUTION -- a frontier can be non-degenerate and still not ordered.
    `balanced_is_degenerate` only asks whether the frontier has an interior. It
    does not ask whether the points are far enough apart for their ORDER to be
    real. On the `normal` mean-VaR frontier the adjacent gaps in expected
    revenue are $9,961 / $2,714 / $6,777 / $5,971 / $323, and that last pair was
    measured directly: over 200 replicates the CRN standard deviation of their
    difference is $322, so the gap is 1.0x the noise and ranks 5 and 6 are NOT
    ordered by this sweep. Picking between them implies a precision the data
    does not have.

    So each pick also reports `nearest_neighbour_gap` (its distance along the
    return axis to the closest other efficient point) and `ordering_unresolved`
    (that gap is under `2 * noise_scale`). This flags the problem rather than
    hiding it; it does not change which point is chosen.
    """
    rows = []
    for scenario_id, sub in df.groupby("scenario_id", sort=True):
        eff = frontier_for_pair(sub, return_key, risk_key, higher_risk_is_better)
        eff = eff[eff["is_efficient"]].reset_index(drop=True)
        ret = eff[return_key].to_numpy(dtype=float)
        risk = eff[risk_key].to_numpy(dtype=float)
        risk_good = risk if higher_risk_is_better else -risk

        def _norm(v: np.ndarray) -> np.ndarray:
            span = float(v.max() - v.min())
            return np.zeros_like(v) if span == 0.0 else (v - v.min()) / span

        picks = {
            "max_return": int(np.lexsort((-risk_good, -ret))[0]),
            "min_risk": int(np.lexsort((-ret, -risk_good))[0]),
            "balanced": _knee_index(_norm(ret), _norm(risk_good)),
        }
        degenerate = len(eff) < 3
        for label, idx in picks.items():
            row = eff.iloc[idx].to_dict()
            others = np.delete(ret, idx)
            gap = float(np.abs(others - ret[idx]).min()) if others.size else float("nan")
            row.update({
                "recommendation": label,
                "objective_pair": f"{return_key} vs {risk_key}",
                "n_efficient": int(len(eff)),
                "balanced_is_degenerate": bool(degenerate and label == "balanced"),
                "nearest_neighbour_gap": gap,
                "ordering_unresolved": bool(np.isfinite(gap) and gap < 2.0 * noise_scale),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def profit_per_unit_risk(df: pd.DataFrame) -> pd.DataFrame:
    """`(mean_revenue - total_spend) / std_revenue`, ranked, per scenario.

    A Sharpe analogue with the budget itself as the risk-free benchmark: expected
    PROFIT per dollar of revenue standard deviation. It needs no invented
    risk-free rate -- not spending is genuinely the zero-risk, zero-profit
    alternative -- and its maximiser is always on the mean-std efficient frontier
    (anything with higher mean and lower std would score higher), so it is a
    frontier point selected by a single scalar rule rather than an extra
    candidate from nowhere.
    """
    out = df.copy()
    out["profit_per_unit_risk"] = (
        (out["mean_revenue"] - out["total_spend"]) / out["std_revenue"]
    )
    out["rank_in_scenario"] = out.groupby("scenario_id")["profit_per_unit_risk"].rank(
        ascending=False, method="min"
    ).astype(int)
    return out.sort_values(["scenario_id", "rank_in_scenario"]).reset_index(drop=True)


def frontier_robustness(
    df_a: pd.DataFrame, df_b: pd.DataFrame, return_key: str, risk_key: str,
    higher_risk_is_better: bool,
) -> pd.DataFrame:
    """How much frontier MEMBERSHIP moves between two independent replicates.

    The frontier is a discrete verdict read off noisy numbers, so the honest
    question is not "is it correct" but "how much of it survives a different
    random world". Jaccard overlap of the efficient sets, per scenario.
    """
    rows = []
    for scenario_id in sorted(set(df_a["scenario_id"]) & set(df_b["scenario_id"])):
        sets = []
        for df in (df_a, df_b):
            sub = df[df["scenario_id"] == scenario_id]
            eff = frontier_for_pair(sub, return_key, risk_key, higher_risk_is_better)
            sets.append(set(eff.loc[eff["is_efficient"], "allocation_id"]))
        a, b = sets
        union = a | b
        rows.append({
            "scenario_id": scenario_id,
            "objective_pair": f"{return_key} vs {risk_key}",
            "n_efficient_a": len(a),
            "n_efficient_b": len(b),
            "n_shared": len(a & b),
            "jaccard": (len(a & b) / len(union)) if union else float("nan"),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Independent reference points (no simulation involved)
# =============================================================================

def analytic_expected_revenue_weights(
    weights: np.ndarray, inputs: SweepInputs, scenario: Scenario,
    clip_subsidy: bool = False, per_channel: bool = False,
):
    """Closed-form E[revenue] for a weight vector -- no Monte Carlo, no RNG.

        E[rev_i] = spend_i * E[1/CPC_i] * E[CVR_i] * E[RPC_i]
        E[1/CPC] = exp(-mu + sigma^2/2) for LogNormal(mu, sigma)

    with the saturation curve folded into the CPC mean. CVR, CPC and RPC are
    mutually independent in this model, so the expectation factorises exactly:
    this is a HARD target for the simulated mean, not an approximation. Used to
    check the sweep from outside itself.

    `clip_subsidy=True` floors the saturation multiplier at 1.0: a channel funded
    ABOVE the reference still pays its volume penalty, but a channel funded BELOW
    it gets NO discount. That is the counterfactual for
    `subsidy_decomposition` -- see there for why it matters more to a boundary
    search than it did to Phase 3's 5%-floor grid.
    """
    w = np.asarray(weights, dtype=float)
    spend = w * inputs.total_budget
    mean_cpc = np.asarray(inputs.mean_cpc, dtype=float) * scenario.cpc_multiplier
    std_cpc = np.asarray(inputs.std_cpc, dtype=float) * scenario.cpc_multiplier
    mean_cvr = np.asarray(inputs.mean_cvr, dtype=float) * scenario.cvr_multiplier
    mean_rpc = np.asarray(inputs.mean_rpc, dtype=float) * scenario.revenue_multiplier

    if inputs.theta is not None:
        th = np.asarray(inputs.theta, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mult = np.where(spend > 0.0, (spend / inputs.reference_spend) ** th, 0.0)
        if clip_subsidy:
            mult = np.maximum(mult, 1.0)
        eff_mean_cpc = np.where(spend > 0.0, mean_cpc * mult, np.inf)
        eff_std_cpc = std_cpc * mult if inputs.saturate_std_cpc else std_cpc
    else:
        eff_mean_cpc = np.where(spend > 0.0, mean_cpc, np.inf)
        eff_std_cpc = std_cpc

    by_channel = np.zeros(inputs.k, dtype=float)
    for i in range(inputs.k):
        if spend[i] <= 0.0:
            continue
        sigma_sq = np.log1p((eff_std_cpc[i] / eff_mean_cpc[i]) ** 2)
        mu = np.log(eff_mean_cpc[i]) - sigma_sq / 2.0
        e_inv_cpc = float(np.exp(-mu + sigma_sq / 2.0))
        by_channel[i] = float(spend[i]) * e_inv_cpc * float(mean_cvr[i]) * float(mean_rpc[i])
    return by_channel if per_channel else float(by_channel.sum())


def subsidy_decomposition(
    candidates: Sequence[Candidate], inputs: SweepInputs, scenario: Scenario
) -> pd.DataFrame:
    """How much of each allocation's expected revenue comes from the SUB-REFERENCE
    CPC discount -- the part of the model that bronze cannot identify.

    WHY THIS EXISTS. `effective_cpc = mean_cpc * (spend/100_000)**theta` is a
    penalty above the reference and a DISCOUNT below it, and the discount has no
    data behind it: bronze has no spend variation at all, let alone at a quarter
    or a thousandth of the reference. The prerequisite measured this at
    `dominant_paid_search`, whose thinnest channel is $25,000 (0.25x reference),
    and found $48,161 -- 4.8% of its expected revenue -- came from the discount,
    enough to flip it from beating `even_split` to losing to it.

    A SIMPLEX SEARCH MAKES THIS WORSE, NOT BETTER, and that is the point of
    measuring it here. Lattice vertices and low-alpha Dirichlet draws put
    channels at $0 to a few thousand dollars, i.e. 1e-5 to 1e-2 of the reference,
    where the power law is being extrapolated four to five orders of magnitude
    beyond anything observed. The discount is unbounded as spend falls: at $1 in
    paid_search the model prices a click at 3.6 cents instead of $2.10.

    Columns:
      `expected_revenue`          as modelled
      `expected_revenue_clipped`  with the multiplier floored at 1.0 (penalty
                                  above reference kept, discount below removed)
      `subsidy_dollars` / `subsidy_share`   the difference, and its share
      `min_spend_ratio`           smallest funded channel as a multiple of
                                  reference -- the depth of the extrapolation
      `n_below_reference`         how many channels are in the extrapolated region
    """
    rows = []
    for cand in candidates:
        w = np.asarray(cand.shares(inputs.total_budget), dtype=float)
        full = analytic_expected_revenue_weights(w, inputs, scenario)
        clipped = analytic_expected_revenue_weights(w, inputs, scenario, clip_subsidy=True)
        funded = [s for s in cand.spend if s > 0.0]
        rows.append({
            "allocation_id": cand.allocation_id,
            "family": cand.family,
            "expected_revenue": full,
            "expected_revenue_clipped": clipped,
            "subsidy_dollars": full - clipped,
            "subsidy_share": (full - clipped) / full if full > 0 else float("nan"),
            "min_spend_ratio": (min(funded) / inputs.reference_spend) if funded else 0.0,
            "n_below_reference": int(sum(
                1 for s in cand.spend if 0.0 < s < inputs.reference_spend)),
            "n_idle": int(sum(1 for s in cand.spend if s == 0.0)),
        })
    return pd.DataFrame(rows)


def analytic_interior_optimum(
    inputs: SweepInputs, scenario: Scenario, jensen: bool = True
) -> dict:
    """Continuous expected-revenue optimum over the simplex, in closed form.

    Under saturation each channel contributes `A_i * spend_i ** (1 - theta_i)`
    with `A_i` constant, so maximising the sum subject to a budget gives a
    strictly interior solution. Solved by bisection on the Lagrange multiplier,
    which is exact to machine precision for a one-dimensional monotone equation.

    Deliberately independent of the search: if the discrete argmax is far from
    this point, the candidate set is too coarse, and that is a finding about the
    SEARCH rather than about the model. Only valid with saturation on -- without
    it the problem is linear and the optimum is a corner, so it raises.

    `jensen=True` includes the exp(sigma^2) correction on E[1/CPC], matching what
    the simulation actually produces. Note that with the default (mean-only)
    saturation spec, sigma itself depends on spend, so the corrected objective is
    no longer exactly a power law and the closed form is a very good
    approximation rather than an identity -- the uncorrected version is exact.
    """
    if inputs.theta is None:
        raise ValueError(
            "no interior optimum without saturation: with theta=None expected revenue is "
            "linear in the weights and the maximum is a simplex corner"
        )
    th = np.asarray(inputs.theta, dtype=float)
    mean_cpc = np.asarray(inputs.mean_cpc, dtype=float) * scenario.cpc_multiplier
    std_cpc = np.asarray(inputs.std_cpc, dtype=float) * scenario.cpc_multiplier
    mean_cvr = np.asarray(inputs.mean_cvr, dtype=float) * scenario.cvr_multiplier
    mean_rpc = np.asarray(inputs.mean_rpc, dtype=float) * scenario.revenue_multiplier

    # rev_i(s) = A_i * s ** (1 - theta_i)
    a = (inputs.reference_spend ** th) / mean_cpc * mean_cvr * mean_rpc
    if jensen:
        a = a * np.exp(np.log1p((std_cpc / mean_cpc) ** 2))

    # d rev_i / d s = A_i (1-theta_i) s ** (-theta_i) = lam  ->
    #   s_i(lam) = (A_i (1-theta_i) / lam) ** (1/theta_i)
    def spend_at(lam: float) -> np.ndarray:
        return (a * (1.0 - th) / lam) ** (1.0 / th)

    lo, hi = 1e-12, 1e12
    for _ in range(400):
        mid = math.sqrt(lo * hi)
        if spend_at(mid).sum() > inputs.total_budget:
            lo = mid
        else:
            hi = mid
    spend = spend_at(math.sqrt(lo * hi))
    spend = spend * (inputs.total_budget / spend.sum())
    weights = spend / inputs.total_budget
    return {
        "weights": {c: float(w) for c, w in zip(inputs.channels, weights)},
        "spend": {c: float(s) for c, s in zip(inputs.channels, spend)},
        "expected_revenue": analytic_expected_revenue_weights(weights, inputs, scenario),
        "marginal_roas": {
            c: float(a[i] * (1.0 - th[i]) * spend[i] ** (-th[i]))
            for i, c in enumerate(inputs.channels)
        },
    }


def marginal_roas(weights: np.ndarray, inputs: SweepInputs, scenario: Scenario) -> dict[str, float]:
    """d E[revenue] / d spend, per channel, at a given allocation.

    Under saturation this varies across allocations; under the linear Phase 2/3
    model it is a constant, which is exactly why that model always answered
    "put everything in the best channel". At the interior optimum every FUNDED
    channel's marginal is equal -- that equality is the optimality condition and
    is a good check on the search.
    """
    w = np.asarray(weights, dtype=float)
    spend = w * inputs.total_budget
    mean_cpc = np.asarray(inputs.mean_cpc, dtype=float) * scenario.cpc_multiplier
    std_cpc = np.asarray(inputs.std_cpc, dtype=float) * scenario.cpc_multiplier
    mean_cvr = np.asarray(inputs.mean_cvr, dtype=float) * scenario.cvr_multiplier
    mean_rpc = np.asarray(inputs.mean_rpc, dtype=float) * scenario.revenue_multiplier
    th = (np.zeros(inputs.k) if inputs.theta is None
          else np.asarray(inputs.theta, dtype=float))

    a = (inputs.reference_spend ** th) / mean_cpc * mean_cvr * mean_rpc
    a = a * np.exp(np.log1p((std_cpc / mean_cpc) ** 2))
    out = {}
    for i, c in enumerate(inputs.channels):
        out[c] = float("nan") if spend[i] <= 0.0 else float(
            a[i] * (1.0 - th[i]) * spend[i] ** (-th[i])
        )
    return out


# =============================================================================
# CRN validation and measurement
# =============================================================================

def validate_crn_sharing(
    inputs: SweepInputs,
    allocation_a: dict[str, float],
    allocation_b: dict[str, float],
    scenario: Scenario,
    seed: int,
) -> dict:
    """Show that two DIFFERENT allocations at the same seed share their randomness.

    Returns the actual matrix differences rather than a pass/fail, because the
    expected answer differs by matrix and the distinction matters:

      * `cvr_max_abs_diff` and `rpc_max_abs_diff` must be EXACTLY 0.0. Those
        marginals do not depend on the allocation, so the draws are bitwise
        identical -- this is the direct evidence that the stream is shared.
      * `cpc_max_abs_diff` will NOT be zero under saturation, and should not be:
        the mean CPC is spend-dependent by design. What must hold instead is that
        the UNDERLYING standard normals match, which is what
        `cpc_recovered_z_max_abs_diff` measures by inverting
        `cpc = exp(mu + sigma Z)` with each allocation's own fitted (mu, sigma).
        That is the definition of inverse-transform CRN.
      * `cpc_rank_agreement` is the fraction of channels whose CPC draws are
        perfectly rank-aligned between the two allocations (Spearman 1.0), which
        follows from a shared Z under any monotone map and is a distribution-free
        version of the same statement.
    """
    def _run(alloc: dict[str, float]) -> engine.SimulationResult:
        return engine.simulate(
            channels=list(inputs.channels), allocation=alloc,
            mean_cvr=np.asarray(inputs.mean_cvr) * scenario.cvr_multiplier,
            std_cvr=np.asarray(inputs.std_cvr) * scenario.cvr_multiplier,
            mean_cpc=np.asarray(inputs.mean_cpc) * scenario.cpc_multiplier,
            std_cpc=np.asarray(inputs.std_cpc) * scenario.cpc_multiplier,
            mean_rpc=np.asarray(inputs.mean_rpc) * scenario.revenue_multiplier,
            std_rpc=np.asarray(inputs.std_rpc) * scenario.revenue_multiplier,
            correlation=inputs.correlation, n_paths=inputs.n_paths, seed=seed,
            theta=inputs.theta, reference_spend=inputs.reference_spend,
            saturate_std_cpc=inputs.saturate_std_cpc,
        )

    ra, rb = _run(allocation_a), _run(allocation_b)

    z_a = np.empty_like(ra.cpc)
    z_b = np.empty_like(rb.cpc)
    for i, c in enumerate(inputs.channels):
        mu_a, sg_a = ra.lognormal_cpc_params[c]
        mu_b, sg_b = rb.lognormal_cpc_params[c]
        z_a[:, i] = (np.log(ra.cpc[:, i]) - mu_a) / sg_a
        z_b[:, i] = (np.log(rb.cpc[:, i]) - mu_b) / sg_b

    ranks_match = sum(
        1 for i in range(inputs.k)
        if np.array_equal(np.argsort(ra.cpc[:, i]), np.argsort(rb.cpc[:, i]))
    )
    return {
        "allocations_differ": not np.array_equal(ra.allocation, rb.allocation),
        "cvr_max_abs_diff": float(np.abs(ra.cvr - rb.cvr).max()),
        "cvr_bitwise_identical": bool(np.array_equal(ra.cvr, rb.cvr)),
        "rpc_max_abs_diff": float(np.abs(ra.rpc - rb.rpc).max()),
        "rpc_bitwise_identical": bool(np.array_equal(ra.rpc, rb.rpc)),
        "cpc_max_abs_diff": float(np.abs(ra.cpc - rb.cpc).max()),
        "cpc_bitwise_identical": bool(np.array_equal(ra.cpc, rb.cpc)),
        "cpc_recovered_z_max_abs_diff": float(np.abs(z_a - z_b).max()),
        "cpc_rank_agreement": ranks_match / inputs.k,
        "revenue_corr": float(np.corrcoef(ra.total_revenue, rb.total_revenue)[0, 1]),
    }


def measure_crn_variance_reduction(
    inputs: SweepInputs,
    candidates: Sequence[Candidate],
    scenario: Scenario,
    n_replicates: int,
    master_seed: int = MASTER_SEED,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Run the same allocation pairs BOTH ways and measure the variance reduction.

    For each replicate r, every candidate is simulated twice:
      * CRN arm         -- all candidates share `crn_seed(scenario, r)`;
      * independent arm -- each gets its own `independent_seed(alloc, scenario, r)`.

    Then for every unordered pair (A, B), the difference in mean revenue is
    computed within each replicate and the variance of that difference is taken
    ACROSS replicates. The variance-reduction factor is

        VRF = Var_indep(mean_A - mean_B) / Var_crn(mean_A - mean_B)

    which is the quantity that actually decides whether a dominance verdict is
    signal.

    TWO THEORETICAL COMPANIONS ARE RETURNED, AND THE DIFFERENCE BETWEEN THEM
    MATTERS -- getting this wrong is an easy way to report a false discrepancy:

      `vrf_upper_bound_from_rho` = 1 / (1 - rho) is the familiar textbook figure,
        but it is EXACT ONLY IF the two allocations have equal estimator variance.
        In general 2*rho*sqrt(vA*vB) <= rho*(vA+vB) by AM-GM, so this is a strict
        UPPER BOUND and the measured factor should sit below it. It is reported
        as a bound, not as a prediction.
      `vrf_predicted` = (vA + vB) / (vA + vB - 2*rho*sqrt(vA*vB)) uses both
        variances and IS the right prediction. Comparing it against the measured
        factor is a real check, because the numerator comes from the CRN arm
        while the measured numerator comes from the independent arm: they agree
        only if CRN leaves each estimator's marginal variance alone AND the
        independent arm really is independent.

    `indep_rho` is the direct control on that last point: allocations in the
    independent arm should be uncorrelated across replicates, so it should sit
    near zero. If it does not, the "independent" seeds are not independent and
    the whole comparison is void.

    Cost is `2 * n_replicates * len(candidates)` cell runs, so keep the candidate
    list to a handful; the pair count grows quadratically at no extra simulation
    cost.
    """
    try:
        from .sweep_seeding import independent_seed
    except ImportError:
        from sweep_seeding import independent_seed

    n_c = len(candidates)
    if n_c < 2:
        raise ValueError("need at least 2 candidates to form a pair")
    if n_replicates < 3:
        raise ValueError("need at least 3 replicates for a variance estimate")

    means_crn = np.empty((n_replicates, n_c))
    means_ind = np.empty((n_replicates, n_c))
    done, total = 0, 2 * n_replicates * n_c
    for r in range(n_replicates):
        shared = crn_seed(scenario.scenario_id, master_seed, r)
        for j, cand in enumerate(candidates):
            alloc = cand.as_dict(inputs.channels)
            means_crn[r, j] = summarize_cell(
                cand.allocation_id, scenario.scenario_id, alloc, inputs, scenario, shared
            )["mean_revenue"]
            means_ind[r, j] = summarize_cell(
                cand.allocation_id, scenario.scenario_id, alloc, inputs, scenario,
                independent_seed(cand.allocation_id, scenario.scenario_id, master_seed, r),
            )["mean_revenue"]
            done += 2
            if progress is not None:
                progress(done, total)

    pairs = list(itertools.combinations(range(n_c), 2))
    rows = []
    for i, j in pairs:
        d_crn = means_crn[:, i] - means_crn[:, j]
        d_ind = means_ind[:, i] - means_ind[:, j]
        v_crn = float(d_crn.var(ddof=1))
        v_ind = float(d_ind.var(ddof=1))
        rho = float(np.corrcoef(means_crn[:, i], means_crn[:, j])[0, 1])
        rho_ind = float(np.corrcoef(means_ind[:, i], means_ind[:, j])[0, 1])
        va = float(means_crn[:, i].var(ddof=1))
        vb = float(means_crn[:, j].var(ddof=1))
        denom = va + vb - 2.0 * rho * math.sqrt(va * vb)
        rows.append({
            "allocation_a": candidates[i].allocation_id,
            "allocation_b": candidates[j].allocation_id,
            "mean_diff_crn": float(d_crn.mean()),
            "mean_diff_indep": float(d_ind.mean()),
            "sd_diff_crn": math.sqrt(v_crn),
            "sd_diff_indep": math.sqrt(v_ind),
            "variance_reduction_factor": v_ind / v_crn if v_crn > 0 else float("inf"),
            "crn_rho": rho,
            "indep_rho": rho_ind,
            "vrf_predicted": (va + vb) / denom if denom > 0 else float("inf"),
            "vrf_upper_bound_from_rho": 1.0 / (1.0 - rho) if rho < 1 else float("inf"),
        })
    pair_df = pd.DataFrame(rows)
    return {
        "pairs": pair_df,
        "n_replicates": n_replicates,
        "n_candidates": n_c,
        "n_pairs": len(pairs),
        "median_vrf": float(pair_df["variance_reduction_factor"].median()),
        "mean_vrf": float(pair_df["variance_reduction_factor"].mean()),
        "min_vrf": float(pair_df["variance_reduction_factor"].min()),
        "max_vrf": float(pair_df["variance_reduction_factor"].max()),
        "median_rho": float(pair_df["crn_rho"].median()),
        "median_indep_rho": float(pair_df["indep_rho"].median()),
        "max_abs_indep_rho": float(pair_df["indep_rho"].abs().max()),
        "median_sd_diff_crn": float(pair_df["sd_diff_crn"].median()),
        "median_sd_diff_indep": float(pair_df["sd_diff_indep"].median()),
        "means_crn": means_crn,
        "means_indep": means_ind,
    }
