"""
Common-random-numbers (CRN) seeding for the Phase 4 optimization sweep.

THIS DELIBERATELY CONTRADICTS `seeding.py`. READ BOTH.
======================================================
Phase 3 gave every (allocation, scenario) cell its own independent stream, and
`seeding.py` argues for that at length. Phase 4 does the opposite WITHIN a
scenario: every allocation draws the same underlying randomness. That is not a
correction of Phase 3, it is a different design for a different question, and
the reason is worth stating precisely because the two modules read as if they
disagree.

WHAT EACH DESIGN IS GOOD AT
---------------------------
Phase 3 asks "what is the revenue distribution of allocation A?", one cell at a
time. Independent streams are the right default: nothing is being differenced,
and independence means a bug in one cell cannot correlate itself into another.

Phase 4 asks "does allocation A beat allocation B?", ~800 x 799 / 2 times per
scenario. That is a question about the DIFFERENCE, and the difference is where
independence hurts. Write mu_A for the true mean and X_A for the 10,000-path
estimate:

    independent seeds:  Var(X_A - X_B) = Var(X_A) + Var(X_B)
    shared seeds:       Var(X_A - X_B) = Var(X_A) + Var(X_B) - 2 Cov(X_A, X_B)

With independent seeds the standard error of the difference is about $2,400 at
n = 10,000 on this model. Frontier-adjacent allocations are separated by far
less than that in places, so pairwise dominance -- which is exactly what a Pareto
front is computed from -- would be decided by sampling noise, and the reported
frontier would change every time the master seed changed. Sharing the stream
makes Cov(X_A, X_B) strongly positive (both allocations see the same CVR shocks,
the same auction shocks, the same order-value shocks) and most of that noise
cancels out of the difference. The measured factor is reported by
`run_phase4_frontier.py`; it is MEASURED on a subset run both ways, never
asserted.

WHAT CRN DOES NOT DO -- THE HONEST CAVEATS
------------------------------------------
1. It does not reduce the error on any single allocation's LEVEL. Each X_A still
   has its full ~$1,700 standard error, and the whole frontier is drawn from one
   random world, so the whole thing shifts together if the master seed changes.
   Differences are precise; levels are not. A reported mean of $1,011,028 is
   still +/- ~$1,700, and nothing here changes that.
2. It makes the estimates across allocations statistically DEPENDENT. Any
   downstream analysis that treats the ~800 rows of a scenario as independent
   observations -- fitting a curve through them and quoting a standard error,
   say -- is wrong under CRN, and would have been right under Phase 3's scheme.
   This is the real cost, and it is the thing `seeding.py` was warning about.
3. It is NOT applied ACROSS scenarios. Each scenario derives its own seed, so
   the four scenarios are independent worlds and a normal-vs-recession contrast
   carries the full unpaired error. That is deliberate: scenario contrasts are
   read as separate what-ifs, not as a paired experiment, and keeping them
   independent means a fluke in one scenario cannot masquerade as a pattern
   across all four.
4. CRN's variance reduction relies on the two allocations responding
   MONOTONICALLY to the same shocks. Here they do, structurally: revenue is
   increasing in every CVR and RPC draw and decreasing in every CPC draw for
   every allocation, so the induced correlation is positive by construction, not
   by luck. Antithetic-style pathologies (where CRN can increase variance) need
   an inverse response somewhere, and this funnel has none.

HOW THE SHARING ACTUALLY WORKS -- AND ITS ONE SUBTLETY
------------------------------------------------------
`engine.simulate` draws in a fixed order: correlated standard normals for the
CVR copula, then one lognormal block per channel for CPC, then one per channel
for RPC. NONE of that consumption depends on the allocation -- the counts and the
order are functions of `n_paths` and `len(channels)` only. So passing the same
seed to two allocations gives them the same underlying stream, and:

  * the CVR matrix is BITWISE IDENTICAL across allocations (the copula uniforms
    and the Beta parameters are both allocation-free);
  * the RPC matrix is BITWISE IDENTICAL (its moments are allocation-free);
  * the CPC matrix is NOT identical once saturation is on, and should not be --
    the whole point of the curve is that a bigger position pays a higher mean
    CPC. What is shared is the underlying standard normal: numpy draws a
    lognormal as `exp(mu + sigma * Z)`, and Z depends only on the stream
    position, so the two allocations see the same Z and differ only through
    their own (mu, sigma). That is textbook inverse-transform CRN, not a
    weakened version of it. With `theta=None` the CPC matrices are bitwise
    identical too.
  * an IDLE channel (zero spend) still consumes its own CPC and RPC block, so
    allocations with different support stay aligned on the stream. See
    `saturation.effective_cpc_moments`.

`validate_crn_sharing` in `frontier.py` checks all of this on real draws rather
than trusting the argument.

THE SCHEME
==========
    blake2b(scenario_id)   -> stable 64-bit int
    blake2b(CRN_STREAM_LABEL) -> stable 64-bit int   (domain separation)
    master_seed, replicate

    SeedSequence(entropy=[master, h_scenario, h_label, replicate])
        .generate_state(1, dtype=uint64)  ->  the scenario's shared seed

Same primitives as `seeding.py` (blake2b for process-stable hashing, numpy's
SeedSequence to do the actual mixing) so there is one hashing convention in the
project. Two additions:

  * CRN_STREAM_LABEL is domain separation. Without it, nothing structurally
    prevents a Phase 4 scenario seed from colliding with some other derivation
    off the same master seed; with it, the Phase 4 sweep occupies its own
    labelled corner of the seed space and `derive_seed(a, s)` from Phase 3 and
    `crn_seed(s)` from Phase 4 are guaranteed-distinct constructions rather than
    coincidentally-distinct ones.
  * `replicate` exists so the sweep can be re-run on genuinely independent
    randomness -- replicate 0, 1, 2, ... -- which is the only way to put an
    honest standard error on a CRN result (batched means over replicates). The
    measurement harness uses it, and it is the recommended way to check whether
    a frontier is stable rather than a feature of one lucky world.
"""

from __future__ import annotations

import numpy as np

try:  # package-style import (simulation.sweep_seeding / ad_mc_sim.sweep_seeding)
    from .seeding import MASTER_SEED, stable_hash_int
except ImportError:  # flat import, which is how the rest of this repo runs
    from seeding import MASTER_SEED, stable_hash_int


# Domain separator. Changing this string reseeds the entire Phase 4 sweep, which
# is exactly what it is for: it makes "Phase 4's CRN stream" a named thing rather
# than an unlabelled function of the master seed.
CRN_STREAM_LABEL: str = "phase4-optimization-sweep-crn"

# Label for the deliberately INDEPENDENT arm of the CRN measurement, so the
# control arm cannot accidentally reuse the treatment arm's stream.
INDEPENDENT_STREAM_LABEL: str = "phase4-optimization-sweep-independent"


def crn_seed(
    scenario_id: str,
    master_seed: int = MASTER_SEED,
    replicate: int = 0,
    stream_label: str = CRN_STREAM_LABEL,
) -> int:
    """The seed EVERY allocation in `scenario_id` shares.

    Pure: a function of the scenario id, the master seed, the replicate index and
    the stream label, and of nothing else -- not the allocation, not the
    partition, not the wall clock. Two allocations evaluated in the same scenario
    and the same replicate therefore see the same underlying random stream, which
    is the entire point.
    """
    if not isinstance(replicate, (int, np.integer)) or replicate < 0:
        raise ValueError(f"replicate must be a non-negative int, got {replicate!r}")
    seq = np.random.SeedSequence(
        entropy=[
            int(master_seed),
            stable_hash_int(scenario_id),
            stable_hash_int(stream_label),
            int(replicate),
        ]
    )
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def independent_seed(
    allocation_id: str,
    scenario_id: str,
    master_seed: int = MASTER_SEED,
    replicate: int = 0,
    stream_label: str = INDEPENDENT_STREAM_LABEL,
) -> int:
    """Phase-3-style per-cell seed: a different stream for every allocation.

    Only used as the CONTROL ARM of the CRN measurement -- run the same
    allocations both ways and compare the variance of the difference. Not used by
    the sweep itself.
    """
    if not isinstance(replicate, (int, np.integer)) or replicate < 0:
        raise ValueError(f"replicate must be a non-negative int, got {replicate!r}")
    seq = np.random.SeedSequence(
        entropy=[
            int(master_seed),
            stable_hash_int(allocation_id),
            stable_hash_int(scenario_id),
            stable_hash_int(stream_label),
            int(replicate),
        ]
    )
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def crn_seed_plan(
    scenario_ids: list[str],
    master_seed: int = MASTER_SEED,
    replicate: int = 0,
) -> dict[str, int]:
    """scenario_id -> shared seed, with the distinctness check run.

    Materialise this on the driver and carry it into the plan DataFrame so the
    seeds are visible in the job's lineage instead of being recomputed inside a
    UDF. Every allocation row for a scenario carries that scenario's seed.
    """
    plan = {s: crn_seed(s, master_seed, replicate) for s in scenario_ids}
    assert_scenarios_distinct(plan)
    return plan


def assert_scenarios_distinct(plan: dict[str, int]) -> None:
    """Raise if two SCENARIOS share a seed.

    Allocations sharing a seed is the design. Scenarios sharing one is not: it
    would silently pair the four scenarios as well, so a normal-vs-recession
    contrast would inherit CRN's correlation without the analysis accounting for
    it. At 64 bits a genuine collision is ~1e-18 likely for four scenarios, so in
    practice this fires only if the derivation has been broken.
    """
    seen: dict[int, str] = {}
    for scenario_id, seed in plan.items():
        if seed in seen:
            raise AssertionError(
                f"CRN seed collision: scenarios {seen[seed]!r} and {scenario_id!r} both "
                f"got seed {seed}; the four scenarios must stay independent worlds"
            )
        seen[seed] = scenario_id
