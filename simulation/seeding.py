"""
Deterministic per-cell seeding for the Phase 3 distributed batch.

THE PROBLEM
===========
Phase 3 runs ~21 allocations x 4 scenarios = ~84 independent Monte Carlo cells,
scattered across Spark executors in whatever order the scheduler decides. Two
requirements pull against each other:

  (a) REPRODUCIBILITY -- rerunning the job, on a different cluster, with a
      different partition count, must give byte-identical results. So the seed
      for a cell has to be a pure function of that cell's identity, never of
      partition index, task id, wall clock, or arrival order.

  (b) INDEPENDENCE -- different cells must draw statistically INDEPENDENT
      streams. This is not cosmetic. If every cell reused one stream, then two
      allocations would see the identical sequence of CVR/CPC/RPC shocks, and
      the difference between their revenue distributions would contain zero
      sampling noise in the common part while the LEVEL of each would still be
      wrong by a common amount. Comparing allocations -- which is the entire
      point of Phase 4's frontier -- would then be comparing two draws of the
      same random world, producing an artificial cross-allocation correlation
      of ~1 and risk contrasts that are an artifact of one shared sample path
      rather than a property of the allocations. Antithetic or common-random-
      numbers designs deliberately exploit that coupling, but they require the
      variance analysis to account for it; we are not doing that here, so the
      streams must simply be independent.

WHY NOT `hash()`
================
Python's builtin `hash()` of a str is SALTED PER PROCESS (PYTHONHASHSEED is
random by default). It would give different seeds on every run, and -- worse --
different seeds on different Spark executors within the SAME run, so the job
would be non-reproducible in a way that is invisible in the output. It is never
used here.

THE SCHEME
==========
    blake2b(allocation_id) -> stable 64-bit int   \
    blake2b(scenario_id)   -> stable 64-bit int    >-- entropy tuple
    master_seed (project constant)                /

    numpy.random.SeedSequence(entropy=[master, h_alloc, h_scen])
        .generate_state(1, dtype=uint64)  ->  the cell's seed

blake2b is a cryptographic hash: stable across processes, machines, Python
versions and platforms, with strong avalanche, so `heavy_display` and
`heavy_video` produce entropy words with no exploitable relationship.

SeedSequence is then doing the load-bearing work, and that is deliberate:
independence here is a DOCUMENTED PROPERTY OF NUMPY, not something hand-rolled.
SeedSequence exists precisely to take arbitrary user entropy of arbitrary
quality and run it through a hashing-and-mixing construction that yields
initial states which are, to the limit of testing, statistically independent
for distinct entropy inputs. Concatenating ids into one seed integer, XOR-ing
hashes together, or adding an offset to the master seed would all be exactly
the kind of hand-rolled scheme SeedSequence was written to replace.

The returned 64-bit integer is then handed to `np.random.default_rng(seed)`
inside the engine, which re-runs it through SeedSequence to initialise PCG64.
Double mixing is harmless.

COLLISIONS
==========
Seeds are 64-bit. For n cells the birthday collision probability is about
n^2 / 2^65; at n = 84 that is ~1.9e-16. `assert_distinct_seeds` checks the
actual grid anyway, because a silent collision would mean two cells sharing a
stream -- exactly the failure mode this module exists to prevent.

THE PHASE 2 ANCHOR -- A REAL DECISION, NOT A DETAIL
===================================================
Phase 2's verified result ($968,027 / $169,364 / $715,966) was produced at seed
20260808 for the even-split / normal cell. `derive_seed("even_split", "normal")`
does NOT return 20260808 -- it returns a hash-derived value -- so a Phase 3 run
seeded purely by derivation would put a DIFFERENT number in that silver row
(same distribution, different sample; expect the mean to differ by a few
thousand dollars, on the order of the ~$1,700 Monte Carlo standard error).

DECISION (2026-08-13): the grid is UNIFORMLY DERIVED. `pin_phase2_anchor`
defaults to False, so no cell -- not even the anchor -- is special-cased, and
the silver row for (even_split, normal) carries its derived-seed draw rather
than Phase 2's pinned one.

The reasoning is about what each comparison is allowed to prove. Pinning the
anchor would make silver echo $968,027 exactly, but that number would then be
a restatement of Phase 2's seed, not independent evidence. Keeping the grid
uniform separates two genuinely different claims:

  1. "wrapping and distributing the engine did not change the math" -- proven
     EXACTLY, by running the anchor cell at seed 20260808 and comparing the
     revenue vector bitwise against `engine.simulate()`. That is a side test,
     not a grid cell.
  2. "the Monte Carlo estimate is stable across seeds" -- proven
     STATISTICALLY, by checking the derived-seed anchor lands within Monte
     Carlo error of Phase 2's mean.

A pinned anchor would collapse (1) and (2) into one number and prove neither
cleanly. `pin_phase2_anchor=True` remains available for anyone who wants the
old behaviour, but nothing in Phase 3 passes it.
"""

from __future__ import annotations

import hashlib

import numpy as np

try:  # package-style import (simulation.seeding)
    from .config import RANDOM_SEED
except ImportError:  # flat import, which is how the rest of this repo runs
    from config import RANDOM_SEED


# Single source of truth: the Phase 3 master seed IS the Phase 2 seed, so the
# whole project has one number to change if a full resample is ever wanted.
MASTER_SEED: int = RANDOM_SEED

# The cell whose numbers Phase 2 verified, and the seed that produced them.
PHASE2_ANCHOR_CELL: tuple[str, str] = ("even_split", "normal")
PHASE2_ANCHOR_SEED: int = RANDOM_SEED

_DIGEST_BYTES = 8  # 64-bit entropy word per id


def stable_hash_int(text: str) -> int:
    """Process-stable 64-bit integer for a string id.

    blake2b rather than `hash()`: `hash()` is salted per process and would make
    the whole batch silently non-reproducible across runs and across executors.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str id, got {type(text).__name__}: {text!r}")
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big")


def derive_seed(
    allocation_id: str, scenario_id: str, master_seed: int = MASTER_SEED
) -> int:
    """Seed for one (allocation_id, scenario_id) cell.

    Pure: no special cases, no state, no environment dependence. The same three
    inputs always give the same 64-bit integer, on any machine, in any process.
    """
    seq = np.random.SeedSequence(
        entropy=[int(master_seed), stable_hash_int(allocation_id), stable_hash_int(scenario_id)]
    )
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def seed_for_cell(
    allocation_id: str,
    scenario_id: str,
    master_seed: int = MASTER_SEED,
    pin_phase2_anchor: bool = False,
) -> int:
    """Seed a Phase 3 runner should actually use for a cell.

    Defaults to pure derivation -- identical to `derive_seed`, no privileged
    cell. Phase 3 relies on that default; see the module docstring for why a
    uniformly derived grid proves more than a pinned anchor would.

    `pin_phase2_anchor=True` restores the old behaviour, pinning
    ("even_split", "normal") to `PHASE2_ANCHOR_SEED` so that cell replays
    Phase 2's exact draw. Nothing in Phase 3 passes it.
    """
    if pin_phase2_anchor and (allocation_id, scenario_id) == PHASE2_ANCHOR_CELL:
        return PHASE2_ANCHOR_SEED
    return derive_seed(allocation_id, scenario_id, master_seed)


def seed_grid(
    allocation_ids: list[str],
    scenario_ids: list[str],
    master_seed: int = MASTER_SEED,
    pin_phase2_anchor: bool = False,
) -> dict[tuple[str, str], int]:
    """Every cell's seed, keyed by (allocation_id, scenario_id).

    Convenience for a driver that wants to materialise the plan (and its seeds)
    as a Spark DataFrame before fanning out, so the seeds are visible in the
    job's own lineage rather than recomputed inside a UDF.
    """
    grid = {
        (a, s): seed_for_cell(a, s, master_seed, pin_phase2_anchor)
        for a in allocation_ids
        for s in scenario_ids
    }
    assert_distinct_seeds(grid)
    return grid


def assert_distinct_seeds(grid: dict[tuple[str, str], int]) -> None:
    """Raise if any two cells share a seed.

    Astronomically unlikely at 64 bits, and catastrophic if it happened: two
    cells would share a stream and their comparison would be meaningless.
    """
    seen: dict[int, tuple[str, str]] = {}
    for cell, seed in grid.items():
        if seed in seen:
            raise AssertionError(
                f"seed collision: cells {seen[seed]} and {cell} both got seed {seed}"
            )
        seen[seed] = cell
