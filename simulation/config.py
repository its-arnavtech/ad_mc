"""
Phase 2 simulation configuration.

Everything the run is parameterised by lives here rather than as literals in
the engine, because Phase 4 varies the allocation to trace an efficient
frontier and will drive these directly.
"""

from __future__ import annotations

# --- Budget and allocation --------------------------------------------------
TOTAL_BUDGET = 500_000.0
ALLOCATION_STRATEGY = "even_split"

# --- Simulation -------------------------------------------------------------
N_PATHS = 10_000
RANDOM_SEED = 20260808  # fixed so the run is reproducible; change to resample

# --- Scenario ---------------------------------------------------------------
# Phase 2 is "normal" only: no macro multipliers applied anywhere. Scenario
# overlays (recession / boom / seasonal) are Phase 4 work. The constant is
# here so the reporting can state which scenario produced a given number.
SCENARIO = "normal"


def even_split_allocation(channels: list[str], total_budget: float = TOTAL_BUDGET) -> dict[str, float]:
    """Split `total_budget` equally across `channels`.

    Returns a channel -> dollars mapping. Kept as a function (not a frozen
    dict) so the channel list comes from the bronze table rather than being
    hardcoded -- if a sixth channel appears, this keeps working.
    """
    if not channels:
        raise ValueError("no channels to allocate across")
    per_channel = total_budget / len(channels)
    return {c: per_channel for c in channels}


def build_allocation(channels: list[str], strategy: str = ALLOCATION_STRATEGY,
                     total_budget: float = TOTAL_BUDGET) -> dict[str, float]:
    if strategy == "even_split":
        return even_split_allocation(channels, total_budget)
    raise ValueError(f"unknown allocation strategy: {strategy!r}")
