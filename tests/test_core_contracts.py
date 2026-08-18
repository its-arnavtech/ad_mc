import math

import numpy as np

from simulation.allocations import build_allocation_candidates
from simulation.frontier import (
    assert_candidate_set_valid,
    generate_stage1,
    non_dominated_mask,
)
from simulation.saturation import apply_spend_floor
from simulation.seeding import derive_seed, seed_grid


CHANNELS = ["display", "paid_search", "paid_social", "programmatic", "video"]


def test_phase3_grid_is_deterministic_and_exact():
    first = build_allocation_candidates(CHANNELS, 500_000.0)
    second = build_allocation_candidates(CHANNELS, 500_000.0)

    assert first == second
    assert len(first) == 21
    assert all(math.fsum(spend.values()) == 500_000.0 for spend in first.values())
    assert first["even_split"] == {channel: 100_000.0 for channel in CHANNELS}


def test_stage1_candidate_contract_is_reproducible():
    first = generate_stage1(CHANNELS, master_seed=20260808)
    second = generate_stage1(CHANNELS, master_seed=20260808)

    assert [(c.allocation_id, c.spend) for c in first] == [
        (c.allocation_id, c.spend) for c in second
    ]
    stats = assert_candidate_set_valid(first, CHANNELS, 500_000.0)
    assert stats["n_candidates"] == 571
    assert stats["n_with_a_zero_channel"] == 107


def test_seed_grid_is_stable_distinct_and_domain_separated():
    allocations = ["even_split", "dominant_paid_search"]
    scenarios = ["normal", "recession"]
    first = seed_grid(allocations, scenarios, master_seed=20260808)
    second = seed_grid(allocations, scenarios, master_seed=20260808)

    assert first == second
    assert len(set(first.values())) == len(first)
    assert derive_seed("a", "b", 20260808) != derive_seed("b", "a", 20260808)


def test_spend_floor_preserves_zero_and_clips_only_positive_subfloor_spend():
    spend = np.array([0.0, 50.0, 100.0, 150.0])
    floor = np.array([100.0, 100.0, 100.0, 100.0])

    clipped, mask = apply_spend_floor(spend, floor)

    np.testing.assert_array_equal(clipped, [0.0, 100.0, 100.0, 150.0])
    np.testing.assert_array_equal(mask, [False, True, False, False])


def test_non_dominated_mask_respects_mixed_objective_directions():
    # Return is maximized; risk is minimized. The last row is dominated by row 1.
    objectives = np.array([[10.0, 5.0], [9.0, 4.0], [8.0, 6.0]])
    np.testing.assert_array_equal(
        non_dominated_mask(objectives, maximize=[True, False]),
        [True, True, False],
    )
