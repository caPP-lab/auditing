from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

import auditcs.categorical_ppr_linear as categorical_ppr_linear
from auditcs.categorical_ppr_linear import (
    LinearMatrixScore,
    brute_force_running_row_endpoint,
    cumulative_histories,
    project_linear_matrix_score,
    solve_linear_ppr_row_endpoint,
)


def test_master_rechecks_presolve_infeasibility(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    presolve_infeasible = SimpleNamespace(status=2)
    verified_result = SimpleNamespace(status=0)

    def fake_milp(*args, **kwargs):
        del args
        calls.append(dict(kwargs["options"]))
        return presolve_infeasible if len(calls) == 1 else verified_result

    monkeypatch.setattr(categorical_ppr_linear, "_milp", fake_milp)
    result = categorical_ppr_linear._solve_master(
        np.asarray([0.0, 1.0]),
        np.zeros(2),
        np.full(2, 4.0),
        4,
        [],
        [],
        time_limit=5.0,
    )
    assert result is verified_result
    assert len(calls) == 2
    assert calls[1]["presolve"] is False


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_certified_endpoint_matches_tiny_bruteforce(seed: int, side: str) -> None:
    rng = np.random.default_rng(seed)
    categories = 3 + seed % 2
    row_size = 8 + seed % 4
    audits = 2 + seed % 4
    destinations = rng.integers(0, categories, size=audits)
    histories = cumulative_histories(destinations, categories)
    costs = rng.integers(-2, 4, size=categories)
    expected, _ = brute_force_running_row_endpoint(
        row_size,
        histories,
        costs,
        alpha_row=0.05,
        side=side,
    )
    actual = solve_linear_ppr_row_endpoint(
        row_size,
        histories,
        costs,
        alpha_row=0.05,
        side=side,
    )
    assert actual.status == "exact"
    assert actual.integer_gap == 0.0
    assert actual.integer_numerator == expected
    assert costs @ actual.witness_counts == expected
    assert actual.witness_counts.sum() == row_size
    assert actual.minimum_ppr_slack > 0.0


def test_running_intersection_endpoints_are_nested_and_collapse_at_census() -> None:
    destinations = np.asarray([0, 1, 0, 2, 1, 0, 2, 0])
    histories = cumulative_histories(destinations, 3)
    costs = np.asarray([0, 1, 2])
    intervals = []
    for audits in range(destinations.size + 1):
        prefix = histories[:audits]
        lower = solve_linear_ppr_row_endpoint(
            destinations.size, prefix, costs, alpha_row=0.05, side="lower"
        )
        upper = solve_linear_ppr_row_endpoint(
            destinations.size, prefix, costs, alpha_row=0.05, side="upper"
        )
        intervals.append((lower.integer_numerator, upper.integer_numerator))
    assert all(
        current[0] >= previous[0] and current[1] <= previous[1]
        for previous, current in zip(intervals, intervals[1:])
    )
    truth = int(costs @ np.bincount(destinations, minlength=3))
    assert intervals[-1] == (truth, truth)


def test_constant_score_zero_cells_and_no_audits() -> None:
    histories = cumulative_histories([0, 0], 3)
    for prefix in (histories[:0], histories):
        for side in ("lower", "upper"):
            result = solve_linear_ppr_row_endpoint(
                7, prefix, [4, 4, 4], alpha_row=0.01, side=side
            )
            assert result.integer_numerator == 28
            assert result.status == "exact"


def test_singleton_category_collapses_for_every_look() -> None:
    histories = cumulative_histories([0, 0, 0], 1)
    for audits in range(4):
        result = solve_linear_ppr_row_endpoint(
            6, histories[:audits], [3], alpha_row=0.05, side="upper"
        )
        assert result.integer_numerator == 18
        assert np.array_equal(result.witness_counts, [6])


def test_matrix_projection_equals_cartesian_row_extrema() -> None:
    row_sizes = np.asarray([5, 4])
    histories = (
        cumulative_histories([0, 1], 2),
        cumulative_histories([1], 2),
    )
    score = LinearMatrixScore(
        score_id="tiny",
        integer_costs=np.asarray([[0, 2], [3, 1]]),
        denominator=9,
        description="tiny fixture",
        higher_is_better=False,
    )
    result = project_linear_matrix_score(row_sizes, histories, score, alpha=0.05)
    row_sets = []
    for row, size in enumerate(row_sizes):
        lower, _ = brute_force_running_row_endpoint(
            int(size),
            histories[row],
            score.integer_costs[row],
            alpha_row=0.025,
            side="lower",
        )
        upper, _ = brute_force_running_row_endpoint(
            int(size),
            histories[row],
            score.integer_costs[row],
            alpha_row=0.025,
            side="upper",
        )
        row_sets.append(range(lower, upper + 1))
    joint_values = [sum(values) / score.denominator for values in product(*row_sets)]
    assert result.lower == min(joint_values)
    assert result.upper == max(joint_values)


def test_score_validation_and_influences() -> None:
    score = LinearMatrixScore(
        score_id="binary",
        integer_costs=np.asarray([[0, 1], [0, 0]]),
        denominator=10,
        description="binary",
        higher_is_better=False,
    )
    assert np.allclose(score.row_influences([4, 6]), [0.2, 0.0])
    with pytest.raises(ValueError):
        LinearMatrixScore("bad", np.asarray([0, 1]), 1, "bad", False)
