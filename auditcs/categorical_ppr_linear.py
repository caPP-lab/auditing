"""Exact linear projection of running categorical-PPR row confidence sets.

The CIFAR-scale solver uses a continuous solution only to generate globally
valid tangents.  Exactness comes from a zero-gap integer outer-approximation
master together with an integer PPR-feasible witness attaining the same score.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from time import perf_counter
from typing import Literal, Sequence
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp as scipy_milp, minimize
from scipy.special import digamma, gammaln


EndpointSide = Literal["lower", "upper"]


def _milp(*args, **kwargs):
    """Run one deterministic single-threaded HiGHS MILP.

    Phase-level parallelism is across independent audit-order seeds.  Limiting
    each inner solve to one HiGHS thread avoids severe process oversubscription
    and makes the per-row time gate interpretable.
    """

    options = dict(kwargs.pop("options", {}))
    options.setdefault("threads", 1)
    options.setdefault("output_flag", False)
    options.setdefault("log_to_console", False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        return scipy_milp(*args, options=options, **kwargs)


def _milp_with_verified_infeasibility(*args, **kwargs):
    """Never use a presolve-only infeasibility result as a proof.

    The endpoint certificate closes gaps by proving an integer outer master or
    objective slice infeasible.  HiGHS presolve can rarely make a false
    infeasibility declaration on these scaled tangent systems.  Re-solving the
    identical model with presolve disabled makes that proof step independent
    of the problematic reduction.  If the verification is limited or fails,
    its non-infeasible status is returned and the caller must remain
    indeterminate rather than certify the endpoint.
    """

    options = dict(kwargs.pop("options", {}))
    started = perf_counter()
    result = _milp(*args, options=options, **kwargs)
    if result.status != 2:
        return result
    time_limit = float(options.get("time_limit", np.inf))
    remaining = (
        time_limit - (perf_counter() - started)
        if np.isfinite(time_limit)
        else time_limit
    )
    verification_options = dict(options)
    verification_options["presolve"] = False
    if np.isfinite(remaining):
        verification_options["time_limit"] = max(0.01, remaining)
    return _milp(*args, options=verification_options, **kwargs)


@dataclass(frozen=True)
class LinearMatrixScore:
    score_id: str
    integer_costs: np.ndarray
    denominator: int
    description: str
    higher_is_better: bool

    def __post_init__(self) -> None:
        costs = np.asarray(self.integer_costs)
        if costs.ndim != 2 or costs.shape[0] == 0:
            raise ValueError("integer_costs must be a nonempty matrix")
        if not np.issubdtype(costs.dtype, np.integer):
            raise ValueError("integer_costs must contain integers")
        if self.denominator <= 0:
            raise ValueError("denominator must be positive")
        frozen = np.array(costs, dtype=int, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "integer_costs", frozen)

    def evaluate_counts(self, confusion: np.ndarray) -> float:
        confusion = np.asarray(confusion, dtype=int)
        if confusion.shape != self.integer_costs.shape:
            raise ValueError("confusion shape does not match score costs")
        return float(np.sum(self.integer_costs * confusion) / self.denominator)

    def row_influences(self, row_sizes: Sequence[int]) -> np.ndarray:
        """Return sharp full-catalog row transfer coefficients for planning."""

        sizes = np.asarray(row_sizes, dtype=float)
        if sizes.shape != (self.integer_costs.shape[0],):
            raise ValueError("row_sizes shape does not match score")
        ranges = np.ptp(self.integer_costs, axis=1).astype(float)
        return sizes * ranges / (2.0 * self.denominator)


@dataclass(frozen=True)
class RowEndpointCertificate:
    side: EndpointSide
    status: str
    integer_numerator: int
    witness_counts: np.ndarray
    master_bound: float
    integer_gap: float
    minimum_ppr_slack: float
    history_count: int
    continuous_success: bool
    continuous_iterations: int
    master_solves: int
    tangent_cuts: int
    milp_nodes: int
    candidate_count: int
    runtime_seconds: float


@dataclass(frozen=True)
class MatrixProjectionCertificate:
    lower: float
    upper: float
    width: float
    lower_numerator: int
    upper_numerator: int
    lower_rows: tuple[RowEndpointCertificate, ...]
    upper_rows: tuple[RowEndpointCertificate, ...]
    runtime_seconds: float


def cumulative_histories(destinations: Sequence[int], categories: int) -> np.ndarray:
    destinations = np.asarray(destinations, dtype=int)
    if categories <= 0:
        raise ValueError("categories must be positive")
    if np.any((destinations < 0) | (destinations >= categories)):
        raise ValueError("destination outside category vocabulary")
    histories = np.zeros((destinations.size, categories), dtype=int)
    running = np.zeros(categories, dtype=int)
    for look, destination in enumerate(destinations):
        running[int(destination)] += 1
        histories[look] = running
    return histories


def _history_constants(
    row_size: int,
    histories: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    audits = histories.sum(axis=1)
    beta_sum = float(beta.sum())
    return (
        gammaln(row_size + 1)
        - gammaln(row_size - audits + 1)
        + gammaln(beta_sum)
        - gammaln(beta_sum + audits)
        + np.sum(gammaln(beta[None, :] + histories) - gammaln(beta), axis=1)
    )


def _ppr_values_and_jacobian(
    candidate: np.ndarray,
    histories: np.ndarray,
    constants: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate, dtype=float)
    residual = candidate[None, :] - histories
    values = constants + np.sum(
        gammaln(residual + 1) - gammaln(candidate[None, :] + 1), axis=1
    )
    jacobian = digamma(residual + 1) - digamma(candidate[None, :] + 1)
    return values, jacobian


def _append_tangent(
    rows: list[np.ndarray],
    uppers: list[float],
    seen: set[tuple[int, ...]],
    point: np.ndarray,
    value: float,
    gradient: np.ndarray,
    threshold: float,
) -> bool:
    upper = threshold - value + float(gradient @ point)
    scale = float(np.max(np.abs(gradient), initial=0.0))
    if scale > 0.0:
        gradient = np.asarray(gradient, dtype=float) / scale
        upper /= scale
    key = tuple(
        np.rint(np.r_[np.asarray(gradient) * 1e12, upper * 1e12]).astype(np.int64)
    )
    if key in seen:
        return False
    seen.add(key)
    rows.append(np.asarray(gradient, dtype=float))
    uppers.append(float(upper))
    return True


def _append_boundary_tangent(
    rows: list[np.ndarray],
    uppers: list[float],
    seen: set[tuple[int, ...]],
    *,
    interior: np.ndarray,
    violating: np.ndarray,
    observed: np.ndarray,
    constant: float,
    threshold: float,
) -> bool:
    """Add a strong supporting cut at the segment's constraint boundary."""

    def evaluate(point: np.ndarray) -> tuple[float, np.ndarray]:
        residual = point - observed
        value = constant + np.sum(
            gammaln(residual + 1) - gammaln(point + 1)
        )
        gradient = digamma(residual + 1) - digamma(point + 1)
        return float(value), np.asarray(gradient, dtype=float)

    interior_value, _ = evaluate(interior)
    violating_value, violating_gradient = evaluate(violating)
    if interior_value > threshold + 1e-7 or violating_value <= threshold:
        return _append_tangent(
            rows,
            uppers,
            seen,
            violating,
            violating_value,
            violating_gradient,
            threshold,
        )
    low, high = 0.0, 1.0
    direction = violating - interior
    for _ in range(80):
        midpoint = (low + high) / 2.0
        value, _ = evaluate(interior + midpoint * direction)
        if value <= threshold:
            low = midpoint
        else:
            high = midpoint
    boundary = interior + low * direction
    value, gradient = evaluate(boundary)
    return _append_tangent(
        rows,
        uppers,
        seen,
        boundary,
        value,
        gradient,
        threshold,
    )


def _solve_master(
    objective: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    row_size: int,
    tangent_rows: list[np.ndarray],
    tangent_uppers: list[float],
    *,
    time_limit: float,
):
    rows = [np.ones(objective.size, dtype=float), *tangent_rows]
    lows = [float(row_size), *([-np.inf] * len(tangent_rows))]
    highs = [float(row_size), *tangent_uppers]
    return _milp_with_verified_infeasibility(
        objective,
        integrality=np.ones(objective.size, dtype=int),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            np.asarray(rows), np.asarray(lows), np.asarray(highs)
        ),
        options={"mip_rel_gap": 0.0, "time_limit": max(0.01, time_limit)},
    )


def _certify_small_integer_master_gap(
    master,
    *,
    objective: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    row_size: int,
    tangent_rows: list[np.ndarray],
    tangent_uppers: list[float],
    time_limit: float,
    maximum_gap_units: int = 8,
) -> tuple[np.ndarray | None, int, bool]:
    """Close a small numerical MILP gap by exact objective-slice feasibility.

    HiGHS can report optimal status for an integer objective while retaining a
    sub-unit numerical primal/dual gap.  The score lattice lets us close that
    gap without relaxing certification: every integer objective between the
    rounded dual bound and incumbent is tested as a separate integer
    feasibility problem.  Infeasibility of all better slices, or feasibility
    of the first slice above the dual bound, proves the returned master bound.
    """

    # SciPy marks a time-limited MILP as ``success=False`` even when HiGHS
    # returns a valid integer incumbent and a finite dual bound.  Those two
    # objects are sufficient for the exact objective-slice certificate below;
    # the time-limited status itself is never treated as optimality.
    if master.x is None or not np.all(np.isfinite(master.x)):
        return None, 0, False
    dual_bound = getattr(master, "mip_dual_bound", None)
    if dual_bound is None or not np.isfinite(dual_bound):
        return None, 0, False
    incumbent = np.rint(master.x).astype(int)
    incumbent_objective = int(np.rint(objective @ incumbent))
    lower_objective = int(np.ceil(float(dual_bound) - 1e-7))
    gap_units = incumbent_objective - lower_objective
    if gap_units < 0 or gap_units > maximum_gap_units:
        return None, 0, False
    if gap_units == 0:
        return incumbent, 0, True

    started = perf_counter()
    nodes = 0
    rows = [np.ones(objective.size, dtype=float), *tangent_rows, objective]
    lows = [float(row_size), *([-np.inf] * len(tangent_rows)), 0.0]
    highs = [float(row_size), *tangent_uppers, 0.0]
    for candidate_objective in range(lower_objective, incumbent_objective):
        remaining = time_limit - (perf_counter() - started)
        if remaining <= 0:
            return None, nodes, False
        lows[-1] = float(candidate_objective)
        highs[-1] = float(candidate_objective)
        result = _milp_with_verified_infeasibility(
            np.zeros(objective.size, dtype=float),
            integrality=np.ones(objective.size, dtype=int),
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(
                np.asarray(rows), np.asarray(lows), np.asarray(highs)
            ),
            options={"mip_rel_gap": 0.0, "time_limit": max(0.01, remaining)},
        )
        nodes += int(result.mip_node_count or 0)
        if result.success:
            counts = np.rint(result.x).astype(int)
            if (
                counts.sum() != row_size
                or int(np.rint(objective @ counts)) != candidate_objective
            ):
                return None, nodes, False
            if tangent_rows:
                lhs = np.asarray(tangent_rows) @ counts
                if np.any(lhs > np.asarray(tangent_uppers) + 1e-7):
                    return None, nodes, False
            return counts, nodes, True
        if result.status != 2:
            return None, nodes, False
    return incumbent, nodes, True


def _round_bounded_fixed_sum(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: int,
) -> np.ndarray:
    """Deterministically round a bounded point while preserving its sum."""

    clipped = np.clip(np.asarray(point, dtype=float), lower, upper)
    counts = np.floor(clipped + 1e-10).astype(int)
    counts = np.maximum(counts, np.ceil(lower - 1e-10).astype(int))
    counts = np.minimum(counts, np.floor(upper + 1e-10).astype(int))
    deficit = int(total - counts.sum())
    if deficit > 0:
        order = np.argsort(-(clipped - counts), kind="stable")
        while deficit:
            eligible = order[counts[order] < np.floor(upper[order] + 1e-10)]
            if eligible.size == 0:
                raise RuntimeError("bounded rounding could not attain the required sum")
            take = eligible[:deficit]
            counts[take] += 1
            deficit -= take.size
    elif deficit < 0:
        order = np.argsort(clipped - counts, kind="stable")
        while deficit:
            eligible = order[counts[order] > np.ceil(lower[order] - 1e-10)]
            if eligible.size == 0:
                raise RuntimeError("bounded rounding could not attain the required sum")
            take = eligible[: -deficit]
            counts[take] -= 1
            deficit += take.size
    return counts


def _nearest_bounded_fixed_sum_roundings(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: int,
) -> list[np.ndarray]:
    """Enumerate the nearest componentwise floor/ceiling integer roundings.

    With ten CIFAR destinations there are at most ``choose(10, 5)=252`` such
    candidates.  Checking all of them against the nonlinear running-history
    constraints often supplies a strict feasible incumbent before the MILP;
    no candidate is trusted without that literal all-history check.
    """

    clipped = np.clip(np.asarray(point, dtype=float), lower, upper)
    base = np.floor(clipped + 1e-10).astype(int)
    base = np.maximum(base, np.ceil(lower - 1e-10).astype(int))
    base = np.minimum(base, np.floor(upper + 1e-10).astype(int))
    deficit = int(total - base.sum())
    if deficit < 0:
        # This should not occur for an in-bounds fixed-sum point, but retain a
        # safe deterministic fallback for numerical edge cases.
        return [_round_bounded_fixed_sum(clipped, lower, upper, total)]
    if deficit == 0:
        return [base]
    eligible = np.flatnonzero(base < np.floor(upper + 1e-10).astype(int))
    if deficit > eligible.size:
        return [_round_bounded_fixed_sum(clipped, lower, upper, total)]
    residual = clipped - base
    choices = list(combinations((int(value) for value in eligible), deficit))
    choices.sort(key=lambda choice: -sum(float(residual[index]) for index in choice))
    candidates: list[np.ndarray] = []
    for choice in choices:
        candidate = base.copy()
        candidate[list(choice)] += 1
        candidates.append(candidate)
    return candidates


def _nearest_fixed_objective_counts(
    point: np.ndarray,
    *,
    integer_costs: np.ndarray,
    target_cost: int,
    lower: np.ndarray,
    upper: np.ndarray,
    row_size: int,
    tangent_rows: list[np.ndarray],
    tangent_uppers: list[float],
    time_limit: float,
    tangent_tolerance: float = 1e-7,
) -> tuple[np.ndarray | None, int]:
    categories = integer_costs.size
    variables = 2 * categories
    objective = np.r_[np.zeros(categories), np.ones(categories)]
    variable_lower = np.r_[lower, np.zeros(categories)]
    variable_upper = np.r_[upper, np.full(categories, np.inf)]
    integrality = np.r_[np.ones(categories, dtype=int), np.zeros(categories, dtype=int)]

    rows: list[np.ndarray] = []
    lows: list[float] = []
    highs: list[float] = []
    for coefficients, target in (
        (np.ones(categories), row_size),
        (integer_costs, target_cost),
    ):
        row = np.zeros(variables)
        row[:categories] = coefficients
        rows.append(row)
        lows.append(float(target))
        highs.append(float(target))
    for tangent, tangent_upper in zip(tangent_rows, tangent_uppers, strict=True):
        row = np.zeros(variables)
        row[:categories] = tangent
        rows.append(row)
        lows.append(-np.inf)
        highs.append(tangent_upper + tangent_tolerance)
    for category in range(categories):
        positive = np.zeros(variables)
        positive[category] = 1.0
        positive[categories + category] = -1.0
        rows.append(positive)
        lows.append(-np.inf)
        highs.append(float(point[category]))

        negative = np.zeros(variables)
        negative[category] = -1.0
        negative[categories + category] = -1.0
        rows.append(negative)
        lows.append(-np.inf)
        highs.append(float(-point[category]))

    result = _milp(
        objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lows), np.asarray(highs)),
        options={"mip_rel_gap": 0.0, "time_limit": max(0.01, time_limit)},
    )
    nodes = int(result.mip_node_count or 0)
    if not result.success:
        return None, nodes
    counts = np.rint(result.x[:categories]).astype(int)
    if counts.sum() != row_size or int(integer_costs @ counts) != target_cost:
        # Large but exact integer objectives (for example a rational
        # linear-fractional Dinkelbach step scaled to the lattice) can make the
        # auxiliary L1 model numerically fragile.  Returning no auxiliary
        # witness is safe: the caller independently checks the tangent-master
        # lattice point at the same target and never certifies an unchecked
        # candidate.
        return None, nodes
    return counts, nodes


def _fixed_objective_lattice_search(
    *,
    seed_point: np.ndarray,
    integer_costs: np.ndarray,
    target_cost: int,
    lower: np.ndarray,
    upper: np.ndarray,
    row_size: int,
    histories: np.ndarray,
    constants: np.ndarray,
    threshold: float,
    tangent_rows: list[np.ndarray],
    tangent_uppers: list[float],
    seen: set[tuple[int, ...]],
    maximum_candidates: int = 250_000,
    time_limit: float = 15.0,
) -> tuple[str, np.ndarray | None, int, int]:
    """Certify fixed-score feasibility by exhausting a tangent-bounded lattice.

    Nonlinear continuous coordinate optimizations are used only to seed valid
    tangents. Integer MILPs then certify an outer coordinate box. Exhausting
    every lattice point in that box either finds a genuine PPR witness or
    proves that no integer point at ``target_cost`` is feasible.
    """

    started = perf_counter()
    categories = integer_costs.size

    def values(candidate: np.ndarray) -> np.ndarray:
        return _ppr_values_and_jacobian(candidate, histories, constants)[0]

    def jacobian(candidate: np.ndarray) -> np.ndarray:
        return _ppr_values_and_jacobian(candidate, histories, constants)[1]

    nearest, nodes = _nearest_fixed_objective_counts(
        seed_point,
        integer_costs=integer_costs,
        target_cost=target_cost,
        lower=lower,
        upper=upper,
        row_size=row_size,
        tangent_rows=[],
        tangent_uppers=[],
        time_limit=max(0.01, time_limit),
    )
    if nearest is None:
        return "indeterminate", None, nodes, 0
    fixed_constraints = [
        {
            "type": "eq",
            "fun": lambda candidate: float(candidate.sum() - row_size),
            "jac": lambda candidate: np.ones(categories, dtype=float),
        },
        {
            "type": "eq",
            "fun": lambda candidate: float(integer_costs @ candidate - target_cost),
            "jac": lambda candidate: integer_costs.astype(float),
        },
        {
            "type": "ineq",
            "fun": lambda candidate: threshold - values(candidate),
            "jac": lambda candidate: -jacobian(candidate),
        },
    ]
    fixed = minimize(
        lambda candidate: float(np.sum(np.square(candidate - seed_point))),
        nearest.astype(float),
        jac=lambda candidate: 2.0 * (candidate - seed_point),
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=fixed_constraints,
        options={"ftol": 1e-11, "maxiter": 2_000, "disp": False},
    )
    continuous_points = [np.asarray(fixed.x, dtype=float)] if fixed.success else []
    coordinate_points: list[np.ndarray] = []
    starting_point = continuous_points[0] if continuous_points else nearest.astype(float)
    for category in range(categories):
        for sign in (1.0, -1.0):
            objective = np.zeros(categories, dtype=float)
            objective[category] = sign
            optimized = minimize(
                lambda candidate, coefficients=objective: float(coefficients @ candidate),
                starting_point,
                jac=lambda candidate, coefficients=objective: coefficients,
                method="SLSQP",
                bounds=list(zip(lower, upper, strict=True)),
                constraints=fixed_constraints,
                options={"ftol": 1e-10, "maxiter": 2_000, "disp": False},
            )
            if optimized.success and np.all(np.isfinite(optimized.x)):
                coordinate_points.append(np.asarray(optimized.x, dtype=float))
    for point in (*continuous_points, *coordinate_points):
        point_values, point_jacobian = _ppr_values_and_jacobian(
            point, histories, constants
        )
        looks = np.unique(
            np.r_[
                np.argsort(point_values)[-min(20, histories.shape[0]) :],
                np.flatnonzero(point_values > threshold - 1e-5),
            ]
        ).astype(int)
        for look in looks:
            _append_tangent(
                tangent_rows,
                tangent_uppers,
                seen,
                point,
                float(point_values[look]),
                point_jacobian[look],
                threshold,
            )

    constraint_rows = [np.ones(categories), integer_costs.astype(float), *tangent_rows]
    constraint_lows = [float(row_size), float(target_cost), *([-np.inf] * len(tangent_rows))]
    constraint_highs = [float(row_size), float(target_cost), *tangent_uppers]
    constraint = LinearConstraint(
        np.asarray(constraint_rows),
        np.asarray(constraint_lows),
        np.asarray(constraint_highs),
    )
    integer_lower = np.empty(categories, dtype=int)
    integer_upper = np.empty(categories, dtype=int)
    for category in range(categories):
        for side in ("lower", "upper"):
            remaining = time_limit - (perf_counter() - started)
            if remaining <= 0:
                return "indeterminate", None, nodes, 0
            objective = np.zeros(categories)
            objective[category] = 1.0 if side == "lower" else -1.0
            result = _milp_with_verified_infeasibility(
                objective,
                integrality=np.ones(categories, dtype=int),
                bounds=Bounds(lower, upper),
                constraints=constraint,
                options={"mip_rel_gap": 0.0, "time_limit": max(0.01, remaining)},
            )
            nodes += int(result.mip_node_count or 0)
            if result.status == 2:
                # The outer tangent region itself contains no lattice point.
                return "infeasible", None, nodes, 0
            if not result.success:
                return "indeterminate", None, nodes, 0
            value = int(np.rint(result.x[category]))
            if side == "lower":
                integer_lower[category] = value
            else:
                integer_upper[category] = value

    widths = integer_upper - integer_lower + 1
    candidate_bound = int(np.prod(widths.astype(object), dtype=object))
    if candidate_bound > maximum_candidates:
        return "indeterminate", None, nodes, candidate_bound
    checked = 0
    ranges = [
        range(int(integer_lower[index]), int(integer_upper[index]) + 1)
        for index in range(categories)
    ]
    for values_tuple in product(*ranges):
        candidate = np.fromiter(values_tuple, dtype=int, count=categories)
        if candidate.sum() != row_size or int(integer_costs @ candidate) != target_cost:
            continue
        checked += 1
        ppr_values, _ = _ppr_values_and_jacobian(candidate, histories, constants)
        if np.all(ppr_values < threshold - 1e-8):
            return "feasible", candidate, nodes, checked
    return "infeasible", None, nodes, checked


def solve_linear_ppr_row_endpoint(
    row_size: int,
    histories: np.ndarray,
    integer_costs: Sequence[int],
    *,
    alpha_row: float,
    side: EndpointSide,
    beta: Sequence[float] | float = 0.5,
    time_limit: float = 15.0,
    maximum_master_solves: int = 2_500,
    numerical_margin: float = 1e-8,
) -> RowEndpointCertificate:
    """Return a certified exact row endpoint over the running PPR intersection.

    A successful result is exact for the strict categorical-PPR set.  Near-
    boundary witnesses are rejected as numerically indeterminate rather than
    being silently treated as exact.
    """

    started = perf_counter()
    histories = np.asarray(histories, dtype=int)
    costs = np.asarray(integer_costs, dtype=int)
    if histories.ndim != 2 or histories.shape[1] != costs.size:
        raise ValueError("histories and integer_costs have incompatible shapes")
    if row_size <= 0 or not 0.0 < alpha_row < 1.0:
        raise ValueError("invalid row size or alpha_row")
    if side not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    if histories.shape[0] and (
        np.any(histories < 0)
        or np.any(np.diff(histories, axis=0) < 0)
        or np.any(histories.sum(axis=1) != np.arange(1, histories.shape[0] + 1))
    ):
        raise ValueError("histories must contain cumulative counts at every local look")
    if histories.shape[0] > row_size:
        raise ValueError("audit history exceeds row size")

    categories = costs.size
    beta_array = (
        np.full(categories, float(beta))
        if np.isscalar(beta)
        else np.asarray(beta, dtype=float)
    )
    if beta_array.shape != (categories,) or np.any(beta_array <= 0):
        raise ValueError("beta must be positive and match the category dimension")
    current = (
        histories[-1].copy()
        if histories.shape[0]
        else np.zeros(categories, dtype=int)
    )
    upper = row_size - (int(current.sum()) - current)
    # Adding a constant to every cell leaves the objective order unchanged
    # because completed row counts sum to ``row_size``.  Anchor the optimized
    # coefficients at the favorable extreme so the difficult minimum-risk
    # problems use nonpositive coefficients instead of a large positive
    # offset; this materially improves HiGHS' integer bounds without changing
    # the endpoint lattice.
    anchor = int(costs.max()) if side == "lower" else int(costs.min())
    objective = (costs.astype(float) - anchor) * (
        1.0 if side == "lower" else -1.0
    )

    if histories.shape[0] == row_size:
        numerator = int(costs @ current)
        return RowEndpointCertificate(
            side=side,
            status="exact",
            integer_numerator=numerator,
            witness_counts=current,
            master_bound=float(numerator),
            integer_gap=0.0,
            minimum_ppr_slack=float("inf"),
            history_count=row_size,
            continuous_success=True,
            continuous_iterations=0,
            master_solves=0,
            tangent_cuts=0,
            milp_nodes=0,
            candidate_count=1,
            runtime_seconds=perf_counter() - started,
        )

    threshold = float(np.nextafter(np.log(1.0 / alpha_row), -np.inf))
    constants = _history_constants(row_size, histories, beta_array)
    tangent_rows: list[np.ndarray] = []
    tangent_uppers: list[float] = []
    seen: set[tuple[int, ...]] = set()
    continuous_success = True
    continuous_iterations = 0
    seed_point = current.astype(float)
    remaining = row_size - int(current.sum())
    weights = current.astype(float) + beta_array
    seed_point += remaining * weights / weights.sum()
    seed_point[int(np.argmin(costs) if side == "lower" else np.argmax(costs))] += (
        row_size - float(seed_point.sum())
    )
    rounding_seed_points = [seed_point.copy()]

    if histories.shape[0]:
        def values(candidate: np.ndarray) -> np.ndarray:
            return _ppr_values_and_jacobian(candidate, histories, constants)[0]

        def jacobian(candidate: np.ndarray) -> np.ndarray:
            return _ppr_values_and_jacobian(candidate, histories, constants)[1]

        continuous = minimize(
            lambda candidate: float(objective @ candidate),
            seed_point,
            jac=lambda candidate: objective,
            method="SLSQP",
            bounds=list(zip(current.astype(float), upper.astype(float), strict=True)),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda candidate: float(candidate.sum() - row_size),
                    "jac": lambda candidate: np.ones(categories, dtype=float),
                },
                {
                    "type": "ineq",
                    "fun": lambda candidate: threshold - values(candidate),
                    "jac": lambda candidate: -jacobian(candidate),
                },
            ],
            options={"ftol": 1e-11, "maxiter": 2_000, "disp": False},
        )
        continuous_success = bool(continuous.success)
        continuous_iterations = int(continuous.nit)
        if np.all(np.isfinite(continuous.x)):
            proposed_seed = np.asarray(continuous.x, dtype=float)
            proposed_values, _ = _ppr_values_and_jacobian(
                proposed_seed, histories, constants
            )
            if (
                abs(float(proposed_seed.sum()) - row_size) <= 1e-5
                and np.all(proposed_seed >= current - 1e-6)
                and np.all(proposed_seed <= upper + 1e-6)
                and float(proposed_values.max(initial=-np.inf)) <= threshold + 1e-6
            ):
                # A max-iteration SLSQP point is still useful when it is
                # independently feasible. It remains only a cut/witness seed,
                # never the endpoint certificate.
                seed_point = proposed_seed
                rounding_seed_points.insert(0, proposed_seed.copy())
        seed_values, seed_jacobian = _ppr_values_and_jacobian(
            seed_point, histories, constants
        )
        largest = np.argsort(seed_values)[-min(20, histories.shape[0]) :]
        near_active = np.flatnonzero(seed_values > threshold - 1e-5)
        for look in np.unique(np.r_[largest, near_active]).astype(int):
            _append_tangent(
                tangent_rows,
                tangent_uppers,
                seen,
                seed_point,
                float(seed_values[look]),
                seed_jacobian[look],
                threshold,
            )

    master_solves = 0
    nodes = 0
    candidates = 0
    last_bound = float("nan")
    lattice_searched_targets: set[int] = set()
    best_witness: np.ndarray | None = None
    best_slack = float("nan")
    for rounding_point in rounding_seed_points:
        for rounded_seed in _nearest_bounded_fixed_sum_roundings(
            rounding_point, current.astype(float), upper.astype(float), row_size
        ):
            if histories.shape[0]:
                rounded_values, _ = _ppr_values_and_jacobian(
                    rounded_seed, histories, constants
                )
                rounded_slack = threshold - rounded_values
            else:
                rounded_slack = np.asarray([np.inf])
            if float(rounded_slack.min(initial=np.inf)) > numerical_margin:
                if (
                    best_witness is None
                    or float(objective @ rounded_seed)
                    < float(objective @ best_witness)
                ):
                    best_witness = rounded_seed
                    best_slack = float(rounded_slack.min(initial=np.inf))

    for _ in range(maximum_master_solves):
        remaining_time = time_limit - (perf_counter() - started)
        if remaining_time <= 0:
            raise TimeoutError("categorical-PPR row endpoint exceeded time limit")
        master_rows = list(tangent_rows)
        master_uppers = list(tangent_uppers)
        if best_witness is not None:
            best_objective = int(np.rint(objective @ best_witness))
            scale = float(np.max(np.abs(objective), initial=1.0))
            master_rows.append(objective / scale)
            master_uppers.append(float(best_objective - 1) / scale)
        master = _solve_master(
            objective,
            current.astype(float),
            upper.astype(float),
            row_size,
            master_rows,
            master_uppers,
            time_limit=remaining_time,
        )
        master_solves += 1
        nodes += int(master.mip_node_count or 0)
        if master.status == 2 and best_witness is not None:
            best_target = int(costs @ best_witness)
            return RowEndpointCertificate(
                side=side,
                status="exact",
                integer_numerator=best_target,
                witness_counts=best_witness,
                master_bound=float(best_target),
                integer_gap=0.0,
                minimum_ppr_slack=best_slack,
                history_count=histories.shape[0],
                continuous_success=continuous_success,
                continuous_iterations=continuous_iterations,
                master_solves=master_solves,
                tangent_cuts=len(tangent_rows),
                milp_nodes=nodes,
                candidate_count=candidates,
                runtime_seconds=perf_counter() - started,
            )
        master_certified = bool(
            master.success
            and (master.mip_gap is None or master.mip_gap <= 1e-9)
        )
        master_counts = (
            np.rint(master.x).astype(int)
            if master.x is not None and np.all(np.isfinite(master.x))
            else None
        )
        if not master_certified and master_counts is not None:
            master_counts, gap_nodes, master_certified = (
                _certify_small_integer_master_gap(
                    master,
                    objective=objective,
                    lower=current.astype(float),
                    upper=upper.astype(float),
                    row_size=row_size,
                    tangent_rows=master_rows,
                    tangent_uppers=master_uppers,
                    time_limit=max(0.01, time_limit - (perf_counter() - started)),
                )
            )
            nodes += gap_nodes
        if not master_certified or master_counts is None:
            raise RuntimeError(
                "integer master failed to certify optimality: "
                f"success={master.success}, status={master.status}, "
                f"mip_gap={master.mip_gap!r}, fun={master.fun!r}, "
                f"mip_dual_bound={getattr(master, 'mip_dual_bound', None)!r}, "
                f"message={master.message}"
            )
        target = int(costs @ master_counts)
        last_bound = float(target)

        remaining_time = time_limit - (perf_counter() - started)
        witness, witness_nodes = _nearest_fixed_objective_counts(
            seed_point,
            integer_costs=costs,
            target_cost=target,
            lower=current.astype(float),
            upper=upper.astype(float),
            row_size=row_size,
            tangent_rows=[],
            tangent_uppers=[],
            time_limit=remaining_time,
        )
        nodes += witness_nodes
        if witness is None:
            # The master solution itself is always a lattice point at its
            # bound. HiGHS can occasionally reject the auxiliary L1 model at
            # an active numerical equality; fall back to that master witness.
            witness = master_counts.copy()
        proposed = [witness]
        if not np.array_equal(witness, master_counts):
            proposed.append(master_counts)
        violations: list[
            tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for candidate in proposed:
            candidates += 1
            if histories.shape[0]:
                candidate_values, candidate_jacobian = _ppr_values_and_jacobian(
                    candidate, histories, constants
                )
                slack = threshold - candidate_values
                violated = np.flatnonzero(slack <= numerical_margin)
            else:
                candidate_values = np.empty(0)
                candidate_jacobian = np.empty((0, categories))
                slack = np.asarray([np.inf])
                violated = np.empty(0, dtype=int)
            if violated.size == 0:
                return RowEndpointCertificate(
                    side=side,
                    status="exact",
                    integer_numerator=target,
                    witness_counts=candidate,
                    master_bound=last_bound,
                    integer_gap=0.0,
                    minimum_ppr_slack=float(slack.min(initial=np.inf)),
                    history_count=histories.shape[0],
                    continuous_success=continuous_success,
                    continuous_iterations=continuous_iterations,
                    master_solves=master_solves,
                    tangent_cuts=len(tangent_rows),
                    milp_nodes=nodes,
                    candidate_count=candidates,
                    runtime_seconds=perf_counter() - started,
                )
            violations.append(
                (
                    float(slack[violated].min()),
                    candidate,
                    candidate_values,
                    candidate_jacobian,
                    violated,
                )
            )

        # Literal fixed-objective lattice exhaustion is valuable for tiny
        # validation fixtures.  At CIFAR scale its coordinate box contains
        # tens of millions of points and the preliminary coordinate MILPs add
        # hundreds of redundant cuts before returning indeterminate.  The
        # ordinary outer-approximation loop is already globally exact, so skip
        # that optional acceleration when either dimension is large.
        use_lattice_exhaustion = categories <= 4 or row_size <= 100
        if (
            histories.shape[0]
            and use_lattice_exhaustion
            and target not in lattice_searched_targets
        ):
            lattice_searched_targets.add(target)
            remaining_time = time_limit - (perf_counter() - started)
            lattice_status, lattice_witness, lattice_nodes, lattice_checked = (
                _fixed_objective_lattice_search(
                    seed_point=seed_point,
                    integer_costs=costs,
                    target_cost=target,
                    lower=current.astype(float),
                    upper=upper.astype(float),
                    row_size=row_size,
                    histories=histories,
                    constants=constants,
                    threshold=threshold,
                    tangent_rows=tangent_rows,
                    tangent_uppers=tangent_uppers,
                    seen=seen,
                    time_limit=max(0.01, remaining_time),
                )
            )
            nodes += lattice_nodes
            candidates += lattice_checked
            if lattice_status == "feasible" and lattice_witness is not None:
                lattice_values, _ = _ppr_values_and_jacobian(
                    lattice_witness, histories, constants
                )
                lattice_slack = threshold - lattice_values
                return RowEndpointCertificate(
                    side=side,
                    status="exact",
                    integer_numerator=target,
                    witness_counts=lattice_witness,
                    master_bound=last_bound,
                    integer_gap=0.0,
                    minimum_ppr_slack=float(lattice_slack.min(initial=np.inf)),
                    history_count=histories.shape[0],
                    continuous_success=continuous_success,
                    continuous_iterations=continuous_iterations,
                    master_solves=master_solves,
                    tangent_cuts=len(tangent_rows),
                    milp_nodes=nodes,
                    candidate_count=candidates,
                    runtime_seconds=perf_counter() - started,
                )
            if lattice_status == "infeasible":
                # The tangent master already excludes every better integer
                # objective. Exhaustion just proved that its current bound is
                # unattainable in the nonlinear PPR region.
                objective_cut = -costs.astype(float) if side == "lower" else costs.astype(float)
                objective_upper = float(
                    -(target + 1) if side == "lower" else target - 1
                )
                scale = float(np.max(np.abs(objective_cut), initial=1.0))
                objective_cut /= scale
                objective_upper /= scale
                key = tuple(
                    np.rint(
                        np.r_[objective_cut * 1e12, objective_upper * 1e12]
                    ).astype(np.int64)
                )
                if key not in seen:
                    seen.add(key)
                    tangent_rows.append(objective_cut)
                    tangent_uppers.append(objective_upper)
                    continue
        added = False
        added_count = 0
        for _, candidate, candidate_values, candidate_jacobian, violated in sorted(
            violations, key=lambda item: item[0]
        ):
            ordered_looks = violated[
                np.argsort((threshold - candidate_values)[violated])
            ]
            for look in ordered_looks[:10]:
                if _append_boundary_tangent(
                    tangent_rows,
                    tangent_uppers,
                    seen,
                    interior=seed_point,
                    violating=candidate.astype(float),
                    observed=histories[int(look)],
                    constant=float(constants[int(look)]),
                    threshold=threshold,
                ):
                    added = True
                    added_count += 1
            if added_count >= 10:
                break
        if not added:
            # A boundary tangent can duplicate an existing cut when the MILP
            # revisits the same integer point from a slightly different
            # numerical path.  A supporting tangent at the violating point is
            # also globally valid by convexity and strictly separates any
            # violation with material negative slack.
            for _, candidate, candidate_values, candidate_jacobian, violated in sorted(
                violations, key=lambda item: item[0]
            ):
                ordered_looks = violated[
                    np.argsort((threshold - candidate_values)[violated])
                ]
                for look in ordered_looks[:10]:
                    if _append_tangent(
                        tangent_rows,
                        tangent_uppers,
                        seen,
                        candidate.astype(float),
                        float(candidate_values[int(look)]),
                        candidate_jacobian[int(look)],
                        threshold,
                    ):
                        added = True
                        added_count += 1
                if added_count >= 10:
                    break
        if not added:
            minimum_slack = min(item[0] for item in violations)
            boundary_candidate = min(violations, key=lambda item: item[0])[1]
            raise RuntimeError(
                "numerically indeterminate PPR boundary witness: "
                f"minimum_slack={minimum_slack!r}, target={target}, "
                f"candidate={boundary_candidate.tolist()}"
            )
    raise RuntimeError(
        f"integer endpoint did not converge after {maximum_master_solves} master solves; "
        f"last outer bound={last_bound}"
    )


def project_linear_matrix_score(
    row_sizes: Sequence[int],
    histories_by_row: Sequence[np.ndarray],
    score: LinearMatrixScore,
    *,
    alpha: float = 0.05,
    beta: Sequence[float] | float = 0.5,
    time_limit_per_row: float = 15.0,
) -> MatrixProjectionCertificate:
    started = perf_counter()
    sizes = np.asarray(row_sizes, dtype=int)
    if len(histories_by_row) != sizes.size:
        raise ValueError("histories_by_row must contain one history per row")
    if score.integer_costs.shape[0] != sizes.size:
        raise ValueError("score row count does not match row_sizes")
    alpha_row = alpha / sizes.size
    lower_rows = tuple(
        solve_linear_ppr_row_endpoint(
            int(sizes[row]),
            histories_by_row[row],
            score.integer_costs[row],
            alpha_row=alpha_row,
            side="lower",
            beta=beta,
            time_limit=time_limit_per_row,
        )
        for row in range(sizes.size)
    )
    upper_rows = tuple(
        solve_linear_ppr_row_endpoint(
            int(sizes[row]),
            histories_by_row[row],
            score.integer_costs[row],
            alpha_row=alpha_row,
            side="upper",
            beta=beta,
            time_limit=time_limit_per_row,
        )
        for row in range(sizes.size)
    )
    lower_numerator = int(sum(result.integer_numerator for result in lower_rows))
    upper_numerator = int(sum(result.integer_numerator for result in upper_rows))
    lower = lower_numerator / score.denominator
    upper = upper_numerator / score.denominator
    return MatrixProjectionCertificate(
        lower=lower,
        upper=upper,
        width=upper - lower,
        lower_numerator=lower_numerator,
        upper_numerator=upper_numerator,
        lower_rows=lower_rows,
        upper_rows=upper_rows,
        runtime_seconds=perf_counter() - started,
    )


def _weak_compositions(total: int, parts: int):
    if parts == 1:
        yield np.asarray([total], dtype=int)
        return
    for separators in combinations(range(total + parts - 1), parts - 1):
        augmented = (-1, *separators, total + parts - 1)
        yield np.fromiter(
            (augmented[index + 1] - augmented[index] - 1 for index in range(parts)),
            dtype=int,
        )


def brute_force_running_row_endpoint(
    row_size: int,
    histories: np.ndarray,
    integer_costs: Sequence[int],
    *,
    alpha_row: float,
    side: EndpointSide,
    beta: Sequence[float] | float = 0.5,
) -> tuple[int, np.ndarray]:
    """Tiny-fixture oracle for testing the certified CIFAR endpoint solver."""

    histories = np.asarray(histories, dtype=int)
    costs = np.asarray(integer_costs, dtype=int)
    categories = costs.size
    beta_array = (
        np.full(categories, float(beta))
        if np.isscalar(beta)
        else np.asarray(beta, dtype=float)
    )
    current = histories[-1] if histories.shape[0] else np.zeros(categories, dtype=int)
    constants = _history_constants(row_size, histories, beta_array)
    threshold = float(np.nextafter(np.log(1.0 / alpha_row), -np.inf))
    feasible: list[np.ndarray] = []
    for residual in _weak_compositions(row_size - int(current.sum()), categories):
        candidate = current + residual
        values, _ = _ppr_values_and_jacobian(candidate, histories, constants)
        if np.all(values < threshold):
            feasible.append(candidate)
    if not feasible:
        raise RuntimeError("running PPR row region is empty")
    matrix = np.asarray(feasible, dtype=int)
    objectives = matrix @ costs
    index = int(np.argmin(objectives) if side == "lower" else np.argmax(objectives))
    return int(objectives[index]), matrix[index]


def running_ppr_candidate_slack(
    row_size: int,
    histories: np.ndarray,
    candidate_counts: Sequence[int],
    *,
    alpha_row: float,
    beta: Sequence[float] | float = 0.5,
) -> float:
    """Return the smallest strict-threshold slack over all historical looks."""

    histories = np.asarray(histories, dtype=int)
    candidate = np.asarray(candidate_counts, dtype=int)
    if histories.ndim != 2 or candidate.shape != (histories.shape[1],):
        raise ValueError("candidate and histories have incompatible shapes")
    if candidate.sum() != row_size or (
        histories.shape[0] and np.any(candidate < histories[-1])
    ):
        return float("-inf")
    categories = candidate.size
    beta_array = (
        np.full(categories, float(beta))
        if np.isscalar(beta)
        else np.asarray(beta, dtype=float)
    )
    constants = _history_constants(row_size, histories, beta_array)
    values, _ = _ppr_values_and_jacobian(candidate, histories, constants)
    threshold = float(np.nextafter(np.log(1.0 / alpha_row), -np.inf))
    return float((threshold - values).min(initial=np.inf))
