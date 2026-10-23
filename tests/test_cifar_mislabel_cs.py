from __future__ import annotations

import numpy as np

from auditcs.cifar_mislabel_cs import (
    CifarMislabelConfig,
    CifarMislabelRateAudit,
    GROUP_ALL_DIFFERENT,
    GROUP_BOTH_OPPOSE,
    GROUP_BOTH_SUPPORT,
    GROUP_ONE_SUPPORT,
    agreement_groups,
    calibrate_stratum_priors,
)
from auditcs.mislabel_cs import POLICY_PAS, POLICY_SRS


def _fixture(repeats: int = 20):
    submitted_block = np.repeat(np.arange(10), 4)
    one_block = np.column_stack(
        (
            np.arange(10),
            np.arange(10),
            (np.arange(10) + 1) % 10,
            (np.arange(10) + 1) % 10,
        )
    ).reshape(-1)
    two_block = np.column_stack(
        (
            np.arange(10),
            (np.arange(10) + 1) % 10,
            (np.arange(10) + 1) % 10,
            (np.arange(10) + 2) % 10,
        )
    ).reshape(-1)
    submitted = np.tile(submitted_block, repeats)
    one = np.tile(one_block, repeats)
    two = np.tile(two_block, repeats)
    verified = submitted.copy()
    verified[(one == two) & (one != submitted)] = one[(one == two) & (one != submitted)]
    groups = agreement_groups(submitted, one, two)
    one_support = np.flatnonzero(groups == GROUP_ONE_SUPPORT)
    all_different = np.flatnonzero(groups == GROUP_ALL_DIFFERENT)
    verified[one_support[::2]] = (submitted[one_support[::2]] + 3) % 10
    verified[all_different[::3]] = (submitted[all_different[::3]] + 4) % 10
    return submitted, verified, one, two


def test_agreement_group_encoding() -> None:
    submitted = np.array([0, 0, 0, 0])
    one = np.array([0, 0, 1, 1])
    two = np.array([0, 1, 1, 2])
    assert agreement_groups(submitted, one, two).tolist() == [
        GROUP_BOTH_SUPPORT,
        GROUP_ONE_SUPPORT,
        GROUP_BOTH_OPPOSE,
        GROUP_ALL_DIFFERENT,
    ]


def test_calibration_produces_finite_40_stratum_prior() -> None:
    submitted, verified, one, two = _fixture()
    priors = calibrate_stratum_priors(submitted, verified, one, two)
    assert priors.shape == (40,)
    assert np.all((priors > 0.0) & (priors < 1.0))


def test_srs_is_uniform_over_remaining_records() -> None:
    submitted, verified, one, two = _fixture()
    priors = calibrate_stratum_priors(submitted, verified, one, two)
    audit = CifarMislabelRateAudit(
        submitted,
        verified,
        one,
        two,
        prior_error=priors,
        seed=7,
        policy=POLICY_SRS,
    )
    request = audit.prepare()
    expected = audit.N_r / audit.N_r.sum()
    assert np.allclose(request["row_probabilities"], expected)


def test_pas_reweights_rows_and_interval_is_nested() -> None:
    submitted, verified, one, two = _fixture(repeats=80)
    priors = calibrate_stratum_priors(submitted, verified, one, two)
    audit = CifarMislabelRateAudit(
        submitted,
        verified,
        one,
        two,
        prior_error=priors,
        seed=9,
        policy=POLICY_PAS,
        config=CifarMislabelConfig(srs_floor=0.1),
    )
    first = audit.prepare()
    proportional = audit.N_r / audit.N_r.sum()
    assert not np.allclose(first["row_probabilities"], proportional)
    previous = (0.0, 1.0)
    for _ in range(75):
        request = audit.prepare()
        audit.observe(int(request["record_id"]))
        interval = audit.interval()
        assert interval["lower"] >= previous[0] - 1e-12
        assert interval["upper"] <= previous[1] + 1e-12
        previous = (interval["lower"], interval["upper"])
