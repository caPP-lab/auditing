import numpy as np

from auditcs.mislabel_cs import (
    MislabelAuditConfig,
    MislabelRateAudit,
    POLICY_PAS,
    POLICY_SRS,
    serfling_half_width,
    serfling_target_reviews,
)


def _population():
    rows = np.repeat(np.arange(3), [6, 5, 4])
    submitted = np.arange(3)
    verified = submitted[rows].copy()
    verified[[1, 8, 11, 13]] = [1, 2, 0, 0]
    return rows, verified, submitted


def test_serfling_matches_declared_population_numbers():
    assert np.isclose(100 * serfling_half_width(100, 33_403), 13.75, atol=0.01)
    assert np.isclose(100 * serfling_half_width(1_000, 33_403), 4.29, atol=0.01)
    assert serfling_target_reviews(0.02, 33_403) == 4_154


def test_srs_is_uniform_over_remaining_records_and_cs_is_nested():
    rows, verified, submitted = _population()
    audit = MislabelRateAudit(rows, verified, submitted, seed=5, policy=POLICY_SRS)
    old = (0.0, 1.0)
    for _ in range(10):
        request = audit.prepare()
        remaining = audit.N_r - audit.audits
        expected = remaining / remaining.sum()
        assert np.allclose(request["row_probabilities"], expected)
        audit.observe(request["record_id"])
        interval = audit.interval()
        assert interval["lower"] >= old[0] - 1e-12
        assert interval["upper"] <= old[1] + 1e-12
        old = (interval["lower"], interval["upper"])


def test_pas_probabilities_include_declared_srs_floor():
    rows, verified, submitted = _population()
    config = MislabelAuditConfig(srs_floor=0.3)
    audit = MislabelRateAudit(
        rows,
        verified,
        submitted,
        seed=11,
        policy=POLICY_PAS,
        config=config,
    )
    for _ in range(8):
        request = audit.prepare()
        remaining = audit.N_r - audit.audits
        active = remaining > 0
        proportional = remaining / remaining.sum()
        assert np.all(
            request["row_probabilities"][active]
            >= config.srs_floor * proportional[active] - 1e-12
        )
        audit.observe(request["record_id"])


def test_paired_policies_share_within_row_record_prefixes():
    rows, verified, submitted = _population()
    paths = {}
    for policy in (POLICY_SRS, POLICY_PAS):
        audit = MislabelRateAudit(rows, verified, submitted, seed=19, policy=policy)
        local = np.zeros(3, dtype=int)
        stream = {}
        for _ in range(12):
            request = audit.prepare()
            row = request["row"]
            stream[(row, int(local[row]))] = request["record_id"]
            local[row] += 1
            audit.observe(request["record_id"])
        paths[policy] = stream
    shared = set(paths[POLICY_SRS]).intersection(paths[POLICY_PAS])
    assert shared
    assert all(paths[POLICY_SRS][key] == paths[POLICY_PAS][key] for key in shared)
