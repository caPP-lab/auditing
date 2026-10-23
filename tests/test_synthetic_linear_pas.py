import numpy as np

from auditcs.synthetic_linear_pas import (
    POLICY_PAS,
    POLICY_SRS,
    SyntheticLinearRiskAudit,
    initial_oracle_variances,
    make_synthetic_risk_scenario,
)


def test_scenario_is_square_multirow_and_has_declared_variance_ratio():
    scenario = make_synthetic_risk_scenario(20, active_rows=4, row_size=40)
    assert scenario.confusion.shape == (20, 20)
    assert len(scenario.influential_indices) == 4
    assert np.isclose(scenario.truth, 0.30)
    proportional, neyman = initial_oracle_variances(scenario)
    assert np.isclose(proportional / neyman, 5.0)


def test_pas_targets_all_and_only_score_variable_rows_at_start():
    scenario = make_synthetic_risk_scenario(20, active_rows=4, row_size=40)
    audit = SyntheticLinearRiskAudit(scenario, seed=7, policy=POLICY_PAS)
    request = audit.prepare()
    probabilities = np.asarray(request["row_probabilities"])
    active = np.asarray(scenario.influential_indices)
    nuisance = np.setdiff1d(np.arange(scenario.rows), active)
    assert np.all(probabilities[active] > 0.0)
    assert np.isclose(probabilities[active].sum(), 1.0)
    assert np.allclose(probabilities[nuisance], 0.0)


def test_srs_is_uniform_over_remaining_records_and_intervals_are_nested():
    scenario = make_synthetic_risk_scenario(8, active_rows=2, row_size=40)
    audit = SyntheticLinearRiskAudit(scenario, seed=11, policy=POLICY_SRS)
    old = (0.0, 1.0)
    for _ in range(30):
        request = audit.prepare()
        remaining = audit.N_r - audit.audits
        assert np.allclose(request["row_probabilities"], remaining / remaining.sum())
        audit.observe(request["record_id"])
        interval = audit.interval()
        assert interval["lower"] >= old[0] - 1e-12
        assert interval["upper"] <= old[1] + 1e-12
        old = (float(interval["lower"]), float(interval["upper"]))


def test_paired_policies_share_within_row_record_prefixes():
    scenario = make_synthetic_risk_scenario(8, active_rows=2, row_size=40)
    streams = {}
    for policy in (POLICY_SRS, POLICY_PAS):
        audit = SyntheticLinearRiskAudit(scenario, seed=19, policy=policy)
        local = np.zeros(scenario.rows, dtype=int)
        stream = {}
        for _ in range(50):
            request = audit.prepare()
            row = int(request["row"])
            stream[(row, int(local[row]))] = int(request["record_id"])
            local[row] += 1
            audit.observe(request["record_id"])
        streams[policy] = stream
    shared = set(streams[POLICY_SRS]).intersection(streams[POLICY_PAS])
    assert shared
    assert all(streams[POLICY_SRS][key] == streams[POLICY_PAS][key] for key in shared)
