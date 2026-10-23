import numpy as np

from auditcs.cifar_tail_betting_cs import (
    CifarWorstThreeAudit,
    POLICY_PAS,
    POLICY_SRS,
    SCORE_WORST3_PRECISION,
    SCORE_WORST3_TAXONOMIC_RISK,
    TailBettingConfig,
    _fractional_tail_membership,
)


def _tiny_population(seed=1):
    rng = np.random.default_rng(seed)
    submitted = np.repeat(np.arange(10), 8)
    verified = submitted.copy()
    flip = rng.random(submitted.size) < np.repeat(np.linspace(0.05, 0.5, 10), 8)
    verified[flip] = rng.integers(0, 10, size=int(flip.sum()))
    distance = np.abs(np.arange(10)[:, None] - np.arange(10)[None, :])
    return submitted, verified, distance


def test_fractional_membership_has_exact_tail_mass():
    values = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    bottom = _fractional_tail_membership(values, size=2, tail="bottom")
    top = _fractional_tail_membership(values, size=2, tail="top")
    assert np.isclose(bottom.sum(), 2.0)
    assert np.isclose(top.sum(), 2.0)
    assert np.allclose(bottom[:3], 2.0 / 3.0)


def test_both_score_confidence_sequences_are_nested_through_census():
    submitted, verified, distance = _tiny_population()
    for score in (SCORE_WORST3_PRECISION, SCORE_WORST3_TAXONOMIC_RISK):
        audit = CifarWorstThreeAudit(
            submitted,
            verified,
            score_id=score,
            policy=POLICY_SRS,
            seed=7,
            taxonomy_distances=distance,
            taxonomy_d_max=9,
            config=TailBettingConfig(grid_step=0.01),
        )
        previous = (0.0, 1.0)
        for _ in range(len(submitted)):
            request = audit.prepare()
            audit.observe(request["record_id"])
            interval = audit.interval()
            assert interval["lower"] >= previous[0] - 1e-12
            assert interval["upper"] <= previous[1] + 1e-12
            previous = (interval["lower"], interval["upper"])


def test_srs_is_uniform_over_remaining_records_and_pas_has_floor():
    submitted, verified, distance = _tiny_population(4)
    audits = {}
    for policy in (POLICY_SRS, POLICY_PAS):
        audit = CifarWorstThreeAudit(
            submitted,
            verified,
            score_id=SCORE_WORST3_PRECISION,
            policy=policy,
            seed=13,
            taxonomy_distances=distance,
            taxonomy_d_max=9,
            config=TailBettingConfig(grid_step=0.01),
        )
        for _ in range(30):
            request = audit.prepare()
            remaining = audit.N_r - audit.audits_by_row
            srs = remaining / remaining.sum()
            probability = np.asarray(request["row_probabilities"])
            if policy == POLICY_SRS:
                assert np.allclose(probability, srs)
            else:
                assert np.all(probability[remaining > 0] >= 0.3 * srs[remaining > 0] - 1e-12)
            audit.observe(request["record_id"])
        audits[policy] = audit


def test_paired_policies_share_within_row_record_prefixes():
    submitted, verified, distance = _tiny_population(8)
    streams = {}
    for policy in (POLICY_SRS, POLICY_PAS):
        audit = CifarWorstThreeAudit(
            submitted,
            verified,
            score_id=SCORE_WORST3_PRECISION,
            policy=policy,
            seed=21,
            taxonomy_distances=distance,
            taxonomy_d_max=9,
            config=TailBettingConfig(grid_step=0.01),
        )
        local = np.zeros(10, dtype=int)
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


def test_endpoint_balanced_pas_exposes_valid_mixture_components():
    submitted, verified, distance = _tiny_population(11)
    config = TailBettingConfig(
        grid_step=0.01,
        allocation_mode="endpoint_balanced",
        proportional_weight=0.4,
        range_weight=0.3,
        neyman_weight=0.3,
        active_endpoint_weight=0.5,
    )
    audit = CifarWorstThreeAudit(
        submitted,
        verified,
        score_id=SCORE_WORST3_TAXONOMIC_RISK,
        policy=POLICY_PAS,
        seed=31,
        taxonomy_distances=distance,
        taxonomy_d_max=9,
        config=config,
    )
    for _ in range(35):
        request = audit.prepare()
        probabilities = np.asarray(request["row_probabilities"])
        proportional = np.asarray(request["allocation_proportional"])
        range_component = np.asarray(request["allocation_range"])
        neyman = np.asarray(request["allocation_neyman"])
        profile = np.asarray(request["profile_bottleneck_membership"])
        relevance = np.asarray(request["endpoint_relevance"])
        expected = 0.4 * proportional + 0.3 * range_component + 0.3 * neyman
        assert np.allclose(probabilities, expected)
        assert np.isclose(profile.sum(), 3.0)
        assert np.isclose(relevance.sum(), 3.0)
        assert np.all(probabilities >= 0.4 * proportional - 1e-12)
        audit.observe(request["record_id"])


def test_empty_discrete_grid_freezes_last_nonempty_set_conservatively():
    submitted, verified, distance = _tiny_population(15)
    audit = CifarWorstThreeAudit(
        submitted,
        verified,
        score_id=SCORE_WORST3_PRECISION,
        policy=POLICY_SRS,
        seed=44,
        taxonomy_distances=distance,
        taxonomy_d_max=9,
        config=TailBettingConfig(grid_step=0.01),
    )
    before = audit.interval()
    audit._profile_crossed[:] = True
    audit._adaptive_crossed[:] = True
    after = audit.interval()
    assert after["empty_grid_event"]
    assert after["empty_grid_events"] == 1
    assert after["lower"] == before["lower"]
    assert after["upper"] == before["upper"]
