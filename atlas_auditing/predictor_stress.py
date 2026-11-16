from dataclasses import dataclass
import numpy as np
from atlas_auditing.audit import LabelBlindAudit

@dataclass(frozen=True)
class PredictorStress:
    name: str
    label: str
    kind: str = 'identity'
    logit_multiplier: float = 1.0
    reverse_until: int | None = None
SCENARIOS = (PredictorStress('default', 'Default'), PredictorStress('underconfident_05', 'Underconfident (logit x 0.5)', 'temperature', 0.5), PredictorStress('overconfident_2', 'Overconfident (logit x 2)', 'temperature', 2.0), PredictorStress('overconfident_4', 'Overconfident (logit x 4)', 'temperature', 4.0), PredictorStress('constant_05', 'Constant across strata (p = 0.5)', 'constant'), PredictorStress('reverse_25', 'Ranking reversed: first 25 audits', 'reverse', reverse_until=25), PredictorStress('reverse_50', 'Ranking reversed: first 50 audits', 'reverse', reverse_until=50), PredictorStress('reverse_100', 'Ranking reversed: first 100 audits', 'reverse', reverse_until=100), PredictorStress('reverse_all', 'Ranking reversed throughout', 'reverse'))
SCENARIO_BY_NAME = {s.name: s for s in SCENARIOS}

def distort_prediction(base, stress, completed_audits):
    p = np.asarray(base, dtype=float)
    if stress.kind == 'identity':
        return p.copy()
    if stress.kind == 'constant':
        return np.full_like(p, 0.5)
    if stress.kind == 'temperature':
        a, b = (p ** stress.logit_multiplier, (1 - p) ** stress.logit_multiplier)
        return a / (a + b)
    if stress.kind == 'reverse':
        if stress.reverse_until is not None and completed_audits >= stress.reverse_until:
            return p.copy()
        sd = np.sqrt(np.maximum(p * (1 - p), 0.0))
        return np.where(p <= 0.5, 0.5 - sd, 0.5 + sd)
    raise ValueError(stress.kind)

class MisspecifiedMislabelAudit(LabelBlindAudit):

    def __init__(self, *args, stress, **kwargs):
        self.stress = stress
        super().__init__(*args, **kwargs)

    def _predict_error(self):
        base = super()._predict_error()
        return distort_prediction(base, self.stress, self.reviews)
