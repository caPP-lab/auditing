import numpy as np
from atlas_auditing.mislabel import MislabelAuditConfig, MislabelRateAudit

def common_config():
    return MislabelAuditConfig(submitted_label_probability=0.5)

class LabelBlindAudit(MislabelRateAudit):

    def __init__(self, record_rows, row_to_submitted_label, *, seed, policy, config=None, row_sampler=None):
        super().__init__(record_rows, np.full(len(record_rows), -1, dtype=int), row_to_submitted_label, seed=seed, policy=policy, config=config or common_config(), row_sampler=row_sampler)

    def observe(self, record_id):
        raise RuntimeError('Use observe_error after paying for the selected record')

    def observe_error(self, record_id, error):
        if error not in (0, 1):
            raise ValueError('error must be binary')
        if self._pending is None or int(record_id) != self._pending['record_id']:
            raise ValueError('reveal must match the selected record')
        row = self._pending['row']
        self.verified_labels[record_id] = int(self.row_to_submitted_label[row]) + int(error)
        return super().observe(record_id)

    def interval(self):
        if not np.any(~(self._rejected_plus | self._rejected_minus)):
            return dict(lower=1.0, upper=0.0, half_width=np.nan, empty=True)
        return {**super().interval(), 'empty': False}
