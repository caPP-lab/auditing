"""Small data, reporting, and numerical replay checks."""
from pathlib import Path
import hashlib
import json
from dataclasses import asdict
import unittest
import numpy as np
import pandas as pd
from atlas_auditing.datasets import load
from atlas_auditing import experiments as ex
from atlas_auditing.precision_attainment import first_attainment, censored_quantiles

ROOT=Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_frozen_settings(self):
        settings=json.loads((ROOT/'settings.json').read_text())
        self.assertEqual(settings['linear']['config'],asdict(ex.common_config()))
        self.assertEqual(settings['nonlinear']['cifar_tail'],asdict(ex.TailBettingConfig()))

    def test_label_populations(self):
        for dataset,size,errors in [('kenya',33403,6002),('cifar10n',50000,8617),('cleanconll',301353,3612)]:
            pop=load(dataset)
            self.assertEqual(len(pop['y']),size)
            self.assertEqual(np.count_nonzero(pop['y']!=pop['v']),errors)
        self.assertEqual(len(load('cifar10n')['row_labels']),40)
        self.assertEqual(len(load('kenya')['stratum_rows']),300)

    def test_kenya_counts(self):
        pop=load('kenya');counts=np.zeros((50,50),int)
        np.add.at(counts,(pop['y'],pop['v']),1)
        matrix=pd.read_csv(ROOT/'data/kenya/confusion_matrix.csv',index_col=0)
        expected=matrix.loc[pop['labels'],pop['labels']].to_numpy()
        np.testing.assert_array_equal(counts,expected)

    def test_attainment_uses_observed_checkpoints(self):
        stops=first_attainment([25,50,75],[[.2,.1,.04],[.2,.09,.08]],[.1,.05])
        np.testing.assert_array_equal(stops,[[50,75],[50,np.inf]])
        np.testing.assert_array_equal(censored_quantiles([[25],[50],[75],[np.inf]])[:,0],[43.75,62.5,np.inf])

    def test_current_result_sizes(self):
        targets=pd.read_csv(ROOT/'results/target_summary.csv')
        self.assertEqual(len(targets),57)
        self.assertTrue(targets.paths.eq(100).all())
        checkpoints=pd.read_csv(ROOT/'results/nonlinear/checkpoints.csv.gz',usecols=['setting','policy','seed','audit'])
        self.assertTrue(checkpoints.groupby(['setting','policy']).seed.nunique().eq(100).all())
        self.assertTrue(checkpoints.audit.mod(25).eq(0).all())
        self.assertEqual(checkpoints.groupby(['setting','policy']).ngroups,25)

    def test_mislabel_prefix(self):
        for policy in ('pas','srs','prop_ms','prop_m_cv','ppr'):
            actual=ex.mislabel('kenya',policy,ex.LINEAR_SEEDS[0],budget=250)
            self.assertEqual(len(np.unique(actual['record_ids'])),250)
            with np.load(ROOT/'results/mislabel/kenya'/f'{policy}.npz') as saved:
                for field in ('lower','upper'):
                    np.testing.assert_allclose(actual[field],saved[field][0,:250],rtol=0,atol=1e-12)

    def test_entropy_prefix(self):
        actual=ex.nonlinear('kenya','entropy','pas',ex.NONLINEAR_SEEDS['kenya'][0],budget=100)
        frame=pd.read_csv(ROOT/'results/nonlinear/checkpoints.csv.gz')
        expected=frame[frame.setting.eq('kenya_entropy')&frame.policy.eq('pas')&frame.seed.eq(ex.NONLINEAR_SEEDS['kenya'][0])&frame.audit.le(100)].sort_values('audit')
        for field in ('lower','upper'):
            np.testing.assert_allclose(actual[field],expected[field],rtol=0,atol=1e-10)

    def test_source_checksums(self):
        manifest=ROOT/'SHA256SUMS'
        if not manifest.exists():
            self.skipTest('Checksum manifest is generated when packaging the release.')
        for line in manifest.read_text().splitlines():
            digest,name=line.split('  ',1)
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)


if __name__=='__main__':
    unittest.main()
