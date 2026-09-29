import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from analyze_a import analyze,write_analysis
from review_a import review_analysis,suffix_reference,mapping_reference
from mapping import fit_g
from io_utils import read,save


def example():
    rng=np.random.default_rng(123)
    n=60
    tasks=np.repeat(['suite/task0','suite/task1','suite/task2','suite/task3'],[12,13,16,19])
    s=rng.random(n)
    errors=np.zeros((n,11),dtype=np.float32)
    target=np.arange(n)%11
    for i,j in enumerate(target):
        errors[i,:j]=.02
    # Actual direct action can pass while suffix direct reuse is invalid.
    errors[2,0]=0
    bins=np.searchsorted(np.quantile(s,[.2,.4,.6,.8]),s,side='right')
    return errors,s,tasks,bins


class AnalysisTests(unittest.TestCase):
    def test_summary_labels_and_direct_reuse_are_distinct(self):
        errors,s,tasks,bins=example()
        result,folds,pairs,oracle,selected=analyze(errors,s,tasks,bins)
        np.testing.assert_array_equal(oracle,np.arange(len(s))%11)
        self.assertGreater(result['direct_action_pass_rate'],result['conservative_direct_reuse_rate'])
        self.assertEqual(len(folds),4)
        self.assertEqual(result['phase_b_nominal_shared_first_nfe'],46*len(pairs))
        self.assertFalse(result['overall_goal_complete'])
        for f in folds:
            ids=np.array(f['test_indices']);pred=np.array(f['sigma_indices_all_states'])
            np.testing.assert_array_equal(selected[ids],pred[ids])

    def test_exact_float32_threshold_is_failure(self):
        errors=np.array([.01,0,0],dtype=np.float32)
        self.assertEqual(suffix_reference(errors),1)

    def test_outer_label_perturbation_does_not_change_own_g(self):
        errors,s,tasks,bins=example()
        _,folds,_,_,selected=analyze(errors,s,tasks,bins)
        changed=errors.copy();changed[tasks==tasks[0],:10]=.03
        _,altered,_,_,other=analyze(changed,s,tasks,bins)
        self.assertEqual(folds[0]['mapping'],altered[0]['mapping'])
        np.testing.assert_array_equal(selected[tasks==tasks[0]],other[tasks==tasks[0]])
        self.assertEqual(folds[0]['sigma_indices_all_states'],altered[0]['sigma_indices_all_states'])

    def test_weighted_quantile_independent_grid_cdf(self):
        errors,s,tasks,_=example()
        labels=np.array([suffix_reference(e) for e in errors])
        actual=fit_g(s,labels,tasks)
        edges,raw,monotone=mapping_reference(s,labels,tasks)
        np.testing.assert_array_equal(actual['boundaries'],edges)
        self.assertEqual(actual['raw_sigma_indices'],raw)
        self.assertEqual(actual['sigma_indices'],monotone)

    def test_synthetic_artifact_roundtrip_and_corruption_detection(self):
        errors,s,tasks,bins=example()
        entries=[dict(id=i,key=['suite',int(tasks[i][-1]),0,i],signal=float(s[i]),signal_bin=int(bins[i]))
                 for i in range(len(s))]
        with tempfile.TemporaryDirectory(prefix='sigma-analysis-test-') as temp:
            root=Path(temp)/'analysis'
            write_analysis(root,errors,entries,dict(synthetic=True,not_scientific_results=True))
            self.assertTrue(review_analysis(root))
            result=read(root/'SUMMARY.json')
            result['heldout_g']['pooled_sigma_mae']+=1
            save(root/'SUMMARY.json',result)
            with self.assertRaises(AssertionError):
                review_analysis(root)


if __name__=='__main__':
    unittest.main()
