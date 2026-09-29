import unittest
import numpy as np
from definitions import initialize,rmse,sigma_star,n_star,integer_allocation
from mapping import fit_g,predict_g,outer_mappings,weighted_quantile


class DefinitionTests(unittest.TestCase):
    def test_prior_positions_endpoints_and_sigma_independent_of_n(self):
        p=np.arange(224,dtype=np.float32).reshape(32,7)
        e=np.ones((32,7),dtype=np.float32)
        for sigma in np.arange(11)/10:
            x=initialize(p,e,sigma)
            np.testing.assert_array_equal(x[16:],e[16:])
            # Match production float32 arithmetic; numpy.float64 scalar promotion
            # would instead test a different, double-precision reference.
            np.testing.assert_array_equal(x[:16],(1-float(sigma))*p[16:]+float(sigma)*e[:16])
        np.testing.assert_array_equal(initialize(p,e,1),e)
        np.testing.assert_array_equal(initialize(p,e,0)[:16],p[16:])

    def test_suffix_strict_boundary_and_infeasible(self):
        self.assertEqual(sigma_star([0,.02]+[0]*9),2)
        self.assertEqual(sigma_star([.01]+[0]*10),1)
        self.assertEqual(sigma_star([0]*11),0)
        self.assertIsNone(n_star([0]*9+[.02]))
        self.assertEqual(n_star([.02]*9+[0]),10)
        self.assertEqual(n_star([0]*10),1)

    def test_rmse_all_executed_seven_dimensions(self):
        reference=np.zeros((16,7),dtype=np.float32)
        actions=np.ones((11,16,7),dtype=np.float32)
        np.testing.assert_array_equal(rmse(actions,reference),np.ones(11,dtype=np.float32))
        np.testing.assert_array_equal(integer_allocation([0,1,1.1,9.1,99]),[1,1,2,10,10])

    def test_mapping_monotone_grid_and_constant_signal(self):
        s=np.arange(30)
        tasks=np.repeat(['a','b','c'],10)
        y=np.tile([9,1,0,4,5,1,2,3,2,1],3)
        mapping=fit_g(s,y,tasks)
        predicted=predict_g(mapping,np.arange(-5,40))
        self.assertTrue(np.all(np.diff(predicted)>=0))
        self.assertTrue(((predicted>=0)&(predicted<=10)).all())
        constant=fit_g(np.ones(30),y,tasks)
        self.assertTrue((predict_g(constant,np.ones(30))==weighted_quantile(y,np.ones(30))).all())

    def test_outer_label_perturbation_cannot_change_own_mapping_or_requirements(self):
        s=np.arange(40)/40
        tasks=np.repeat(['a','b','c','d'],10)
        y=np.arange(40)%11
        rows,pairs=outer_mappings(s,y,tasks)
        changed=y.copy();changed[tasks=='a']=10-y[tasks=='a']
        altered,_=outer_mappings(s,changed,tasks)
        self.assertEqual(rows[0],altered[0])
        for row in rows:
            self.assertNotIn(row['test_task'],row['mapping']['training_tasks'])
            np.testing.assert_array_equal(row['sigma_indices_all_states'],predict_g(row['mapping'],s))
        self.assertEqual(len(pairs),len(set(pairs)))


if __name__=='__main__':
    unittest.main()
