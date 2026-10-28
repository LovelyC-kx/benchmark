import unittest
import numpy as np
from calibration_cases import function_lattice
from functional_anova import decompose,error_risk_budget,paired_bootstrap
from robust_review import gated_specialist_predictions
from decoding import fit_probe


class ReviewCalibration(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(402);self.a,self.b=self.rng.normal(size=(2,97,4));self.names=('F','G','U','E')

    def test_finite_additive_interactions_structurally_zero(self):
        h=function_lattice(lambda z:z[:,0]+.3*z[:,1]-.5*z[:,3],self.a,self.b)
        f=decompose(h,self.names,estimator='symmetric_difference')
        self.assertLess(max(abs(float(f['components'][m,0,0])) for m in range(1,16) if m.bit_count()>1),2e-14)

    def test_ignored_factor_exact_zero_even_with_other_interaction(self):
        h=function_lattice(lambda z:z[:,0]*z[:,1],self.a,self.b)
        f=decompose(h,self.names,estimator='symmetric_difference')
        self.assertLess(max(abs(float(f['components'][m,0,0])) for m in range(1,16) if m&4),2e-14)

    def test_shift_invariance(self):
        h=function_lattice(lambda z:z[:,0]*z[:,1],self.a,self.b)
        a=decompose(h,self.names,estimator='symmetric_difference');b=decompose(h+19,self.names,estimator='symmetric_difference')
        np.testing.assert_allclose(a['components'],b['components'],atol=1e-13)

    def test_total_is_paired_difference_covariance(self):
        h=function_lattice(lambda z:z[:,0]*z[:,1],self.a,self.b)
        f=decompose(h,self.names,estimator='symmetric_difference')
        self.assertAlmostEqual(float(f['total_covariance'][0,0]),float(np.var(h[-1]-h[0],ddof=1)/2))

    def test_new_vector_covariance_transports_through_fixed_head(self):
        h=function_lattice(lambda z:np.c_[z[:,0]*z[:,1],z[:,0]+z[:,3]],self.a,self.b)
        w=np.array([[1.,.4],[-.2,.3],[.1,-1.]])
        f=decompose(h,self.names,estimator='symmetric_difference');g=decompose(h@w.T,self.names,estimator='symmetric_difference')
        np.testing.assert_allclose(np.einsum('ai,sij,bj->sab',w,f['components'],w),g['components'],atol=1e-13)

    def test_exact_binary_population_product(self):
        import itertools
        levels=np.array(list(itertools.product((-1.,1.),repeat=4)))
        a=np.repeat(levels,16,axis=0);b=np.tile(levels,(16,1));h=function_lattice(lambda z:z[:,0]*z[:,1],a,b)
        f=decompose(h,self.names,population_enumeration=True,estimator='symmetric_difference')
        self.assertAlmostEqual(float(f['components'][3,0,0]),1.)
        self.assertAlmostEqual(float(f['components'][1,0,0]),0.)

    def test_risk_closes_on_pooled_ab(self):
        h=function_lattice(lambda z:.1+z[:,0]*z[:,1]-z[:,0],self.a,self.b)
        _,r=error_risk_budget(h,self.names,('F',),estimator='symmetric_difference')
        self.assertAlmostEqual(r['direct_mse'],float(np.mean(h[[0,-1]]**2)))
        self.assertLess(abs(r['mse_identity_error']),1e-13)

    def test_bootstrap_records_primary_estimator(self):
        h=function_lattice(lambda z:z[:,0]*z[:,1],self.a,self.b)
        result=paired_bootstrap(h,self.names,('F',),draws=20,estimator='symmetric_difference')
        self.assertEqual(result['estimator'],'symmetric_difference')

    def test_actual_total_label_budget(self):
        f=fit_probe(self.a,self.a[:,:1],self.b[:43],self.b[:43,:1])
        self.assertEqual(f['total_labelled_scenes_for_fit_and_selection'],140)

    def test_gate_can_change_slopes_without_refit(self):
        contexts=[]
        for i in range(6):
            f={'coef':np.array([[i+1.]]),'bias':np.array([.2]),'input_mean':np.array([.3]),'input_scale':2.}
            contexts.append({'evaluation_r':np.array([[1.],[2.]]),'readers':{'F':{'linear':f}}})
        p=gated_specialist_predictions(contexts,'F');w=gated_specialist_predictions(contexts,'F',shift=1)
        self.assertGreater(float(np.max(abs(p-w))),1.)
        self.assertAlmostEqual(p[2,0,0],(1-.3)/2*3+.2)

if __name__=='__main__':unittest.main()
