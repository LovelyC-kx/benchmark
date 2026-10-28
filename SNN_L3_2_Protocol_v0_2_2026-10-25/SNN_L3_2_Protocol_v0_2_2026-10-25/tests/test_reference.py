import importlib.util
from pathlib import Path
import unittest
from dataclasses import replace
import numpy as np
from load_data import LoadConfig,generate,model_payload,make_source_pair,source_reference,loss_components_numpy,projection,classification_labels
from linear_reference import gaussian_linear_capacity,linear_age_matrices,fit_linear_encoding,risk_accounting,fit_ridge,predict_ridge,regression_metrics
from experiment_plan import build_plan,RunConfig

class ProtocolReferences(unittest.TestCase):
    def test_demand_does_not_change_observation_or_latents(self):
        baseline=generate(LoadConfig(required_history=1),48)
        for l in (2,4):
            batch=generate(LoadConfig(required_history=l),48)
            for key in ('x','latents','measurement_noise','event_times','y_all'):
                np.testing.assert_array_equal(baseline[key],batch[key])
            self.assertEqual(int(batch['target_mask'].sum()),4*(l+1))

    def test_prefix_and_sample_id_permutation(self):
        c=LoadConfig();a=generate(c,12);b=generate(c,24)
        np.testing.assert_array_equal(a['x'],b['x'][:12])
        ids=np.array([9,1,6,0]);d=generate(c,4,sample_ids=ids)
        np.testing.assert_array_equal(d['x'],b['x'][ids])

    def test_projection_is_shared_across_k(self):
        np.testing.assert_array_equal(projection(LoadConfig(k=1)),projection(LoadConfig(k=4)))
        p=projection(LoadConfig());np.testing.assert_allclose(p.T@p,np.eye(4),atol=1e-14)

    def test_raw_and_fixed_rms_expected_energy(self):
        for k in (1,2,4):
            c=LoadConfig(k=k)
            self.assertAlmostEqual(c.sensory_scale**2*(k+8*c.sigma**2),4+8*c.sigma**2)
        self.assertEqual(LoadConfig(k=4).sensory_scale,1.)

    def test_blank_age_same_prefix_no_new_measurement(self):
        a=generate(LoadConfig(blank_delay=1),24);b=generate(LoadConfig(blank_delay=11),24)
        np.testing.assert_array_equal(a['x'],b['x'])
        np.testing.assert_array_equal(model_payload(a)['x'],model_payload(b)['x'][:,:14])
        self.assertTrue(np.all(model_payload(b)['x'][:,13:]==0))
        self.assertEqual(set(model_payload(b)),{'x'})

    def test_balanced_timing_and_fixed_anchor(self):
        b=generate(LoadConfig(),60)
        self.assertTrue(np.all(b['event_times'][:,0]==2));self.assertTrue(np.all(b['event_times'][:,4]==12))
        for block in range(0,60,6):
            self.assertEqual(len(set(map(tuple,b['event_times'][block:block+6,1:4]))),6)

    def test_write_pressure_preserves_anchor_measurements(self):
        base=generate(LoadConfig(required_history=1),48)
        for w in (0,1):
            b=generate(LoadConfig(required_history=1,extra_writes=w),48)
            np.testing.assert_array_equal(base['x'][:,[2,12]],b['x'][:,[2,12]])
            self.assertTrue(np.all(b['source_present'].sum(1)==w+2))
            self.assertTrue(np.all(b['x'][:,:,13].sum(1)==w+2))
        with self.assertRaises(ValueError):generate(LoadConfig(required_history=4,extra_writes=0),12)

    def test_active_dimensions_and_labels_no_shortcut_metadata(self):
        c=LoadConfig(k=1);b=generate(c,24)
        self.assertTrue(np.all(b['y_all'][:,:,1:]==0))
        y=classification_labels(c,b)
        np.testing.assert_array_equal(y,classification_labels(c,generate(c,24)))
        self.assertEqual(y.shape,(24,5))

    def test_pair_changes_only_selected_content_and_labels(self):
        c=LoadConfig();b=generate(c,24);p=make_source_pair(c,b,'H1')
        np.testing.assert_array_equal(b['latents'][:,1:],p['latents'][:,1:])
        np.testing.assert_array_equal(b['measurement_noise'],p['measurement_noise'])
        same=np.ones(24,dtype=bool);same[2]=False
        np.testing.assert_array_equal(b['x'][:,same],p['x'][:,same])
        self.assertFalse(np.array_equal(b['y_all'][:,0],p['y_all'][:,0]))

    def test_anchor_loss_coefficient_does_not_shrink_with_l(self):
        p=np.zeros((4,5,4));y=np.zeros_like(p);p[:,0]=2;p[:,4]=3
        vals=[]
        for l in (1,2,4):vals.append(loss_components_numpy(p,y,LoadConfig(required_history=l))['anchor_mse'])
        self.assertEqual(vals,[6.5,6.5,6.5])
        p[:,1:4]=100
        a=loss_components_numpy(p,y,LoadConfig(required_history=1),0.)
        b=loss_components_numpy(p,y,LoadConfig(required_history=4),0.)
        self.assertEqual(a['weighted_loss'],b['weighted_loss'])

    def test_full_observation_oracle_noiseless(self):
        c=LoadConfig(sigma=0.);b=generate(c,120)
        for j,name in enumerate(('H1','H2','H3','H4','C')):
            np.testing.assert_allclose(source_reference(c,b,name),b['latents'][:,j],atol=5e-7)

    def test_independent_splits_and_no_input_permissions_leak(self):
        c=LoadConfig();a=generate(c,20,'encoder_train');b=generate(c,20,'audit_test')
        self.assertFalse(np.array_equal(a['x'],b['x']))
        self.assertFalse(set(a['group_ids'])&set(b['group_ids']))
        self.assertEqual(set(model_payload(a)),{'x'})

    def test_nonorthogonal_invertible_is_recoverable(self):
        a=np.array([[1.,.9],[0.,np.sqrt(1-.9**2)]])
        ref=gaussian_linear_capacity(a)
        np.testing.assert_allclose(ref['per_coordinate_r2'],1.,atol=1e-12)
        noisy=gaussian_linear_capacity(a,.2)
        orthogonal=gaussian_linear_capacity(np.eye(2),.2)
        self.assertLess(noisy['sum_r2'],orthogonal['sum_r2'])

    def test_linear_bound_and_null_signal(self):
        rng=np.random.default_rng(81)
        for d,m in [(2,4),(8,20),(20,8)]:
            a=rng.normal(size=(d,m));r=gaussian_linear_capacity(a,.3)
            self.assertLessEqual(r['sum_r2'],min(d,m)+1e-10)
            self.assertAlmostEqual(r['sum_r2'],r['spectral_sum'],places=10)
        self.assertEqual(gaussian_linear_capacity(np.zeros((2,4)))['sum_r2'],0.)

    def test_fixed_dimension_does_not_imply_constant_total(self):
        weak=gaussian_linear_capacity(.1*np.eye(2),.1)
        strong=gaussian_linear_capacity(np.eye(2),.1)
        self.assertTrue(np.all(strong['per_coordinate_r2']>weak['per_coordinate_r2']))

    def test_age_expansion_matches_actual_linear_recursion(self):
        rng=np.random.default_rng(63);a=rng.normal(size=(3,3))*.1;b=rng.normal(size=(3,2));x=rng.normal(size=(9,2))
        h=np.zeros(3)
        for xt in x:h=a@h+b@xt
        g=linear_age_matrices(a,b,len(x))
        np.testing.assert_allclose(h,np.einsum('tdk,tk->d',g,x),atol=1e-12)

    def test_risk_identity_keeps_cross_terms_on_heldout_data(self):
        rng=np.random.default_rng(221);z=rng.normal(size=(600,4));z[:,1]+=.8*z[:,0]
        r=z@rng.normal(size=(4,3))+.3*rng.normal(size=(600,3))+.4
        enc=fit_linear_encoding(r[:300],z[:300]);w=rng.normal(size=(1,3));bias=np.array([.7])
        risk=risk_accounting(r[300:],z[300:],w,bias,enc,[0])
        self.assertAlmostEqual(risk['direct_mse'],risk['accounted_mse'],places=11)
        self.assertLess(risk['max_pointwise_identity_error'],1e-12)
        self.assertGreater(abs(risk['terms']['cross_target_nontarget']),1e-3)

    def test_ridge_and_negative_heldout_r2(self):
        rng=np.random.default_rng(192);x=rng.normal(size=(100,3));y=x@rng.normal(size=(3,2))+2
        fit=fit_ridge(x[:60],y[:60],0.)
        np.testing.assert_allclose(predict_ridge(fit,x[60:]),y[60:],atol=1e-12)
        bad=regression_metrics(np.full((20,1),100.),rng.normal(size=(20,1)))
        self.assertLess(bad['sum_r2'],0)

    def test_plan_counts_unique_ids_and_resource_formula(self):
        p=build_plan();self.assertEqual(p['screen_count'],15);self.assertEqual(p['primary_count'],135)
        self.assertEqual(p['run_count'],435)
        self.assertEqual(len(set(r['run_id'] for r in p['runs'])),435)
        self.assertEqual(RunConfig(width=32).parameters,8916)
        self.assertEqual(RunConfig(width=8).parameters,5892)

    def test_equal_total_demand_controls_have_identical_inputs(self):
        for ka,la,kb,lb in [(2,1,1,3),(4,1,2,3)]:
            a=generate(LoadConfig(k=ka,required_history=la,content_mode='demand_only'),24)
            b=generate(LoadConfig(k=kb,required_history=lb,content_mode='demand_only'),24)
            np.testing.assert_array_equal(a['x'],b['x'])
            self.assertEqual(int(a['target_mask'].sum()),int(b['target_mask'].sum()))

    def test_invalid_observation_contracts_fail(self):
        for c in (LoadConfig(k=0),LoadConfig(sigma=float('nan')),LoadConfig(blank_delay=2)):
            with self.assertRaises(ValueError):generate(c,2)
        with self.assertRaises(ValueError):generate(LoadConfig(),2,sample_ids=[1,1])

if __name__=='__main__':unittest.main()
