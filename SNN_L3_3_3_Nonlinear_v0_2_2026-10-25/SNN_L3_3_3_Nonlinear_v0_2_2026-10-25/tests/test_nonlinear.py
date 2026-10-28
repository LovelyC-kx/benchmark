import unittest
import importlib.util
from dataclasses import replace
from itertools import product
import numpy as np
from functional_anova import (decompose,power_summary,error_risk_budget,paired_bootstrap,
    conditional_risk,conditional_mean_lattice,mask_for)
from calibration_cases import (product_pairs,function_lattice,additive,ambiguous_product,referenced_product,
    harmful_prediction,focal_target,binding_target,cubic,known_conditional_panels,static_threshold,independent_write_reference)
from decoding import fit_probe,predict,mse,feature_map,fit_scale,ridge
from factor_sampling import factor_schema,full_lattice,conditional_panels
from parent_sampling import fixed_batch
from bridge import coordinate_roles
from load_data import LoadConfig
from context_audit import audit_contexts
from task_objective import requested_objective,whole_task_report

NAMES=('F','G','U','E')


class ProductDecomposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.population,cls.a,cls.b=product_pairs()

    def fit(self,fn):return decompose(function_lattice(fn,self.a,self.b),NAMES,population_enumeration=True)

    def test_additive_only_singletons(self):
        f=self.fit(additive)
        for mask in range(1,16):
            if mask.bit_count()>1:np.testing.assert_allclose(f['components'][mask],0,atol=1e-13)

    def test_additive_coefficients(self):
        f=self.fit(additive)
        for mask,power in [(1,1),(2,4),(4,1),(8,.09)]:self.assertAlmostEqual(np.trace(f['components'][mask]),power)

    def test_pure_product_zero_main(self):
        f=self.fit(ambiguous_product)
        for mask in (1,2,4,8):self.assertAlmostEqual(np.trace(f['components'][mask]),0)
        self.assertAlmostEqual(np.trace(f['components'][3]),1)

    def test_totals_do_not_add(self):
        s=power_summary(self.fit(ambiguous_product))
        self.assertAlmostEqual(s['factor_main_and_total']['F']['total_including_interactions'],1)
        self.assertAlmostEqual(s['factor_main_and_total']['G']['total_including_interactions'],1)
        self.assertAlmostEqual(s['total_variance_per_output'],1)

    def test_third_order_invisible_to_pair_terms(self):
        f=self.fit(cubic)
        for mask in range(1,16):self.assertAlmostEqual(np.trace(f['components'][mask]),1 if mask==7 else 0)

    def test_noise_factor_explicit(self):
        f=self.fit(lambda x:x[:,0]*x[:,3])
        self.assertAlmostEqual(np.trace(f['components'][9]),1)
        self.assertAlmostEqual(np.trace(f['components'][8]),0)

    def test_static_threshold_interaction_without_memory(self):
        f=self.fit(static_threshold)
        for mask in (1,2,3):self.assertAlmostEqual(np.trace(f['components'][mask]),1/16)

    def test_nonlinear_per_write_encoder_no_cross_write_FG(self):
        f=self.fit(independent_write_reference)
        self.assertGreater(np.trace(f['components'][5]),0)
        for mask in range(1,16):
            if mask & 3 == 3:self.assertAlmostEqual(np.trace(f['components'][mask]),0)

    def test_population_covariance_closure(self):
        f=self.fit(additive);self.assertLess(f['covariance_closure_max_error'],1e-13)

    def test_iid_covariance_closure(self):
        rng=np.random.default_rng(2);a=rng.normal(size=(81,4));b=rng.normal(size=(81,4))
        f=decompose(function_lattice(harmful_prediction,a,b),NAMES)
        self.assertLess(f['covariance_closure_max_error'],1e-13)

    def test_fixed_linear_covariance_transport(self):
        rng=np.random.default_rng(4);w=rng.normal(size=(3,2));h=function_lattice(referenced_product,self.a,self.b)
        f=decompose(h,NAMES,population_enumeration=True)
        g=decompose(h @ w.T+2,NAMES,population_enumeration=True)
        np.testing.assert_allclose(g['components'],np.einsum('ai,sij,bj->sab',w,f['components'],w),atol=1e-12)

    def test_nonlinear_readout_changes_allocation(self):
        h=function_lattice(referenced_product,self.a,self.b)
        output=h[:,:,0:1]*h[:,:,1:2]
        f=decompose(output,NAMES,population_enumeration=True)
        self.assertAlmostEqual(np.trace(f['components'][1]),1)
        self.assertAlmostEqual(np.trace(f['components'][3]),0)

    def test_complete_factor_input_required(self):
        with self.assertRaises(ValueError):decompose(np.zeros((8,10,1)),NAMES)

    def test_dependent_factors_rejected(self):
        with self.assertRaises(ValueError):decompose(np.zeros((16,10,1)),NAMES,independent_factors=False)

    def test_nonfinite_rejected(self):
        h=np.zeros((16,10,1));h[1,0,0]=np.nan
        with self.assertRaises(ValueError):decompose(h,NAMES)

    def test_constant_variance_fraction_undefined(self):
        s=power_summary(decompose(np.ones((16,12,1)),NAMES))
        self.assertTrue(all(row['fraction_of_total_variance'] is None for row in s['component_rows']))

    def test_negative_finite_effects_retained(self):
        rng=np.random.default_rng(22);h=rng.normal(size=(16,7,1));f=decompose(h,NAMES)
        self.assertTrue(np.any(f['components'][1:]<0))

    def test_group_order_is_not_polynomial_degree(self):
        # An arbitrary nonlinear function of F alone is still a group main effect.
        f=self.fit(lambda x:np.exp(x[:,0]))
        self.assertGreater(np.trace(f['components'][1]),1)
        for mask in range(2,16):self.assertAlmostEqual(np.trace(f['components'][mask]),0)


class ErrorAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.population,cls.a,cls.b=product_pairs()

    def test_known_harmful_head_mse(self):
        e=function_lattice(harmful_prediction,self.a,self.b)-function_lattice(focal_target,self.a,self.b)
        _,s=error_risk_budget(e,NAMES,('F',),population_enumeration=True)
        self.assertAlmostEqual(s['direct_mse'],.78)
        powers={row['mask']:row['power_per_output'] for row in s['component_rows']}
        for mask,value in [(2,.25),(3,.49),(8,.04)]:self.assertAlmostEqual(powers[mask],value)

    def test_binding_target_is_not_nuisance(self):
        p=function_lattice(binding_target,self.a,self.b)
        s=power_summary(decompose(p,NAMES,population_enumeration=True),('F','G'))
        self.assertAlmostEqual(s['target_contract']['non_target_conditional_variability_power'],0)
        self.assertAlmostEqual(s['target_contract']['target_only_power'],1)

    def test_same_fixed_prediction_wrong_target_mse_two(self):
        e=function_lattice(binding_target,self.a,self.b)-function_lattice(focal_target,self.a,self.b)
        _,s=error_risk_budget(e,NAMES,('F',),population_enumeration=True)
        self.assertAlmostEqual(s['direct_mse'],2)

    def test_brier_can_change_at_perfect_accuracy(self):
        p=function_lattice(lambda x:(1+.6*x[:,0]+.2*x[:,1]+.1*x[:,0]*x[:,1])/2,self.a,self.b)
        y=function_lattice(lambda x:(1+x[:,0])/2,self.a,self.b)
        _,s=error_risk_budget(p-y,NAMES,('F',),population_enumeration=True)
        self.assertAlmostEqual(s['direct_mse'],.0525)
        self.assertEqual(np.mean((p[-1]>=.5)==y[-1]),1)

    def test_vector_reconstruction_error_uses_per_output_mean(self):
        e=function_lattice(lambda x:np.column_stack((.2*x[:,2],.3*x[:,0]*x[:,3])),self.a,self.b)
        _,s=error_risk_budget(e,NAMES,('F','G'),population_enumeration=True)
        self.assertAlmostEqual(s['direct_mse'],.065)

    def test_constant_output_fails_without_nuisance(self):
        _,s=error_risk_budget(-function_lattice(focal_target,self.a,self.b),NAMES,('F',),population_enumeration=True)
        self.assertAlmostEqual(s['direct_mse'],1)
        self.assertAlmostEqual(s['target_contract']['non_target_conditional_variability_power'],0)

    def test_bias_and_variance_iid_risk_closure(self):
        rng=np.random.default_rng(6);a=rng.normal(size=(93,4));b=rng.normal(size=(93,4))
        e=function_lattice(lambda x:2+harmful_prediction(x)-focal_target(x),a,b)
        _,s=error_risk_budget(e,NAMES,('F',))
        self.assertLess(abs(s['mse_identity_error']),1e-12)

    def test_hybrid_targets_are_recomputed(self):
        targets=function_lattice(focal_target,self.a,self.b)
        self.assertFalse(np.array_equal(targets[0],targets[-1]))
        for mask in (1,3,5,7,9,11,13,15):np.testing.assert_array_equal(targets[mask],targets[-1])

    def test_bootstrap_preserves_whole_pairs(self):
        rng=np.random.default_rng(8);a=rng.normal(size=(64,4));b=rng.normal(size=(64,4))
        e=function_lattice(lambda x:.5*x[:,1],a,b)
        s=paired_bootstrap(e,NAMES,('F',),draws=24)
        self.assertEqual(s['independent_pair_rows'],64)
        self.assertEqual(len(s['rows']),18)
        self.assertGreaterEqual(s['max_deviation_band_radius'],0)

    def test_unbiased_repeat_risk_identity(self):
        rng=np.random.default_rng(9);p=rng.normal(size=(21,5,3));y=rng.normal(size=(21,3))
        s=conditional_risk(p,y);self.assertLess(abs(s['identity_error']),1e-13)

    def test_negative_corrected_mean_error_allowed(self):
        p=np.tile(np.array([-1.,1.])[None,:,None],(5,1,1));y=np.zeros((5,1))
        s=conditional_risk(p,y);self.assertLess(s['conditional_mean_error'],0)

    def test_wrong_repeat_target_shape_rejected(self):
        with self.assertRaises(ValueError):conditional_risk(np.ones((5,4,2)),np.ones((5,4,2)))

    def test_conditional_function_truth(self):
        f=conditional_mean_lattice(known_conditional_panels(harmful_prediction),NAMES)
        self.assertLess(f['reconstruction_max_error'],1e-12)
        np.testing.assert_allclose(f['component_functions_estimated'][3,:,0],.7*self.population[:,0]*self.population[:,1],atol=1e-13)

    def test_nested_full_factor_replay_required(self):
        panels=known_conditional_panels(harmful_prediction);panels[15]=panels[15].copy();panels[15][0,0,0]+=1
        with self.assertRaises(ValueError):conditional_mean_lattice(panels,NAMES)


class SamplingAndProbes(unittest.TestCase):
    def test_factors_partition_visible_coordinates(self):
        cfg=LoadConfig(k=2,required_history=1,content_mode='demand_only')
        s=factor_schema(cfg,'C');all_coords=sum(s['coordinates'].values(),[])
        self.assertEqual(len(all_coords),20);self.assertEqual(len(set(all_coords)),20)
        self.assertEqual(len(s['coordinates']['F']),2);self.assertEqual(len(s['coordinates']['G']),2)

    def test_inactive_factors_removed(self):
        s=factor_schema(LoadConfig(required_history=4,sigma=0),'C')
        self.assertEqual(s['names'],('F','G'))

    def test_full_empty_hybrids_and_fixed_schedule(self):
        cfg=LoadConfig(required_history=1)
        h,s,_=full_lattice(cfg,12,'geometry_fit',10000,20000,3,'C')
        a=fixed_batch(cfg,12,'geometry_fit',10000,3);b=fixed_batch(cfg,12,'geometry_fit',20000,3)
        np.testing.assert_array_equal(h[15]['x'],a['x']);np.testing.assert_array_equal(h[0]['x'],b['x'])
        for v in h.values():np.testing.assert_array_equal(v['event_times'],a['event_times'])

    def test_shared_factor_frozen_at_coordinate_level(self):
        cfg=LoadConfig(k=2,required_history=1,content_mode='demand_only')
        h,s,_=full_lattice(cfg,8,'geometry_fit',10000,20000,0,'C')
        for j,k in s['coordinates']['F']:np.testing.assert_array_equal(h[1]['latents'][:,j,k],h[15]['latents'][:,j,k])
        for j,k in s['coordinates']['U']:np.testing.assert_array_equal(h[1]['latents'][:,j,k],h[0]['latents'][:,j,k])

    def test_overlapping_pair_ids_rejected(self):
        with self.assertRaises(ValueError):full_lattice(LoadConfig(),12,'geometry_fit',100,105)

    def test_audit_access_rejected(self):
        with self.assertRaises(ValueError):full_lattice(LoadConfig(),12,'audit_test',100,200)

    def test_repeat_full_mask_exact_input(self):
        cfg=LoadConfig();p,_,s=conditional_panels(cfg,4,3,'geometry_fit',30000,0,'C',[15])
        for b in range(1,3):np.testing.assert_array_equal(p[15][:,0],p[15][:,b])

    def test_repeat_rng_includes_split(self):
        cfg=LoadConfig();a,_,_=conditional_panels(cfg,4,3,'geometry_fit',30000,0,'C',[0])
        b,_,_=conditional_panels(cfg,4,3,'development_test',30000,0,'C',[0])
        self.assertFalse(np.array_equal(a[0],b[0]))

    def test_quadratic_legal_reference_restores_focal(self):
        population,_,_=product_pairs();r=referenced_product(population);y=population[:,:1]
        f=fit_probe(r,y,r,y,'quadratic');self.assertLess(mse(predict(f,r),y),1e-20)
        f=fit_probe(r,y,r,y,'linear');self.assertAlmostEqual(mse(predict(f,r),y),1)

    def test_quadratic_feature_count(self):
        x=np.ones((4,5));self.assertEqual(feature_map(x,np.zeros(5),1,'quadratic').shape[1],20)

    def test_dual_ridge_matches_primal(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(9,20));y=rng.normal(size=(9,3));f=ridge(x,y,.1)
        xc=x-x.mean(0);yc=y-y.mean(0)
        direct=np.linalg.solve(xc.T@xc+.9*np.eye(20),xc.T@yc)
        np.testing.assert_allclose(f['coef'],direct,atol=1e-12)

    def test_probe_uses_train_scale_and_mean_only(self):
        x=np.arange(30.).reshape(10,3);y=x[:,:1]
        f=fit_probe(x,y,x+100,y,'linear')
        np.testing.assert_array_equal(f['input_mean'],x.mean(0))
        self.assertAlmostEqual(f['input_scale'],np.sqrt(np.mean((x-x.mean(0))**2)))

    def test_no_reference_sign_ambiguity(self):
        population,_,_=product_pairs();r=ambiguous_product(population)
        for observed in (-1,1):self.assertAlmostEqual(population[r==observed,0].mean(),0)

    def test_context_pooling_cancels_linear_code(self):
        x=np.array(list(product((-1.,1.),repeat=2)));r=x[:,0]*x[:,1]
        self.assertEqual(np.mean(r*x[:,0]),0)
        self.assertEqual(np.mean((r*x[:,1]-x[:,0])**2),0)

    def test_nonlinear_focal_power_need_not_preserve_sign(self):
        f=np.array([-np.sqrt(1.5),0,np.sqrt(1.5)])
        r=f**2-1
        self.assertAlmostEqual(np.mean(f*r),0)
        self.assertGreater(np.var(r),0)
        self.assertEqual(r[0],r[2])


class ContextAudit(unittest.TestCase):
    def contexts(self):
        y=np.tile(np.array([-1.,1.]),8)[:,None]
        contexts=[]
        for i in range(6):
            r=y*(1 if i%2==0 else -1)
            f=fit_probe(r,y,r,y,'linear')
            contexts.append({'schedule_index':i,'train_r':r,'select_r':r,'evaluation_r':r,
                'train_y':y,'select_y':y,'evaluation_y':y,'target_slices':{'F':np.array([0])},
                'readers':{'F':{'linear':f}}})
        return contexts

    def test_context_reference_extra_permission(self):
        s=audit_contexts(self.contexts())['F']['shared_readers']
        self.assertAlmostEqual(s['legal_observation_only']['quadratic']['evaluation_mse'],1)
        self.assertLess(s['with_extra_schedule']['quadratic']['evaluation_mse'],1e-20)

    def test_shared_and_specialist_presentations_registered(self):
        s=audit_contexts(self.contexts())['F']
        self.assertEqual(s['total_training_presentations'],96)
        self.assertEqual(s['unique_latent_training_scenes'],16)
        self.assertEqual(s['specialist_system_head_count'],6)

    def test_unpaired_context_targets_rejected(self):
        c=self.contexts();c[1]['evaluation_y']=c[1]['evaluation_y']+1
        with self.assertRaises(ValueError):audit_contexts(c)


class TaskObjective(unittest.TestCase):
    def test_anchored_weights_preserve_parent_task_loss(self):
        from load_data import loss_components_numpy
        cfg=LoadConfig(k=2,required_history=3,content_mode='demand_only')
        roles=coordinate_roles(cfg,'C');obj=requested_objective(cfg,roles,'anchored',.5)
        rng=np.random.default_rng(93);p=rng.normal(size=(16,20));y=rng.normal(size=(16,20))
        expected=loss_components_numpy(p,y,cfg,.5)['weighted_loss']
        actual=np.mean(np.sum((p[:,obj['native_head_rows']]-y[:,obj['native_head_rows']])**2*obj['coordinate_loss_weights'],axis=1))
        self.assertAlmostEqual(actual,expected)
        self.assertAlmostEqual(obj['weight_sum'],1.5)

    def test_uniform_weighted_error_anova_closes(self):
        cfg=LoadConfig(required_history=1);roles=coordinate_roles(cfg,'C');obj=requested_objective(cfg,roles,'uniform',1)
        rng=np.random.default_rng(94);p=rng.normal(size=(16,32,8));y=rng.normal(size=(16,32,8))
        report,_=whole_task_report(p,y,p[-1],y[-1],NAMES,obj)
        self.assertAlmostEqual(report['lattice_A_direct_objective'],np.mean((p[-1]-y[-1])**2))
        self.assertLess(abs(report['weighted_error_anova']['mse_identity_error']),1e-13)


class TorchProbe(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch runtime not installed')
    def test_mlp_repeatability_and_rng_preservation(self):
        import torch
        from decoding import fit_mlp
        torch.set_num_threads(1);torch.manual_seed(81)
        before=torch.random.get_rng_state().clone()
        pop,_,_=product_pairs();x=referenced_product(pop);y=pop[:,:1]
        a=fit_mlp(x,y,x,y,width=4,epochs=10,seed=3,weight_decays=(0.,))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        b=fit_mlp(x,y,x,y,width=4,epochs=10,seed=3,weight_decays=(0.,))
        np.testing.assert_array_equal(predict(a,x),predict(b,x))
        self.assertEqual(a['total_optimizer_updates'],10)


if __name__=='__main__':unittest.main()
