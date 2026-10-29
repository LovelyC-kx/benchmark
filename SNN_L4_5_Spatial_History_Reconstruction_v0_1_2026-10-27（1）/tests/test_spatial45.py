import unittest
import numpy as np
import torch
from bridge45 import model_for,site_paths,parameter_hash
from spatial_data45 import batch,replace_source,region_masks,align_observations,latest_visible,render
from spatial_metrics45 import regional_losses,frequency_losses,psnr_summary,spectral_angle,bounded_interval
from spatial_readers45 import ridge_fit,ridge_predict,fit_decoder,decode,aggregate
from spatial_runtime45 import run,state,additive,masked_replace,Edit
from run_spatial_main45 import norm_match,contrast

torch.set_num_threads(1)

class DataAndMetrics(unittest.TestCase):
    def setUp(self):self.a=batch(8,45400000,noise=0);self.b=batch(8,45410000,noise=0)
    def test_deterministic(self):self.assertTrue(np.array_equal(self.a['x'],batch(8,45400000,noise=0)['x']))
    def test_distinct_scenes(self):self.assertFalse(np.array_equal(self.a['x'],self.b['x']))
    def test_input_contract(self):self.assertEqual(self.a['contract']['model_fields'],['x']);self.assertEqual(self.a['x'].shape,(8,6,5,16,16))
    def test_target_range(self):self.assertTrue(np.all((self.a['y']>=0)&(self.a['y']<=1)))
    def test_mask_partition(self):
        m=self.a['masks'];self.assertTrue(np.all(m['C'].astype(int)+m['H']+m['U_vis']==1))
    def test_changed_is_subset_current(self):self.assertTrue(np.all(~self.a['masks']['changed_C']|self.a['masks']['C']))
    def test_stale_same_target(self):
        q=replace_source(self.a,self.b,'U_src');self.assertTrue(np.array_equal(q['y'],self.a['y']));self.assertTrue(np.array_equal(q['x'][:,-1],self.a['x'][:,-1]))
    def test_H_changes_only_history(self):
        q=replace_source(self.a,self.b,'H');self.assertTrue(np.array_equal(q['x'][:,-1],self.a['x'][:,-1]));self.assertFalse(np.array_equal(q['x'][:,2],self.a['x'][:,2]))
    def test_alignment(self):self.assertTrue(np.array_equal(align_observations(batch(8,45400000,noise=0,motion=True)),self.a['x']))
    def test_exact_history_reference(self):
        r=regional_losses(latest_visible(self.a['x']),self.a['y'],self.a['masks']);self.assertEqual(float(r['H'].max()),0)
    def test_empty_region(self):self.assertIsNone(regional_losses(self.a['y'],self.a['y'],{'x':np.zeros((16,16),bool)})['x'])
    def test_invalid_mask(self):
        with self.assertRaises(ValueError):regional_losses(self.a['y'],self.a['y'],{'x':np.zeros((2,2),bool)})
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):regional_losses(self.a['y']*np.nan,self.a['y'],{})
    def test_parseval(self):
        f=frequency_losses(np.zeros_like(self.a['y']),self.a['y'],self.a['masks']);self.assertLess(float(np.max(np.abs(f['parseval_residual']))),1e-13)
    def test_region_frequency_cross(self):
        r=regional_losses(np.zeros_like(self.a['y']),self.a['y'],self.a['masks']);f=frequency_losses(np.zeros_like(self.a['y']),self.a['y'],self.a['masks'])
        self.assertTrue(np.allclose(r['H'],f['H_low']+f['H_high']+f['H_cross']))
    def test_psnr_order(self):
        s=psnr_summary([.001,.1]);self.assertGreater(s['mean_scene_PSNR'],s['PSNR_of_mean_MSE'])
    def test_zero_psnr(self):self.assertIsNone(psnr_summary([0])['mean_scene_PSNR'])
    def test_sam_undefined(self):self.assertEqual(spectral_angle(np.zeros((1,4,2,2)),np.ones((1,4,2,2)))['scene_mean_radians'],[None])
    def test_bounded_scene_count(self):self.assertEqual(bounded_interval(np.zeros(5))['n_scenes'],5)
    def test_unbounded_rejected(self):
        with self.assertRaises(ValueError):bounded_interval(np.array([2.]))
    def test_ratio_floor(self):self.assertIsNone(contrast(*[np.zeros(3)]*4)['recovery_ratio'])
    def test_ratio_not_clipped(self):
        z=np.zeros(3);self.assertEqual(contrast(z+.5,z+1.5,z+.5,z)['recovery_ratio'],1.5)
    def test_negative_losses_rejected(self):
        with self.assertRaises(ValueError):contrast(np.zeros(3),np.ones(3),np.zeros(3),-np.ones(3))
    def test_norm_per_scene(self):
        x=np.ones((4,2,2,2));y=x*np.arange(1,5)[:,None,None,None];r,n=norm_match(x,y);self.assertEqual(n,0);self.assertTrue(np.allclose(np.linalg.norm(r.reshape(4,-1),axis=1),np.linalg.norm(y.reshape(4,-1),axis=1)))
    def test_zero_direction_flagged(self):
        r,n=norm_match(np.zeros((2,3)),np.ones((2,3)));self.assertEqual(n,2);self.assertTrue(np.all(r==0))

class DecoderAndRuntime(unittest.TestCase):
    def setUp(self):self.a=batch(4,45420000);self.m=model_for('cnn2d','lif_hard');self.paths=site_paths(self.m)
    def test_ridge_affine(self):
        g=np.random.default_rng(9);x=g.normal(size=(100,3));y=x@np.ones((3,2))+.5;p=ridge_predict(ridge_fit(x,y,1e-9),x);self.assertLess(float(np.max(np.abs(p-y))),1e-6)
    def test_spatial_decoder_shape(self):
        f=np.random.default_rng(9).normal(size=(4,3,16,16));r=fit_decoder(f,self.a['y'],'spatial3',.01);self.assertEqual(decode(r,f).shape,self.a['y'].shape)
    def test_gap_decoder_shape(self):
        f=np.random.default_rng(9).normal(size=(4,3,16,16));r=fit_decoder(f,self.a['y'],'gap_dense',.01);self.assertEqual(decode(r,f).shape,self.a['y'].shape)
    def test_access_mean(self):
        v=np.arange(6)[None,:,None,None,None];self.assertEqual(float(aggregate(v,'mean').ravel()[0]),2.5)
    def test_noop_exact(self):
        a=run(self.m,self.a['x'],sites=self.paths);b=run(self.m,self.a['x'],[Edit(self.paths[-1],3,lambda h,s:h,'noop')],sites=self.paths);self.assertTrue(np.array_equal(a['features'],b['features']))
    def test_terminal_post_changes_observation(self):
        a=run(self.m,self.a['x'],sites=self.paths);delta=np.ones((4,8,16,16))*.1;b=run(self.m,self.a['x'],[additive(self.paths[-1],5,delta)],sites=self.paths)
        self.assertGreater(float(np.max(np.abs(a['features'][:,-1]-b['features'][:,-1]))),.09)
    def test_first_post_does_not_retroactively_emit(self):
        a=run(self.m,self.a['x'],sites=self.paths);delta=np.ones((4,8,16,16));b=run(self.m,self.a['x'],[additive(self.paths[0],5,delta)],sites=self.paths)
        self.assertTrue(np.array_equal(a['features'][:,-1],b['features'][:,-1]))
    def test_inverse_same_stage(self):
        a=run(self.m,self.a['x'],sites=self.paths);delta=np.ones((4,8,16,16))*.2
        b=run(self.m,self.a['x'],[additive(self.paths[-1],3,delta,'a'),additive(self.paths[-1],3,-delta,'b')],sites=self.paths)
        self.assertLess(float(np.max(np.abs(a['features']-b['features']))),1e-6)
    def test_unknown_site(self):
        with self.assertRaises(ValueError):run(self.m,self.a['x'],[Edit('unknown',3,lambda h,s:h,'x')])
    def test_absent_owner(self):
        with self.assertRaises(ValueError):run(self.m,self.a['x'],[Edit(self.paths[-1],3,lambda h,s:h,'x','a')])
    def test_hook_cleanup_after_failure(self):
        with self.assertRaises(ValueError):run(self.m,self.a['x'],[Edit(self.paths[-1],3,lambda h,s:h*float('nan'),'x')])
        self.assertTrue(all(len(m._forward_hooks)==0 for m in self.m.modules()))
    def test_weights_unchanged(self):
        h=parameter_hash(self.m);run(self.m,self.a['x'],sites=self.paths);self.assertEqual(h,parameter_hash(self.m))
    def test_UNet_spatial_axes(self):
        m=model_for('unet','alif_ref');v=run(m,self.a['x'],sites=site_paths(m));self.assertEqual(v['features'].shape,(4,6,8,16,16));self.assertEqual(state(v,site_paths(m)[-1],3,'a').shape,(4,8,16,16))

class RealInputBoundary(unittest.TestCase):
    def sample(self):
        return {'x':np.zeros((3,4,8,8)),'y':np.ones((3,4,8,8))*.5,'masks':np.zeros((3,1,8,8)),
            'position_days':np.array([0.,3.,10.]),'availability_days':np.array([0.,3.,10.]),'source_id':'ROI_fixture','target_is_verified_clean':True}
    def test_causal_prefix(self):
        from real_reconstruction_contract45 import from_utilise_sample
        v=from_utilise_sample(self.sample(),target_index=1);self.assertEqual(v['x'].shape,(2,5,8,8));self.assertEqual(v['contract']['selected_indices'],[0,1])
    def test_offline_permission_separate(self):
        from real_reconstruction_contract45 import from_utilise_sample
        v=from_utilise_sample(self.sample(),target_index=1,mode='offline');self.assertEqual(len(v['x']),3);self.assertTrue(v['contract']['future_inputs_allowed'])
    def test_derived_future_support_rejected(self):
        from real_reconstruction_contract45 import from_utilise_sample
        s=self.sample();s['availability_days'][0]=9
        with self.assertRaises(ValueError):from_utilise_sample(s,target_index=1)
    def test_mask_convention(self):
        from real_reconstruction_contract45 import from_utilise_sample
        s=self.sample();s['masks'][1,0,:4]=1;v=from_utilise_sample(s,target_index=1)
        self.assertTrue(np.all(v['x'][1,4,:4]==0))
    def test_no_H_without_compatibility(self):
        from real_reconstruction_contract45 import from_utilise_sample
        v=from_utilise_sample(self.sample(),target_index=1);self.assertNotIn('H_target_compatible',v['analysis'])
    def test_source_split_leakage_rejected(self):
        from real_reconstruction_contract45 import validate_groups
        with self.assertRaises(ValueError):validate_groups(['same'],['val'],['same'])
    def test_availability_must_be_supplied(self):
        from real_reconstruction_contract45 import from_utilise_sample
        s=self.sample();del s['availability_days']
        with self.assertRaises(ValueError):from_utilise_sample(s,target_index=1)
    def test_explicit_compatibility(self):
        from real_reconstruction_contract45 import from_utilise_sample
        s=self.sample();s['masks'][1,0,:4]=1;c=np.ones((3,8,8),bool);v=from_utilise_sample(s,target_index=1,compatible_with_target=c)
        self.assertTrue(np.all(v['analysis']['H_target_compatible'][:4]))
    def test_unverified_target_rejected(self):
        from real_reconstruction_contract45 import from_utilise_sample
        s=self.sample();s['target_is_verified_clean']=False
        with self.assertRaises(ValueError):from_utilise_sample(s,target_index=1)
    def test_standard_SSIM_identity(self):
        from run_spatial_quality45 import ssim_scenes
        a=batch(2,45441000);self.assertTrue(np.all(ssim_scenes(a['y'],a['y'])==1))
    def test_standard_SSIM_nonfinite(self):
        from run_spatial_quality45 import ssim_scenes
        a=batch(2,45441000)
        with self.assertRaises(ValueError):ssim_scenes(a['y']*np.nan,a['y'])

if __name__=='__main__':unittest.main()
