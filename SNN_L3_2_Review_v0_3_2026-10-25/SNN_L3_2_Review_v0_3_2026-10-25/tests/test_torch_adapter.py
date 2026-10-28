import importlib.util
import unittest
from dataclasses import replace
import numpy as np
from load_data import LoadConfig,generate,model_payload,make_source_pair
from experiment_plan import RunConfig,CELLS

TORCH_PRESENT=importlib.util.find_spec('torch') is not None
if TORCH_PRESENT:
    import torch
    from torch_adapter import LoadModel,task_loss,frozen_operations
    from snnbench.l2_runtime import StreamSession

@unittest.skipUnless(TORCH_PRESENT,'PyTorch unavailable; not counted as executed runtime checks')
class TorchContractChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_five_cells_forward_backward_parameters_and_streaming(self):
        data=generate(LoadConfig(),6);x=torch.from_numpy(model_payload(data)['x']);y=torch.from_numpy(data['y_all'])
        for cell in CELLS:
            torch.manual_seed(11);run=RunConfig(cell=cell,width=8);model=LoadModel(run)
            pred=model(x);loss,_=task_loss(pred,y,run);loss.backward()
            self.assertEqual(tuple(pred.shape),(6,20));self.assertEqual(sum(p.numel() for p in model.parameters()),run.parameters)
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads);self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            model.eval()
            with torch.no_grad():
                expected=model(x).clone();session=StreamSession(model.core)
                actual=session.run(model.encode(x))
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            self.assertEqual(sum(c.state_scalars() for c in model.core.backbone.cells),2*6*8)

    def test_zero_extra_weight_has_identical_optimization_across_l(self):
        left=RunConfig(data=LoadConfig(required_history=1),width=8,lambda_extra=0.)
        right=replace(left,data=LoadConfig(required_history=4))
        models=[]
        for run in (left,right):
            torch.manual_seed(19);m=LoadModel(run);b=generate(run.data,6)
            opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            loss,_=task_loss(m(torch.from_numpy(model_payload(b)['x'])),torch.from_numpy(b['y_all']),run)
            loss.backward();opt.step();models.append(m)
        for key,value in models[0].state_dict().items():
            torch.testing.assert_close(value,models[1].state_dict()[key],atol=0,rtol=0)

    def test_core_clear_removes_pair_history_at_last_readout(self):
        c=LoadConfig();b=generate(c,6);pair=make_source_pair(c,b,'H1')
        torch.manual_seed(20);m=LoadModel(RunConfig(data=c,width=8))
        a=frozen_operations(m,model_payload(b)['x']);d=frozen_operations(m,model_payload(pair)['x'])
        np.testing.assert_allclose(a['baseline'],a['noop'],atol=1e-6)
        np.testing.assert_allclose(a['clear_core'],d['clear_core'],atol=1e-6)

    def test_equal_demand_loss_is_uniform_and_classification_backpropagates(self):
        r=RunConfig(data=LoadConfig(k=1,required_history=3,content_mode='demand_only'),width=8,loss_mode='uniform')
        b=generate(r.data,6);pred=torch.randn(6,20,requires_grad=True);target=torch.from_numpy(b['y_all'])
        loss,_=task_loss(pred,target,r)
        wanted=(pred.reshape(6,5,4)-target).square()[:,[0,1,2,4],:1].mean()
        torch.testing.assert_close(loss,wanted)
        rc=replace(r,objective='classification');m=LoadModel(rc)
        from load_data import classification_labels
        loss,_=task_loss(m(torch.from_numpy(model_payload(b)['x'])),torch.from_numpy(classification_labels(rc.data,b)),rc)
        loss.backward();self.assertTrue(torch.isfinite(loss))

if __name__=='__main__':unittest.main()
