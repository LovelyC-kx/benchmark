"""Exact frozen fixed-T linear trajectory head with bounded output accumulation."""
import torch
from bridge35 import StreamSession

class FoldedTrajectorySession:
    def __init__(self,model):
        if model.cfg.readout!='trajectory' or model.image_output or not isinstance(model.head,torch.nn.Linear):
            raise ValueError('Explicit fixed-T vector linear trajectory head required')
        if model.head.in_features%model.T:raise ValueError('Head dimension not divisible by T')
        self.session=StreamSession(model,capture=False,retain_trajectory=False)
        self.model=model;self.width=model.head.in_features//model.T;self.accumulator=None

    @torch.no_grad()
    def step(self,xt):
        time=self.session.cursor;feature=self.session.step(xt)
        if len(self.session.features)!=1:raise ValueError('Unexpected internal retained trajectory')
        retained=self.session.features.pop()
        if retained.ndim!=2 or retained.shape[-1]!=self.width:raise ValueError('Deployed per-step vector feature dimension')
        w=self.model.head.weight[:,time*self.width:(time+1)*self.width]
        value=torch.nn.functional.linear(retained,w,None)
        self.accumulator=value.clone() if self.accumulator is None else self.accumulator+value
        return feature

    @torch.no_grad()
    def finish(self):
        if self.session.cursor!=self.model.T:raise ValueError('Fixed-T head requires exactly T observations')
        return self.accumulator if self.model.head.bias is None else self.accumulator+self.model.head.bias

    def run(self,x):
        for time in range(x.shape[1]):self.step(x[:,time])
        return self.finish()

    def readout_contract(self):
        return {'retained_trajectory_length':len(self.session.features),
            'output_accumulator_shape':list(self.accumulator.shape) if self.accumulator is not None else None,
            'output_accumulator_tensor_bytes':self.accumulator.numel()*self.accumulator.element_size() if self.accumulator is not None else 0,
            'per_step_feature_dimension':self.width,'head_parameters':sum(p.numel() for p in self.model.head.parameters()),
            'clock':self.session.cursor,'fixed_horizon':self.model.T,
            'scope':'Output accumulator plus current feature and core/cache state. No retained T-feature history; parameters/time-indexed access and Python bookkeeping remain. No peak/hardware energy claim.'}
