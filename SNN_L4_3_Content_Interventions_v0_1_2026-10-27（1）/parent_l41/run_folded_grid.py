import argparse
from pathlib import Path
import torch,numpy as np
from bridge35 import BenchmarkModel,ModelConfig,BACKBONES,NEURONS
from native_head_audit import parameter_hash
from folded_trajectory import FoldedTrajectorySession
from artifacts35 import write_json

def run():
    torch.set_num_threads(1);rows=[]
    for backbone in BACKBONES:
        for kind in NEURONS:
            for phase in ('output','pre','post'):
                torch.manual_seed(410288);image=backbone in ('cnn2d','sew_resnet18','spike_attention','unet')
                shape=(1,8,8) if image else (4,)
                model=BenchmarkModel(shape,(3,),3,ModelConfig(backbone=backbone,neuron=kind,width=2,observation=phase,readout='trajectory')).eval()
                x=torch.tensor(np.random.default_rng(410288).normal(size=(2,3,*shape)).astype(np.float32));before=parameter_hash(model)
                with torch.no_grad():native=model(x)
                folded=FoldedTrajectorySession(model);actual=folded.run(x)
                gap=float((native-actual).abs().max());assert gap<2e-5 and parameter_hash(model)==before
                contract=folded.readout_contract();assert contract['retained_trajectory_length']==0
                rows.append({'backbone':backbone,'cell':kind,'phase':phase,'max_abs':gap,'parameters_unchanged':True,'trained':False,**contract})
    return {'module':'L4.1 folded linear trajectory deployment','cases':rows,'case_count':288,
        'scope':'Same 288 trajectory interface configurations, a second deployment algorithm. Not 288 additional trained models. Linear fixed-T equivalence does not extend automatically to arbitrary nonlinear trajectory decoders.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();dst=Path(a.output)
    if dst.exists():raise FileExistsError(dst)
    write_json(dst,run());print('288 folded trajectory checks passed')
