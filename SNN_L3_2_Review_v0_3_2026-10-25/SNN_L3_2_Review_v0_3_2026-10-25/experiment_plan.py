"""Staged exploratory matrix. Generating the plan never executes training."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from load_data import LoadConfig

CELLS = ('continuous', 'threshold', 'analog_hard_ref', 'lif_hard', 'lif_soft')
SPLIT_SIZES = {'encoder_train':6144, 'encoder_val':1536, 'geometry_fit':1536,
               'probe_train':6144, 'probe_val':1536, 'audit_test':3072, 'development_test':1536}

@dataclass(frozen=True)
class RunConfig:
    data: LoadConfig = field(default_factory=LoadConfig)
    cell: str = 'lif_hard'
    width: int = 32
    seed: int = 11
    lambda_extra: float = 1.
    objective: str = 'regression'
    epochs: int = 60
    encoder_mode: str = 'learned'
    loss_mode: str = 'anchored'

    def validate(self):
        self.data.validate()
        if self.cell not in CELLS or self.width not in (8,16,32,64):raise ValueError('Cell/width')
        if self.objective not in ('regression','classification'):raise ValueError('Objective')
        if self.loss_mode not in ('anchored','uniform'):raise ValueError('Loss convention')
        if self.loss_mode=='uniform' and self.lambda_extra!=1.:raise ValueError('Uniform objective has no extra-group weight')
        if self.encoder_mode not in ('learned','frozen'):raise ValueError('Encoder convention')
        if self.seed<0 or self.epochs<1 or not 0<=self.lambda_extra<1e6:raise ValueError('Run fields')

    def run_id(self):
        self.validate()
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:16]

    @property
    def output_dim(self):return 20 if self.objective=='regression' else 5

    @property
    def parameters(self):
        q=self.output_dim; w=self.width
        return w*w+(66+q)*w+5120+q

def build_plan():
    rows={}
    def add(run,stage):
        key=run.run_id()
        if key not in rows:
            rows[key]={'run_id':key,'config':asdict(run),'stages':[],
                       'parameters':run.parameters,'cell_state_scalars_per_sample':2*run.width,
                       'primary_observation_dim':run.width,
                       'status':'planned_not_executed'}
        rows[key]['stages'].append(stage)
    for cell in CELLS:
        for l in (1,2,4):add(RunConfig(data=LoadConfig(required_history=l),cell=cell),'screen15')
    for seed in (11,22,33):
        for cell in CELLS:
            for k in (1,2,4):
                for l in (1,2,4):add(RunConfig(data=LoadConfig(k=k,required_history=l),cell=cell,seed=seed),'primary135')
            for width in (8,16,64):
                for l in (1,4):add(RunConfig(data=LoadConfig(required_history=l),cell=cell,width=width,seed=seed),'width_control')
            for delay in (1,7,11):add(RunConfig(data=LoadConfig(required_history=1,blank_delay=delay),cell=cell,seed=seed),'blank_age_control')
            for writes in (0,1):add(RunConfig(data=LoadConfig(required_history=1,extra_writes=writes),cell=cell,seed=seed),'write_pressure_control')
            for k in (1,2):
                for l in (1,4):add(RunConfig(data=LoadConfig(k=k,required_history=l,energy_mode='raw'),cell=cell,seed=seed),'energy_control')
            for weight in (.25,4.):add(RunConfig(data=LoadConfig(required_history=4),cell=cell,seed=seed,lambda_extra=weight),'loss_weight_control')
            for k,l in ((2,1),(1,3),(2,3)):
                add(RunConfig(data=LoadConfig(k=k,required_history=l,content_mode='demand_only'),cell=cell,seed=seed,loss_mode='uniform'),'equal_total_demand_control')
    return {'schema':'l3_2_plan_0.2','status':'exploratory proposal; no jobs executed',
            'run_count':len(rows),'screen_count':sum('screen15' in r['stages'] for r in rows.values()),
            'primary_count':sum('primary135' in r['stages'] for r in rows.values()),
            'split_sizes':SPLIT_SIZES,'runs':list(rows.values()),
            'equal_total_demand_pairs':{'M4':[[2,1],[1,3]],'M8':[[4,1],[2,3]],
                'input':'all four latent coordinates visible at every source; identical raw input across K/L',
                'loss':'uniform requested-coordinate MSE',
                'reuse':'K4,L1 physical_k anchored is exactly the same input and loss as demand_only uniform; reuse primary run, do not count an extra fit'},
            'conditional_extensions_not_in_count':['frozen shared encoder 2x2 control','fixed source schedule control',
                'nonlinear probes','end-to-end binary classification confirmation','image reconstruction transfer',
                'auxiliary-state and other-backbone confirmation'],
            'decision_rule':'Run screen first; inspect task solvability, training loss, event exposure and scale before expanding. Never treat all planned jobs as completed.'}

def config_from_dict(d):
    d=dict(d);d['data']=LoadConfig(**d['data']);return RunConfig(**d)

if __name__=='__main__':
    from pathlib import Path
    root=Path(__file__).resolve().parent
    plan=build_plan();(root/'experiment_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:plan[k] for k in ('run_count','screen_count','primary_count','status')},indent=2))
