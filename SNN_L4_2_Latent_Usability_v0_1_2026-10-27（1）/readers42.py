"""Frozen-core reader families; separate label/search/parameter budgets."""
import numpy as np
import torch
from bridge42 import fit_head

def predict_linear(head, x):
    w, origin, bias = head
    return (np.asarray(x,float)-origin)@w+bias

def linear_bundle(head):
    return {'kind':'ridge','head':head,'parameters':int(head[0].size+head[2].size)}

def budget_hidden(d,k):
    h = (d*k)//(d+k+1)
    if h < 1:
        raise ValueError('Budget too small for this two-layer reader')
    return int(h)

class SmallReader(torch.nn.Module):
    def __init__(self,d,k,h):
        super().__init__()
        self.input = torch.nn.Linear(d,h)
        self.output = torch.nn.Linear(h,k)
    def forward(self,x):
        return self.output(torch.tanh(self.input(x)))

def predict(bundle, x):
    x = np.asarray(x, float)
    if bundle['kind'] == 'ridge':
        return predict_linear(bundle['head'], x)
    s = bundle['state']
    out = np.tanh(x@s['input.weight'].T+s['input.bias'])@s['output.weight'].T+s['output.bias']
    if bundle['kind'] == 'residual':
        out = out + predict_linear(bundle['base_head'],x)
    return out

def fit_mlp(x,y,selection_x,selection_y,kind,seed,base_head=None,steps=400):
    if kind not in ('budget_mlp','residual') or steps<1:
        raise ValueError('Named reader and positive fixed optimization budget')
    x,y = np.asarray(x,float),np.asarray(y,float)
    d,k = x.shape[1],y.shape[1]
    h = budget_hidden(d,k) if kind=='budget_mlp' else 4
    base_train = np.zeros_like(y) if base_head is None else predict_linear(base_head,x)
    if kind=='residual' and base_head is None:
        raise ValueError('Residual reader requires frozen same-permission affine base')
    tx = torch.tensor(x,dtype=torch.float32)
    ty = torch.tensor(y-base_train,dtype=torch.float32)
    best = None; history = []; snapshots = sorted(set((0,steps//4,steps//2,steps)))
    for lr in (.005,.02):
        torch.manual_seed(int(seed))
        model=SmallReader(d,k,h)
        if kind=='residual':
            torch.nn.init.zeros_(model.output.weight);torch.nn.init.zeros_(model.output.bias)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
        for step in range(steps+1):
            if step in snapshots:
                bundle={'kind':kind,'state':{n:p.detach().cpu().numpy().copy() for n,p in model.state_dict().items()},
                    'input_dimension':d,'output_dimension':k,'hidden':h,
                    'trained_parameters':sum(p.numel() for p in model.parameters()),
                    'parameters':sum(p.numel() for p in model.parameters())+(0 if base_head is None else base_head[0].size+base_head[2].size),
                    'lr':lr,'selected_step':step,'initialization_seed':int(seed)}
                if base_head is not None:bundle['base_head']=base_head
                val=float(np.mean((predict(bundle,selection_x)-selection_y)**2))
                history.append({'lr':lr,'step':step,'selection_MSE':val})
                if best is None or val<best[0]:best=(val,bundle)
            if step==steps:break
            optimizer.zero_grad();loss=((model(tx)-ty)**2).mean();loss.backward();optimizer.step()
    return best[1],{'candidates':history,'optimizer_updates':2*steps,'optimization_fits':2,
        'selection_objective':'Mean squared error on all eight original continuous targets; auxiliary class loss never selects this reader.'}

def ridge_optimization_control(x,y,alpha,native_w,native_b,steps=2000):
    """Cold/warm GD toward the SAME regularized empirical affine optimum.

    Objective = mean over scenes of sum over outputs squared error + alpha||W||F^2.
    Intercept analytically eliminated; no changed whitening/preconditioner.
    """
    x,y=np.asarray(x,float),np.asarray(y,float)
    w,xm,ym=fit_head(x,y,alpha);u=x-xm;v=y-ym
    h=u.T@u/len(x)+alpha*np.eye(x.shape[1]);c=u.T@v/len(x)
    rate=1/float(np.linalg.eigvalsh(h).max())
    def objective(a):return float(np.mean(np.sum((u@a-v)**2,axis=1))+alpha*np.sum(a*a))
    target=objective(w);rows={}
    for name,start in [('cold',np.zeros_like(w)),('native',np.asarray(native_w,float).copy())]:
        a=start;initial=objective(a)
        for _ in range(steps):a-=rate*(h@a-c)
        rows[name]={'initial_centered_regularized_objective':initial,'final_centered_regularized_objective':objective(a),
            'excess_over_closed_form':objective(a)-target,'weight_distance_to_closed_form':float(np.linalg.norm(a-w)),
            'optimized_intercept':ym-xm@a}
    native_raw=float(np.mean((x@native_w+native_b-y)**2))
    return {'alpha':alpha,'steps':steps,'fixed_step':rate,'closed_form_objective':target,
        'condition_number':float(np.linalg.cond(h)),'initial_native_unregularized_MSE':native_raw,'runs':rows,
        'scope':'Frozen-core affine empirical optimization control; does not establish full-network training sufficiency.'}
