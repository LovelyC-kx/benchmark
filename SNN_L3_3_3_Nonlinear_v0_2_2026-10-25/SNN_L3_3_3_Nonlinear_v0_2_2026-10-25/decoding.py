"""Legal-observation probes; feature maps and selection use no evaluation data."""
import time
import numpy as np

LAMBDAS = (0., 1e-5, 1e-3, .1, 1.)


def validate_xy(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2 or min(x.shape[1], y.shape[1]) < 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite matching matrix inputs/targets required')
    return x, y


def fit_scale(x):
    x = np.asarray(x, float); mean = x.mean(0)
    scale = float(np.sqrt(np.mean((x-mean)**2)))
    return mean, scale if scale > 0 else 1., scale == 0


def feature_map(x, mean, scale, family):
    x = (np.asarray(x, float)-mean)/scale
    if family == 'linear': return x
    if family != 'quadratic': raise ValueError('Linear or complete quadratic feature map required')
    i, j = np.triu_indices(x.shape[1])
    weights = np.where(i == j, 1., np.sqrt(2.))
    return np.concatenate((x, x[:, i]*x[:, j]*weights), axis=1)


def ridge(x, y, penalty):
    x, y = validate_xy(x, y)
    if not np.isfinite(penalty) or penalty < 0: raise ValueError('Nonnegative finite mean-SSE penalty required')
    xm, ym = x.mean(0), y.mean(0); xc, yc = x-xm, y-ym
    if penalty == 0:
        coef = np.linalg.lstsq(xc, yc, rcond=1e-10)[0]
    elif x.shape[1] > len(x):
        coef = xc.T @ np.linalg.solve(xc @ xc.T+len(x)*penalty*np.eye(len(x)), yc)
    else:
        coef = np.linalg.solve(xc.T @ xc+len(x)*penalty*np.eye(x.shape[1]), xc.T @ yc)
    return {'coef': coef, 'bias': ym-xm @ coef}


def fit_probe(train_x, train_y, select_x, select_y, family='linear', lambdas=LAMBDAS):
    start = time.perf_counter()
    x, y = validate_xy(train_x, train_y); vx, vy = validate_xy(select_x, select_y)
    if vx.shape[1] != x.shape[1] or vy.shape[1] != y.shape[1]: raise ValueError('Consistent selection schema required')
    mean, scale, degenerate = fit_scale(x)
    tx, sx = feature_map(x, mean, scale, family), feature_map(vx, mean, scale, family)
    candidates = []
    for lam in lambdas:
        f = ridge(tx, y, lam)
        loss = float(np.mean((sx @ f['coef']+f['bias']-vy)**2))
        candidates.append((loss, lam, f))
    loss, lam, f = min(candidates, key=lambda row: (row[0], row[1]))
    return dict(f, family=family, input_mean=mean, input_scale=scale,
        degenerate_scale=degenerate, selected_lambda=float(lam), selection_mse=loss,
        train_count=len(x), selection_count=len(vx), raw_observation_dimension=x.shape[1],
        total_labelled_scenes_for_fit_and_selection=len(x)+len(vx),
        total_labelled_target_scalars=(len(x)+len(vx))*y.shape[1],
        feature_dimension=tx.shape[1], parameter_count=int(f['coef'].size+f['bias'].size),
        hyperparameter_candidate_count=len(lambdas), fit_seconds=time.perf_counter()-start,
        scope='Finite probe learning rule, not a population information estimator; selection is independent of evaluation.')


def predict(probe, x):
    if probe['family'] == 'mlp':
        h = (np.asarray(x, float)-probe['input_mean'])/probe['input_scale']
        h = np.maximum(0., h @ probe['w1']+probe['b1'])
        return h @ probe['w2']+probe['b2']+probe['target_mean']
    return feature_map(x, probe['input_mean'], probe['input_scale'], probe['family']) @ probe['coef']+probe['bias']


def fit_mlp(train_x, train_y, select_x, select_y, seed=1, width=64, epochs=200, weight_decays=(0., .001)):
    """One initialization per call; every declared seed must be retained.

    Selection chooses weight decay and early-stopped epoch, never a seed.
    A separate width would be a separately registered readout budget.
    Exported NumPy weights permit deterministic evaluation without torch.
    """
    import torch
    from torch import nn
    start = time.perf_counter()
    x, y = validate_xy(train_x, train_y); vx, vy = validate_xy(select_x, select_y)
    if width < 1 or epochs < 1: raise ValueError('Positive MLP width and epochs required')
    if x.shape[1] != vx.shape[1] or y.shape[1] != vy.shape[1]: raise ValueError('Selection schema')
    mean, scale, degenerate = fit_scale(x); target_mean = y.mean(0)
    tx = torch.tensor((x-mean)/scale, dtype=torch.float32)
    ty = torch.tensor(y-target_mean, dtype=torch.float32)
    sx = torch.tensor((vx-mean)/scale, dtype=torch.float32)
    sy = torch.tensor(vy-target_mean, dtype=torch.float32)
    choices = []
    updates = 0
    with torch.random.fork_rng(devices=[]):
        for decay in weight_decays:
            if not np.isfinite(decay) or decay < 0: raise ValueError('Nonnegative weight decay')
            torch.random.default_generator.manual_seed(seed)
            model = nn.Sequential(nn.Linear(x.shape[1], width), nn.ReLU(), nn.Linear(width, y.shape[1]))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=decay)
            best_loss, best_state, best_epoch = float('inf'), None, None
            for epoch in range(epochs+1):
                if epoch % 10 == 0 or epoch == epochs:
                    with torch.no_grad(): loss = float(torch.mean((model(sx)-sy)**2))
                    if loss < best_loss:
                        best_loss, best_epoch = loss, epoch
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if epoch == epochs: break
                optimizer.zero_grad(); loss = torch.mean((model(tx)-ty)**2); loss.backward(); optimizer.step(); updates += 1
            choices.append((best_loss, decay, best_epoch, best_state))
    loss, decay, best_epoch, state = min(choices, key=lambda row: (row[0], row[1], row[2]))
    return {'family': 'mlp', 'input_mean': mean, 'input_scale': scale, 'degenerate_scale': degenerate,
        'target_mean': target_mean, 'w1': state['0.weight'].numpy().T.astype(float), 'b1': state['0.bias'].numpy().astype(float),
        'w2': state['2.weight'].numpy().T.astype(float), 'b2': state['2.bias'].numpy().astype(float),
        'seed': seed, 'width': width, 'max_epochs': epochs, 'selected_epoch': best_epoch,
        'selected_weight_decay': decay, 'selection_mse': loss,
        'train_count': len(x), 'selection_count': len(vx), 'raw_observation_dimension': x.shape[1],
        'total_labelled_scenes_for_fit_and_selection':len(x)+len(vx),
        'total_labelled_target_scalars':(len(x)+len(vx))*y.shape[1],
        'feature_dimension': width, 'parameter_count': (x.shape[1]+1)*width+(width+1)*y.shape[1],
        'hyperparameter_candidate_count': len(weight_decays), 'total_optimizer_updates': updates,
        'fit_seconds': time.perf_counter()-start, 'all_initialization_seeds_retained': True,
        'scope': 'Frozen-SNN nonlinear probe. Early stopping/weight decay selected only on selection data. No best-seed reporting.'}


def probe_metadata(probe):
    return {key: value for key, value in probe.items() if key not in {'coef','bias','input_mean','target_mean','w1','b1','w2','b2'}}


def mse(prediction, target):
    p, y = validate_xy(prediction, target)
    if p.shape != y.shape: raise ValueError('Matching prediction and target shape')
    return float(np.mean((p-y)**2))
