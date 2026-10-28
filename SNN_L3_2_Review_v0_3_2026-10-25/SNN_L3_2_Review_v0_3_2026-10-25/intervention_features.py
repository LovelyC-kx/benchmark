"""Frozen-checkpoint operations with matched final-state observations.

These are diagnostic forward interventions. They do not retrain the encoder,
core, head or probe, and do not estimate a causal mediation fraction.
"""
import copy
import numpy as np
import torch
from torch_adapter import StreamSession

OBSERVATIONS = ('last_post', 'joint_post', 'last_pre', 'last_output')


def _current_features(model, prediction):
    first, last = model.core.backbone.cells
    values = {
        'prediction': prediction,
        'last_post': last.last['post'],
        'joint_post': torch.cat((first.last['post'], last.last['post']), dim=1),
        'last_pre': last.last['pre'],
        'last_output': last.last['output'],
    }
    return {key: value.detach().cpu().numpy().copy() for key, value in values.items()}


@torch.no_grad()
def collect_operation_features(model, x, cut=12, batch_size=128):
    """Clear after step 11 and before C at 12; recompute all natural futures.

    baseline/noop/clear_first_cell/clear_core share the same prefix snapshot.
    Reset feedback suppression reruns the full sequence in a fresh copied model.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or not len(x) or not 0 < cut < x.shape[1] or batch_size < 1:
        raise ValueError('Nonempty B,T,F input, legal cut and positive batch size required')
    model.eval()
    device = next(model.parameters()).device
    reset_control = None
    if model.run_config.cell in ('analog_hard_ref', 'lif_hard', 'lif_soft'):
        reset_control = copy.deepcopy(model).eval()
        reset_control.core.cfg.reset_strength = 0.
        for cell in reset_control.core.backbone.cells:
            cell.reset_strength = 0.
    chunks = {}
    for start in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[start:start+batch_size], device=device)
        encoded = model.encode(xb)
        session = StreamSession(model.core)
        for t in range(cut):
            session.step(encoded[:, t])
        prefix = session.snapshot()
        for operation in ('baseline', 'noop', 'clear_first_cell', 'clear_core'):
            if operation != 'baseline':
                session.restore(prefix)
            if operation == 'clear_first_cell':
                session.clear('cell', paths=['backbone.cells.0'])
            elif operation == 'clear_core':
                session.clear('core')
            pred = session.run(encoded[:, cut:])
            for key, value in _current_features(model, pred).items():
                chunks.setdefault(operation, {}).setdefault(key, []).append(value)
        if reset_control is not None:
            pred = reset_control(xb)
            for key, value in _current_features(reset_control, pred).items():
                chunks.setdefault('reset_feedback_disabled_full_sequence', {}).setdefault(key, []).append(value)
    return {op: {key: np.concatenate(value) for key, value in values.items()}
            for op, values in chunks.items()}
