"""Write compact known-answer examples. No SNN scientific results."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from deep_diagnostics import (gaussian_readout_reference, pickfreeze_statistics,
    repeated_signal_and_noise, repeated_prediction_risk, scalar_lif)


def calibrate():
    af, ao = np.array([[1.], [0.]]), np.array([[0.], [1.]])
    shear = np.array([[1., 2.], [0., 1.]])
    noise = .04*np.eye(2)
    first = gaussian_readout_reference(af, ao, noise)
    second = gaussian_readout_reference(shear @ af, shear @ ao, shear @ noise @ shear.T)
    cosine = float(((shear @ af).T @ (shear @ ao)).item()/
                   (np.linalg.norm(shear @ af)*np.linalg.norm(shear @ ao)))
    aligned = gaussian_readout_reference(af, np.empty((2, 0)), np.diag([1., .01]))
    orthogonal = gaussian_readout_reference(af, np.empty((2, 0)), np.diag([.01, 1.]))
    grid = np.asarray(list(itertools.product([-1., 1.], repeat=4)))
    x, y, xp, yp = grid.T
    product = pickfreeze_statistics((x*y)[:, None], (xp*yp)[:, None], (x*yp)[:, None])
    rng = np.random.default_rng(713)
    focal = rng.normal(size=(30000, 1))
    p = .5*(focal[:, None]+rng.normal(size=(30000, 8, 1))+.2*rng.normal(size=(30000, 8, 1)))
    repeated = repeated_prediction_risk(p, focal)
    signal = rng.normal(size=(30000, 1, 1))
    responses = signal+.8*rng.normal(size=(30000, 4, 1))
    correction = repeated_signal_and_noise(responses)
    jumps = {}
    for kind in ('hard', 'soft'):
        jumps[kind] = float(scalar_lif([1+1e-6], kind=kind)['post'][-1]-
                            scalar_lif([1-1e-6], kind=kind)['post'][-1])
    base_currents = np.full(15, .3)
    changed_currents = base_currents.copy(); changed_currents[0] += .4
    base_phase = scalar_lif(base_currents)
    changed_phase = scalar_lif(changed_currents)
    result = {'kind': 'known_answer_calibrations', 'new_snn_training_runs': 0,
        'invertible_mixing': {'angle_before_degrees': 90.,
            'angle_after_degrees': float(np.degrees(np.arccos(cosine))),
            'mmse_before': first['mse_per_focal_coordinate'],
            'mmse_after': second['mse_per_focal_coordinate'],
            'maximum_transported_decoder_difference': float(np.max(np.abs(second['w'] @ shear-first['w']))),
            'warning': 'Noise and readout transformed together; this is a coordinate control, not a biological perturbation.'},
        'equal_noise_trace': {'trace_both': 1.01,
            'parallel_noise_mmse': aligned['mse_per_focal_coordinate'],
            'orthogonal_noise_mmse': orthogonal['mse_per_focal_coordinate']},
        'pure_product_pickfreeze': product,
        'conditional_readout_risk': {'expected_conditional_mean_error': .25,
            'expected_conditional_variability': .26, 'estimate': repeated},
        'repeat_mean_bias': {'true_signal_variance': 1.,
            'expected_raw_mean_variance': 1+.64/4,
            **{key: float(correction[key][0, 0]) for key in
                ('raw_between_covariance', 'corrected_between_covariance', 'cross_replica_covariance')}},
        'discrete_reset_jumps': jumps,
        'history_in_event_phase': {'beta': .8, 'threshold': 1., 'bias_current': .3,
            'initial_current_change': .4,
            'base_reset_indices': np.flatnonzero(base_phase['events']).tolist(),
            'changed_reset_indices': np.flatnonzero(changed_phase['events']).tolist(),
            'terminal_post_states': [float(base_phase['post'][-1]), float(changed_phase['post'][-1])],
            'direct_initial_current_kernel_survival': [int(np.prod(1-base_phase['events'])), int(np.prod(1-changed_phase['events']))],
            'meaning': 'Initial direct contributions are zero on both realized paths; different natural event times preserve a phase distinction. This is not guaranteed memory under every condition.'},
        'observation_permission': {'spike_counts': [2, 2],
            'leaky_terminal_states': [float(scalar_lif(x, beta=.5, kind='none')['post'][-1])
                                     for x in ([1, 0, 1, 0], [0, 1, 0, 1])],
            'warning': 'Temporal integration sees timing which total count discards; no data-processing paradox.'}}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new calibration output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibrate(), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(str(args.output.resolve()))
