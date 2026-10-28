"""Separate identical-input, measurement and focal-conditional variability."""
import numpy as np
from inherited import SOURCES, SPLITS, generate, _render
from roles import coordinate_roles

MODES = ('identical_input', 'measurement_only', 'focal_fixed_full')


def repeat_panel(cfg, scenes=256, repeats=8, split='development_test', focal='C',
                 mode='measurement_only', first_scene_id=100000, panel_seed=3101):
    if (scenes < 2 or repeats < 2 or split not in ('geometry_fit', 'development_test')
            or mode not in MODES or first_scene_id < 0 or panel_seed < 0):
        raise ValueError('Known independent development split, >=2 scenes/repeats, legal mode/seeds required')
    base = generate(cfg, scenes, split, sample_ids=np.arange(first_scene_id, first_scene_id+scenes))
    coordinates = coordinate_roles(cfg, focal, base['source_present'])
    focal_indices = [(SOURCES.index(row['source']), row['coordinate']) for row in coordinates if row['role'] == 'focal']
    xs, zs, noises = [], [], []
    for repetition in range(repeats):
        z = base['latents'].copy()
        noise = base['measurement_noise'].copy()
        for i, scene_id in enumerate(base['sample_ids']):
            rng = np.random.default_rng(np.random.SeedSequence(
                [cfg.data_seed, SPLITS[split], int(scene_id), 30000, panel_seed, repetition]))
            if mode == 'focal_fixed_full':
                z[i] = rng.normal(size=(5, 4))
                for source_id, coordinate in focal_indices:
                    z[i, source_id, coordinate] = base['latents'][i, source_id, coordinate]
            if mode != 'identical_input':
                noise[i] = cfg.sigma*rng.normal(size=(5, 8))
        xs.append(_render(cfg, z, noise, base['event_times'], base['source_present']))
        zs.append(z)
        noises.append(noise)
    return {'x': np.stack(xs, 1), 'latents': np.stack(zs, 1),
            'measurement_noise': np.stack(noises, 1),
            'event_times': base['event_times'], 'sample_ids': base['sample_ids'],
            'mode': mode, 'deadline_steps': cfg.deadline_steps,
            'conditioned_on': 'all contents and schedule' if mode == 'measurement_only' else
                              ('exact input' if mode == 'identical_input' else 'requested focal coordinates and schedule'),
            'schedule_is_fixed_within_scene': True, 'audit_accessed': False}


def within_scene_covariance(responses):
    """Unbiased average conditional covariance from iid repeats, not OLS residual."""
    r = np.asarray(responses, dtype=float)
    if r.ndim != 3 or min(r.shape[:2]) < 2 or not np.isfinite(r).all():
        raise ValueError('Finite [scenes,repeats,state] required')
    centered = r-r.mean(1, keepdims=True)
    cov = np.einsum('gbi,gbj->ij', centered, centered)/(r.shape[0]*(r.shape[1]-1))
    return {'covariance': cov, 'trace': float(np.trace(cov)),
            'scene_mean_estimation_covariance': cov/r.shape[1],
            'scene_count': r.shape[0], 'repeat_count': r.shape[1],
            'note': 'Requires iid repeat draws under the declared conditioning; repeated means remain noisy.'}
