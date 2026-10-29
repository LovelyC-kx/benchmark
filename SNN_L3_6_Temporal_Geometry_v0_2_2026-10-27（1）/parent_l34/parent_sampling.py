"""Separate estimation/selection/evaluation streams and six legal schedules."""
import numpy as np
from bridge import generate, _render, coordinate_roles

SCHEDULES = np.array([[2, 5, 8, 11, 12], [2, 5, 11, 8, 12],
                      [2, 8, 5, 11, 12], [2, 8, 11, 5, 12],
                      [2, 11, 5, 8, 12], [2, 11, 8, 5, 12]])


def fixed_batch(cfg, n, split, first_id, schedule_index=0):
    if schedule_index not in range(6): raise ValueError('One of six legal schedule indices required')
    batch = generate(cfg, n, split, sample_ids=np.arange(first_id, first_id+n))
    coordinate_roles(cfg, 'C', batch['source_present'])
    batch['event_times'] = np.tile(SCHEDULES[schedule_index], (n, 1))
    batch['x'] = _render(cfg, batch['latents'], batch['measurement_noise'],
                        batch['event_times'], batch['source_present'])[:, :cfg.deadline_steps]
    batch['meta'] = dict(batch['meta'], analysis_schedule_index=schedule_index)
    return batch


def repeat_panel(cfg, scenes, repeats, split, first_id, schedule_index, focal='C',
                 mode='focal_fixed', panel_seed=8301):
    if repeats < 2 or scenes < 2 or mode not in ('focal_fixed', 'measurement_only', 'identical'):
        raise ValueError('Known mode and >=2 independent scenes/repeats required')
    batch = fixed_batch(cfg, scenes, split, first_id, schedule_index)
    columns = coordinate_roles(cfg, focal, batch['source_present'])
    focal_columns = [(row['source_id'], row['coordinate']) for row in columns if row['role'] == 'focal']
    from bridge import SOURCES
    focal_source = SOURCES.index(focal)
    responses, latents = [], []
    for repetition in range(repeats):
        z, noise = batch['latents'].copy(), batch['measurement_noise'].copy()
        for i, scene_id in enumerate(batch['sample_ids']):
            # schedule index is deliberately omitted: paired schedules use the
            # same contents/noise, while schedule-dependent dynamics may differ.
            rng = np.random.default_rng(np.random.SeedSequence([cfg.data_seed, int(scene_id), panel_seed, repetition]))
            if mode == 'focal_fixed':
                z[i] = rng.normal(size=(5, 4))
                for j, k in focal_columns: z[i, j, k] = batch['latents'][i, j, k]
            if mode != 'identical': noise[i] = cfg.sigma*rng.normal(size=(5, 8))
        responses.append(_render(cfg, z, noise, batch['event_times'], batch['source_present'])[:, :cfg.deadline_steps])
        latents.append(z)
    latents = np.stack(latents, 1)
    targets = latents[:, 0, focal_source, :cfg.k]
    if not np.array_equal(latents[:, :, focal_source, :cfg.k], np.broadcast_to(targets[:, None], (scenes, repeats, cfg.k))):
        raise AssertionError('Focal target varied inside a conditional repeat scene')
    return {'x': np.stack(responses, 1), 'targets': targets, 'latents': latents,
            'scene_ids': batch['sample_ids'], 'schedule_index': schedule_index, 'mode': mode,
            'conditioning': 'requested focal coordinates and exact schedule' if mode == 'focal_fixed' else
                            ('all contents and exact schedule' if mode == 'measurement_only' else 'exact input')}
