"""Legal independent factor groups; only rendered x is sent to the SNN."""
import numpy as np
from bridge import _render, coordinate_roles
from inherited import SPLITS
from parent_sampling import fixed_batch, SCHEDULES

ROLE_TO_NAME = {'focal': 'F', 'cotarget': 'G', 'unrequested': 'U'}


def factor_schema(cfg, focal, source_present=None):
    rows = coordinate_roles(cfg, focal, source_present)
    groups = {name: [(r['source_id'], r['coordinate']) for r in rows if ROLE_TO_NAME[r['role']] == name] for name in ('F', 'G', 'U')}
    names = tuple(name for name, coords in groups.items() if coords)
    if cfg.sigma > 0: names += ('E',)
    return {'names': names, 'coordinates': groups, 'coordinate_roles': rows,
            'conditioning': 'exact presentation schedule, source presence, model and observation',
            'distribution': 'independent Gaussian visible latent coordinates and independent whole measurement tensor',
            'measurement_noise_is_explicit': cfg.sigma > 0,
            'inactive_groups': [name for name in ('F', 'G', 'U', 'E') if name not in names]}


def splice(cfg, a, b, names, coordinates, mask):
    if mask not in range(2**len(names)) or a['latents'].shape != b['latents'].shape:
        raise ValueError('Valid mask and matched factor batches required')
    if not np.array_equal(a['event_times'], b['event_times']) or not np.array_equal(a['source_present'], b['source_present']):
        raise ValueError('Schedule and presence must remain fixed across hybrids')
    z, e = b['latents'].copy(), b['measurement_noise'].copy()
    for bit, name in enumerate(names):
        if not mask & (1 << bit): continue
        if name == 'E': e = a['measurement_noise'].copy()
        else:
            for source, coordinate in coordinates[name]:
                z[:, source, coordinate] = a['latents'][:, source, coordinate]
    return {'latents': z, 'measurement_noise': e, 'event_times': a['event_times'],
            'source_present': a['source_present'],
            'x': _render(cfg, z, e, a['event_times'], a['source_present'])[:, :cfg.deadline_steps]}


def full_lattice(cfg, n, split, first_a, first_b, schedule_index=0, focal='C'):
    if abs(first_a-first_b) < n:
        raise ValueError('A and B scene-ID ranges must be disjoint')
    if split == 'audit_test': raise ValueError('Audit access is excluded from this analysis runner')
    a = fixed_batch(cfg, n, split, first_a, schedule_index)
    b = fixed_batch(cfg, n, split, first_b, schedule_index)
    schema = factor_schema(cfg, focal, a['source_present'])
    batches = {mask: splice(cfg, a, b, schema['names'], schema['coordinates'], mask) for mask in range(2**len(schema['names']))}
    if not np.array_equal(batches[0]['x'], b['x']) or not np.array_equal(batches[len(batches)-1]['x'], a['x']):
        raise AssertionError('Full/empty hybrids differ from independent A/B renders')
    return batches, schema, {'a_scene_ids': a['sample_ids'], 'b_scene_ids': b['sample_ids'],
        'all_hybrids_in_one_pair_share_row_id': True,
        'model_input_keys': ['x'], 'split': split, 'schedule_index': schedule_index}


def conditional_panels(cfg, scenes, repeats, split, first_id, schedule_index, focal, hold_masks, panel_seed=33301):
    """Independent completion draws include SPLIT in RNG, even with reused IDs.

    Same base scenes/matched completions across masks and schedules are useful
    paired controls. Different split IDs imply independent content AND noise.
    A complete measurement tensor is a single factor; schedules never redraw.
    """
    if split not in SPLITS or split == 'audit_test' or min(scenes, repeats) < 2:
        raise ValueError('Independent development scenes/repeats and known non-audit split required')
    base = fixed_batch(cfg, scenes, split, first_id, schedule_index)
    schema = factor_schema(cfg, focal, base['source_present'])
    names = schema['names']; maximum = 2**len(names)
    if len(set(hold_masks)) != len(hold_masks) or any(mask not in range(maximum) for mask in hold_masks):
        raise ValueError('Unique legal held masks required')
    outputs = {mask: [] for mask in hold_masks}
    for repetition in range(repeats):
        z = base['latents'].copy(); e = base['measurement_noise'].copy()
        for i, scene_id in enumerate(base['sample_ids']):
            rng = np.random.default_rng(np.random.SeedSequence([cfg.data_seed, SPLITS[split], int(scene_id), panel_seed, repetition]))
            z[i] = rng.normal(size=(5, 4)); e[i] = cfg.sigma*rng.normal(size=(5, 8))
        completion = dict(base, latents=z, measurement_noise=e)
        for mask in hold_masks:
            outputs[mask].append(splice(cfg, base, completion, names, schema['coordinates'], mask)['x'])
    targets = {name: np.stack([base['latents'][:, j, k] for j, k in coords], 1) for name, coords in schema['coordinates'].items() if coords}
    return {mask: np.stack(xs, 1) for mask, xs in outputs.items()}, targets, schema
