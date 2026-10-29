"""L1 sidecar: observation permissions and target semantics, no neuron changes.

The legacy generators remain byte-identical. Only x is passed to a model, as
in the original runner. Image visibility is already the second x channel.
"""
from dataclasses import asdict
from copy import deepcopy
import numpy as np
from .tasks import TaskConfig, generate, VECTOR_TASKS, IMAGE_TASKS
from .physical import physical_sequences


_SPECS = {
    'repeat': ('sign(z[0]) / full z', 'repeated noisy evidence; order not structurally required', 'q absent from input', 'measurement_control', 'compare independent versus identical measurements'),
    'stale': ('sign(new z[0]) / full new z', 'fixed-boundary replacement; clean terminal observation sufficient', 'q is the old target', 'update_control', 'do not claim selective multi-item update'),
    'complement': ('sign(z[0]) / full z', 'terminal hides coordinate 0; earlier complementary coordinates useful; unordered sufficient', 'q absent from input', 'complementary_history', 'retain value-coordinate binding; no order claim'),
    'order': ('first cue identity / full cue trajectory', 'two fixed-position cues; first cue identifies class; unordered contents ambiguous', 'q absent from input', 'minimal_order_control', 'not general queried relation reasoning'),
    'interval': ('short versus long / full cue trajectory', 'fixed first cue; second absolute position identifies class', 'q absent from input', 'absolute_position_control', 'not a position-independent physical interval assay'),
    'delay': ('sign(z[0]) / full z', 'delay>0: terminal blanks require earlier evidence; delay=0: terminal visible', 'q absent from input', 'delayed_retention', 'report actual trailing blanks; order not required by target'),
    'xor': ('a XOR b / ordered pair [a,b]', 'two bits on distinct coordinates; unordered aggregation sufficient', 'q absent from input', 'nonlinear_combination', 'single-bit chance does not imply order necessity'),
    'crosslag': ('leading channel identity / full cue trajectory', 'first active channel identifies class', 'q absent from input', 'minimal_lead_control', 'not general cross-channel lag estimation'),
    'drift': ('sign(endpoint z[0]) / endpoint z', 'fixed linear interpolation from q to z; clean terminal observation sufficient', 'q is the starting target', 'endpoint_update_control', 'not arbitrary nonstationary tracking'),
    'binary_adding': ('a+b in {0,1,2} / ordered pair [a,b]', 'two bits on distinct coordinates; unordered aggregation sufficient', 'q absent from input', 'toy_accumulation', 'not official nine-marker balanced Binary Adding'),
    'image_repeat': ('orientation / full clean image', 'classification may use a single frame; reconstruction aggregates repeated noisy views', 'q absent from input', 'measurement_control', 'clipping and repeated noise must be recorded'),
    'image_complement': ('orientation / full clean image', 'classification can use terminal half-image; full detail reconstruction needs complementary history', 'q absent from input', 'classification_reconstruction_control', 'separate targets and assess visible/history-visible/unseen regions'),
    'image_stale': ('new orientation / new clean image', 'fixed-boundary full-scene replacement; clean terminal view sufficient', 'q is the old scene', 'update_control', 'old-scene pixels are not observations of the current target'),
    'image_misaligned': ('orientation / reference clean image', 'historical frames use deterministic wraparound shifts; terminal frame aligned', 'q absent from input', 'alignment_control', 'not natural motion or real occlusion; index-dependent inverse alignment uses order'),
    'image_rain': ('orientation / full clean image', 'repeated scene plus noise and independently resampled vertical streaks', 'q absent from input', 'synthetic_degradation_control', 'repeated_noise repeats Gaussian noise, not rain draws'),
    'image_cloud': ('orientation / full clean image', 'alternating half visibility plus random block occlusion; detail requirements depend on visibility union', 'q absent from input', 'partial_observation', 'synthetic; unseen pixels need priors rather than historical retrieval'),
}


def task_contract(cfg: TaskConfig):
    cfg.validate()
    target, requirement, nuisance, role, limitation = _SPECS[cfg.name]
    is_image = cfg.name in IMAGE_TASKS
    paired_stale = cfg.name in ('stale', 'image_stale')
    return {
        'schema_version': 'l1.0', 'generator_version': 'internal_v1_archive_version_1',
        'task': cfg.name, 'config': asdict(cfg), 'target_rule': target,
        'active_target_rule': target.split(' / ')[0 if cfg.objective == 'classification' else 1],
        'objective_semantics': ('classification' if cfg.objective == 'classification' else
                                'image_reconstruction' if is_image else 'vector_regression'),
        'history_requirement': requirement, 'q_semantics': nuisance,
        'role': role, 'claim_limit': limitation,
        'time_semantics': 'discrete_observation_index; not physical seconds',
        'prediction_deadline': cfg.T-1,
        'observation_unit': 'one image plus its visibility channel' if is_image else 'one full projected input vector',
        'model_keys': ['x'],
        'embedded_model_information': ['image visibility channel'] if is_image else [],
        'analysis_only_keys': ['target', 'distractor', 'mask', 'timestamps', 'meta', 'x_pair', 'classes'],
        'supervision_only_keys': ['y'],
        'availability': 'all x steps available by terminal deadline; timestamps are not passed by the legacy runner',
        'source_split': 'whole independent sequence; shared projection seed; SeedSequence(data_seed, split offset)',
        'learning_curve': 'generate fixed maximum-N pool per split, then take subsets; do not regenerate at each N',
        'pair': {'kind': 'label_preserving_old_history_replacement' if paired_stale else 'input_perturbation_without_relabeling',
                 'use_same_y_for_supervised_pair': paired_stale,
                 'preserved_observation': f'last {cfg.new_steps} full measurements' if paired_stale else 'last full measurement only',
                 'restriction': 'do not train/evaluate x_pair with original y as a valid new sample unless label preservation is certified'},
        'transforms': {
            'support_note': 'Label-invariant observation controls may leave original generator support; register transformed domain separately',
            'item_permutation': ('preserve label; item includes all channels' if cfg.name in ('repeat','complement','xor','binary_adding','image_repeat','image_complement','image_rain','image_cloud') else
                                 'not registered as label preserving; role/position/endpoint semantics can change'),
            'sequence_reverse': 'recompute target under explicit valid generator rule; otherwise perturbation only',
        },
    }


def model_payload(sample):
    """Whitelist rather than filtering: metadata/latent truths never pass through."""
    return {'x': sample['x']}


def generate_with_contract(cfg, n, split='train', paired=False):
    sample = generate(cfg, n, split, paired)
    contract = task_contract(cfg)
    sample['meta'] = deepcopy(sample['meta'])
    sample['meta']['l1_contract'] = contract
    if paired:
        sample['meta']['paired_labels_preserved'] = contract['pair']['use_same_y_for_supervised_pair']
    return sample


def supervised_pair(sample):
    if 'x_pair' not in sample:
        raise ValueError('No paired input exists')
    if sample.get('meta', {}).get('paired_labels_preserved') is not True:
        raise ValueError('Pair is not certified label preserving; recompute targets under a valid rule')
    return {'x': sample['x_pair'], 'y': sample['y']}


def reconstruction_regions(sample, cfg):
    """Partition target coordinates using target-valid, aligned visibility.

    Only the six original image tasks are supported. For image_stale, old frames
    depict a different target and cannot count toward current-target coverage.
    Original misaligned masks are all ones; extending to sparse moving masks
    requires alignment before this function can be generalized.
    """
    if cfg.name not in IMAGE_TASKS:
        raise ValueError('Image task required')
    masks = np.asarray(sample['mask']) > 0
    if masks.ndim != 5 or masks.shape[1] != cfg.T:
        raise ValueError('Expected [N,T,1,H,W] masks')
    eligible = masks[:, -cfg.new_steps:] if cfg.name == 'image_stale' else masks
    current = masks[:, -1]
    union = eligible.any(axis=1)
    history_only = ~current & union
    unseen = ~union
    return {'current_visible': current, 'history_only_visible': history_only,
            'never_visible_for_target': unseen}


def regional_mse(pred, target, regions):
    if np.shape(pred) != np.shape(target):
        raise ValueError('Prediction/target shapes differ')
    sq = (np.asarray(pred, dtype=float) - target)**2
    result = {}
    for name, mask in regions.items():
        if mask.shape != sq.shape:
            raise ValueError('Region shape differs from target')
        count = int(mask.sum())
        result[name] = {'count': count, 'mse': float(sq[mask].mean()) if count else None}
    return result


def physical_with_availability(**kwargs):
    """Annotate unchanged legacy values with causal availability of interpolation.

    Interpolated values can use a right endpoint later than their event timestamp.
    This wrapper supports terminal offline recovery. It does not silently turn
    the interpolation into a causal online sampler.
    """
    sample = physical_sequences(**kwargs)
    T = kwargs.get('T', 16)
    duration = kwargs.get('duration', 1.)
    coarse = kwargs.get('interpolate_from')
    stamps = np.linspace(duration/T, duration, T)
    if coarse is None:
        indices = np.rint(stamps/duration*1024).astype(int)
        available = indices / 1024 * duration
    else:
        base = np.linspace(duration/coarse, duration, coarse)
        indices = np.searchsorted(base, stamps, side='left').clip(0, coarse-1)
        # Availability follows actual coarse acquisition grid, not idealized stamps.
        available = np.rint(base[indices]/duration*1024) / 1024 * duration
    sample['availability_time'] = available
    sample['meta'] = deepcopy(sample['meta'])
    sample['meta'].update({
        'prediction_deadline_seconds': duration,
        'no_future_observation': bool(np.all(available <= duration)),
        'no_future_relative_to_each_timestamp': bool(np.all(available <= stamps + 1e-10)),
        'availability_semantics': 'fine-grid acquisition time of latest source measurement required for this value',
        'legacy_values_unchanged': True,
    })
    return sample


def assert_available(sample, prediction_deadline):
    availability = np.asarray(sample['availability_time'])
    if not np.isfinite(availability).all() or not np.isfinite(prediction_deadline):
        raise ValueError('Availability and prediction deadline must be finite')
    if np.any(availability > prediction_deadline + 1e-10):
        raise ValueError('Observation depends on a measurement acquired after prediction deadline')


def registry():
    return [task_contract(TaskConfig(name=name, T=12, new_steps=3, delay=2))
            for name in VECTOR_TASKS + IMAGE_TASKS]
