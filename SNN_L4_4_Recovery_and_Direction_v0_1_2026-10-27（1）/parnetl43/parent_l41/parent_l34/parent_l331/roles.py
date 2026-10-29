"""Task-relative coordinate roles. Nonfocal never automatically means harmful."""
import numpy as np
from inherited import SOURCES


def coordinate_roles(cfg, focal='C', source_present=None):
    cfg.validate()
    requested = set(SOURCES[:cfg.required_history]) | {'C'}
    if focal not in requested:
        raise ValueError('Primary focal content must be requested by the task')
    if source_present is not None and not np.asarray(source_present).all():
        raise ValueError('This initial protocol requires all sources present; stratify presence before extending')
    rows = []
    for source_id, source in enumerate(SOURCES):
        for coordinate in range(cfg.visible_content_dim):
            asked = source in requested and coordinate < cfg.k
            role = 'focal' if asked and source == focal else ('cotarget' if asked else 'unrequested')
            rows.append({'source': source, 'source_id': source_id, 'coordinate': coordinate,
                         'role': role, 'requested': asked})
    return rows


def flatten_contents(batch, cfg):
    return batch['latents'][:, :, :cfg.visible_content_dim].reshape(len(batch['latents']), -1)
