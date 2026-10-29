"""Unmodified, hash-pinned code from our L3.2 package; not author-paper code."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'parent_l32'))
from load_data import LoadConfig, SOURCES, SPLITS, generate, model_payload, _render
from linear_reference import fit_linear_encoding, fit_ridge, predict_ridge


def load_model(run_directory, device='cpu'):
    import json
    import torch
    from experiment_plan import config_from_dict
    from torch_adapter import LoadModel
    root = Path(run_directory)
    metadata = json.loads((root/'run.json').read_text())
    run = config_from_dict(metadata['config'])
    state = torch.load(root/'best.pt', map_location='cpu', weights_only=True)
    from dataclasses import asdict
    if state['config'] != asdict(run):
        raise ValueError('Checkpoint configuration differs from run metadata')
    model = LoadModel(run).to(device)
    model.load_state_dict(state['model_state'])
    return model.eval(), run, metadata


def collect(model, x, batch_size=128):
    from torch_adapter import collect_features
    return collect_features(model, x, batch_size)[0]
