"""Hash-pinned project parents, not upstream paper implementations."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'parent_l331'))
from inherited import load_model, collect, generate, _render, SOURCES
from roles import coordinate_roles, flatten_contents
from decomposition import fit_joint, apply_joint, representation_score
from linear_reference import fit_ridge, predict_ridge
sys.path.insert(0, str(ROOT/'parent_deep'))
from deep_diagnostics import (role_risk_budget, repeated_signal_and_noise,
    repeated_prediction_risk, projected_variance, gaussian_readout_reference)
