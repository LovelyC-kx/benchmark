"""L2 integration example. Requires PyTorch and the recovered L2 snnbench package.
No training loop is launched by this file. The example is not a full experiment runner.
"""
import torch
from torch import nn
from snnbench.models import BenchmarkModel, ModelConfig

class TaggedModel(nn.Module):
    def __init__(self, neuron="lif_hard", task="current"):
        super().__init__()
        if task not in ("current", "history", "joint"):
            raise ValueError(task)
        self.frame_encoder = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 16))
        cfg = ModelConfig(
            backbone="mlp", neuron=neuron, width=32, depth=2,
            beta=.8, threshold=1., gain=2., dc_normalize=False,
            gradient_mode="full", surrogate="atan", sg_scale=2.,
            detach_reset=True, observation="post", readout="last",
            normalization="none", push_pull=False, reset_strength=1.)
        self.core = BenchmarkModel(
            (16,), (8 if task=="joint" else 4,), 16, cfg, objective="regression")

    def encode(self, x):
        if x.ndim != 3 or x.shape[-1] != 12:
            raise ValueError("Expected B,T,12 legal observations")
        return self.frame_encoder(x)*x[..., -1:]

    def forward(self, x, capture=False):
        return self.core(self.encode(x), capture=capture)

    def traces(self):
        return self.core.traces()

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

def configure_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=1e-4)

