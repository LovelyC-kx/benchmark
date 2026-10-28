"""L3.2 controlled demand, content dimension, write pressure and blank age.

Model-visible input is only x. Latents, source times and masks are analysis data.
This is an independently written generator, not an upstream paper reproduction.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np

SPLITS = {"encoder_train": 0, "encoder_val": 1, "geometry_fit": 2,
          "probe_train": 3, "probe_val": 4, "audit_test": 5, "development_test": 6}
SOURCES = ("H1", "H2", "H3", "H4", "C")
MAX_T = 24

@dataclass(frozen=True)
class LoadConfig:
    k: int = 4
    required_history: int = 2
    blank_delay: int = 3
    energy_mode: str = "fixed_rms"
    sigma: float = .2
    data_seed: int = 13400
    projection_seed: int = 901
    extra_writes: int = 3
    schedule: str = "balanced"
    content_mode: str = "physical_k"

    def validate(self):
        for v in (self.k, self.required_history, self.blank_delay,
                  self.data_seed, self.projection_seed, self.extra_writes):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise ValueError("Integer configuration fields required")
        if self.k not in (1, 2, 4) or self.required_history not in (1, 2, 3, 4):
            raise ValueError("Primary K/L are 1,2,4; L=3 is an equal-demand control")
        if self.blank_delay not in (1, 3, 7, 11):
            raise ValueError("Blank delays are 1,3,7,11")
        if self.energy_mode not in ("raw", "fixed_rms"):
            raise ValueError("Energy convention")
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("Finite nonnegative sigma required")
        if min(self.data_seed, self.projection_seed) < 0:
            raise ValueError("Nonnegative seeds required")
        if self.extra_writes not in (0, 1, 3):
            raise ValueError("Extra write counts are 0,1,3")
        if self.extra_writes != 3 and self.required_history != 1:
            raise ValueError("Write-pressure panel holds L=1; no absent supervised sources")
        if self.schedule not in ("balanced", "fixed"):
            raise ValueError("Schedule is balanced or fixed")
        if self.content_mode not in ('physical_k','demand_only'):
            raise ValueError('Known physical-content/demand-only convention required')

    @property
    def input_dim(self): return 14
    @property
    def output_dim(self): return 20
    @property
    def deadline_steps(self): return 13 + self.blank_delay
    @property
    def visible_content_dim(self): return 4 if self.content_mode=='demand_only' else self.k
    @property
    def sensory_scale(self):
        return 1. if self.energy_mode == "raw" else float(
            np.sqrt((4 + 8*self.sigma**2)/(self.visible_content_dim + 8*self.sigma**2)))

    def fingerprint(self):
        self.validate()
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

def projection(cfg):
    """One shared 8x4 matrix; K changes a prefix, never the random basis."""
    return np.linalg.qr(np.random.default_rng(cfg.projection_seed).normal(size=(8, 4)))[0]

def _rng(cfg, split, sid, stream):
    return np.random.default_rng(np.random.SeedSequence(
        [cfg.data_seed, SPLITS[split], int(sid), int(stream)]))

def _render(cfg, z, noise, times, present):
    n = len(z)
    x = np.zeros((n, MAX_T, 14), dtype=np.float32)
    active = z.copy(); active[:, :, cfg.visible_content_dim:] = 0
    sensory = cfg.sensory_scale*(active @ projection(cfg).T + noise)
    for j in range(5):
        rows = np.flatnonzero(present[:, j]); t = times[rows, j]
        x[rows, t, :8] = sensory[rows, j]
        x[rows, t, 8+j] = 1
        x[rows, t, 13] = 1
    return x

def _targets(cfg, z):
    y = z.copy(); y[:, :, cfg.k:] = 0
    mask = np.zeros((5, 4), dtype=bool)
    mask[:cfg.required_history, :cfg.k] = True; mask[4, :cfg.k] = True
    return y.astype(np.float32), mask

def generate(cfg, n, split="encoder_train", sample_ids=None):
    cfg.validate()
    if split not in SPLITS or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("Known split and positive integer n required")
    ids = np.arange(n, dtype=np.int64) if sample_ids is None else np.asarray(sample_ids)
    if ids.ndim != 1 or len(ids) != n or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Integer sample IDs matching n required")
    if np.any(ids < 0) or len(np.unique(ids)) != n:
        raise ValueError("Nonnegative unique sample IDs required")
    z = np.empty((n, 5, 4)); noise = np.empty((n, 5, 8))
    times = np.empty((n, 5), dtype=int); present = np.ones((n, 5), dtype=bool)
    schedules = np.array([[5,8,11],[5,11,8],[8,5,11],[8,11,5],[11,5,8],[11,8,5]])
    for row, sample_id in enumerate(ids):
        block, slot = divmod(int(sample_id), 6)
        order = _rng(cfg, split, block, 99).permutation(6)
        times[row] = np.r_[2, schedules[order[slot]] if cfg.schedule == "balanced" else schedules[0], 12]
        if cfg.extra_writes != 3:
            present[row, 1:4] = False
            if cfg.extra_writes == 1:
                # Within a complete six-sample block each extra identity appears twice.
                chosen = _rng(cfg, split, block, 98).permutation(3)[slot % 3]
                present[row, 1+chosen] = True
        for j in range(5):
            z[row, j] = _rng(cfg, split, sample_id, j).normal(size=4)
            noise[row, j] = cfg.sigma*_rng(cfg, split, sample_id, 10+j).normal(size=8)
    y, mask = _targets(cfg, z)
    return {"x": _render(cfg, z, noise, times, present), "y_all": y,
            "target_mask": mask, "latents": z, "measurement_noise": noise,
            "event_times": times, "source_present": present,
            "sample_ids": ids.copy(),
            "group_ids": np.array([f"load02:{cfg.data_seed}:{split}:{int(i)}" for i in ids]),
            "meta": {"schema": "load_geometry_0.2", "config": asdict(cfg),
                     "config_hash": cfg.fingerprint(), "split": split,
                     "deadline_steps": cfg.deadline_steps, "model_keys": ["x"],
                     "same_inputs_across_L": True,
                     "time_semantics": "discrete observation index; no physical-time claim"}}

def model_payload(batch):
    return {"x": batch["x"][:, :batch["meta"]["deadline_steps"]]}

def make_source_pair(cfg, batch, source, pair_seed=0):
    if source not in SOURCES or not isinstance(pair_seed, int) or pair_seed < 0:
        raise ValueError("source/pair seed")
    if cfg.fingerprint() != batch["meta"]["config_hash"]:
        raise ValueError("Config mismatch")
    j = SOURCES.index(source)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    for row, sample_id in enumerate(batch["sample_ids"]):
        g = np.random.default_rng(np.random.SeedSequence(
            [cfg.data_seed, SPLITS[batch["meta"]["split"]], int(sample_id), 1000+j, pair_seed]))
        out["latents"][row, j] = g.normal(size=4)
    out["y_all"], out["target_mask"] = _targets(cfg, out["latents"])
    out["x"] = _render(cfg, out["latents"], out["measurement_noise"], out["event_times"], out["source_present"])
    out["meta"] = dict(batch["meta"], pair_changed=source, pair_seed=pair_seed,
                       pair_source_present=bool(batch["source_present"][:, j].all()))
    return out

def source_reference(cfg, batch, source):
    """Known-P Gaussian Bayes reference from the observed tagged event.

    This oracle is only a solvability reference; P is not a model input.
    """
    j = SOURCES.index(source); x = model_payload(batch)["x"]
    tag = x[:, :, 8+j]
    if not np.all(tag.sum(1) == 1):
        raise ValueError("Reference requires the source in every sample")
    t = tag.argmax(1); v = x[np.arange(len(t)), t, :8]/cfg.sensory_scale
    return (v @ projection(cfg)/(1+cfg.sigma**2))[:, :cfg.k]

def classification_labels(cfg, batch):
    """Five balanced binary tasks; fixed task directions independent of samples."""
    theta = np.random.default_rng(401).normal(size=(5, 4))[:, :cfg.k]
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    labels = (np.einsum('nsk,sk->ns', batch['latents'][:, :, :cfg.k], theta) >= 0)
    return labels.astype(np.float32)

def loss_components_numpy(pred, target, cfg, lambda_extra=1.):
    """Fixed weight on H1 and C; mean over additional requested histories."""
    if not np.isfinite(lambda_extra) or lambda_extra < 0:
        raise ValueError("Finite nonnegative auxiliary weight")
    p = np.asarray(pred).reshape(-1, 5, 4); y = np.asarray(target).reshape(-1, 5, 4)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Finite matching predictions and targets")
    sq = (p-y)**2
    anchor = float(sq[:, [0,4], :cfg.k].mean())
    extra = float(sq[:, 1:cfg.required_history, :cfg.k].mean()) if cfg.required_history > 1 else 0.
    return {"anchor_mse": anchor, "extra_mse": extra,
            "weighted_loss": anchor+lambda_extra*extra,
            "active_target_mse": float((sq[:, :cfg.required_history, :cfg.k].sum()
                                      +sq[:, 4, :cfg.k].sum())/(len(p)*(cfg.required_history+1)*cfg.k))}
