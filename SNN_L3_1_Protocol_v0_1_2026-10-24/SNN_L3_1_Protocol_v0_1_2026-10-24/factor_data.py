"""Tagged content reference, NumPy only; this module never trains an SNN."""
from dataclasses import dataclass, asdict
from itertools import permutations
import hashlib
import json
import numpy as np

SOURCES = ("A", "B", "C")
TASKS = ("current", "history", "joint")
SPLITS = {"encoder_train": 0, "encoder_val": 1, "geometry_fit": 2,
          "probe_train": 3, "probe_val": 4, "audit_test": 5}

@dataclass(frozen=True)
class FactorConfig:
    latent_dim: int = 4
    sensory_dim: int = 8
    T: int = 16
    history_slots: tuple = (2, 5, 8)
    current_time: int = 12
    sigma: float = 0.2
    projection_seed: int = 901
    data_seed: int = 12400

    def validate(self):
        if self.latent_dim < 1 or self.sensory_dim < self.latent_dim:
            raise ValueError("sensory_dim must be >= latent_dim >= 1")
        if self.T < 4 or not 0 <= self.current_time < self.T - 1:
            raise ValueError("Require current evidence followed by blank(s)")
        if len(set(self.history_slots)) != len(self.history_slots) or len(self.history_slots) < 2:
            raise ValueError("At least two unique history slots")
        if any(not isinstance(t, int) or not 0 <= t < self.current_time
               for t in self.history_slots):
            raise ValueError("History slots must precede C")
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("Nonnegative finite noise")
        if min(self.projection_seed, self.data_seed) < 0:
            raise ValueError("Nonnegative seeds")

    @property
    def input_dim(self):
        return self.sensory_dim + 4

def config_hash(cfg):
    cfg.validate()
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()

def projection(cfg):
    cfg.validate()
    g = np.random.default_rng(cfg.projection_seed)
    return np.linalg.qr(g.normal(size=(cfg.sensory_dim, cfg.latent_dim)))[0]

def _rng(cfg, split, sample_id, stream):
    return np.random.default_rng(np.random.SeedSequence(
        [cfg.data_seed, SPLITS[split], int(sample_id), int(stream)]))

def _render(cfg, z, noise, times):
    n = len(z)
    x = np.zeros((n, cfg.T, cfg.input_dim), dtype=np.float32)
    P = projection(cfg)
    for j in range(3):
        t = times[:, j]
        x[np.arange(n), t, :cfg.sensory_dim] = z[:, j] @ P.T + noise[:, j]
        x[np.arange(n), t, cfg.sensory_dim+j] = 1
        x[np.arange(n), t, -1] = 1
    return x

def generate(cfg, n, split="encoder_train", sample_ids=None):
    """Counter-based samples: increasing n or generating chunks preserves values."""
    cfg.validate()
    if split not in SPLITS:
        raise ValueError("Unknown split")
    ids = np.arange(n, dtype=np.int64) if sample_ids is None else np.asarray(sample_ids)
    if ids.ndim != 1 or len(ids) != n or n < 1:
        raise ValueError("Positive n matching one-dimensional IDs")
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("Nonnegative integer IDs")
    if len(np.unique(ids)) != n:
        raise ValueError("Repeated IDs")
    schedules = list(permutations(cfg.history_slots, 2))
    K = len(schedules)
    z = np.empty((n, 3, cfg.latent_dim))
    noise = np.empty((n, 3, cfg.sensory_dim))
    times = np.empty((n, 3), dtype=np.int64)
    for row, sid in enumerate(ids):
        block, slot = divmod(int(sid), K)
        order = _rng(cfg, split, block, 99).permutation(K)
        a_time, b_time = schedules[order[slot]]
        times[row] = (a_time, b_time, cfg.current_time)
        for j in range(3):
            z[row, j] = _rng(cfg, split, sid, j).normal(size=cfg.latent_dim)
            noise[row, j] = cfg.sigma*_rng(cfg, split, sid, 10+j).normal(
                size=cfg.sensory_dim)
    return {
        "x": _render(cfg, z, noise, times),
        "latents": z, "noise": noise, "event_times": times, "sample_ids": ids.copy(),
        "group_ids": np.array([f"tf01:{cfg.data_seed}:{split}:{int(i)}" for i in ids]),
        "meta": {
            "schema": "tagged_factors_0.1", "config_hash": config_hash(cfg),
            "config": asdict(cfg), "split": split, "model_keys": ["x"],
            "time_semantics": "zero-based observation index",
            "analysis_only": ["latents", "noise", "event_times", "sample_ids", "group_ids"],
            "claim_scope": "retention and identity binding; no order necessity",
        },
    }

def make_pair(cfg, batch, changed, pair_seed=0):
    """Independent redraw of one source; other causes, noise and tags are shared."""
    if changed not in SOURCES or pair_seed < 0:
        raise ValueError("Invalid source or pair seed")
    if batch["meta"]["config_hash"] != config_hash(cfg):
        raise ValueError("Config mismatch")
    j = SOURCES.index(changed)
    z = batch["latents"].copy()
    for row, sid in enumerate(batch["sample_ids"]):
        g = np.random.default_rng(np.random.SeedSequence([
            cfg.data_seed, SPLITS[batch["meta"]["split"]],
            int(sid), 1000+j, int(pair_seed)]))
        z[row, j] = g.normal(size=cfg.latent_dim)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    out["latents"] = z
    out["meta"] = dict(batch["meta"], pair_changed=changed, pair_seed=int(pair_seed))
    out["x"] = _render(cfg, z, out["noise"], out["event_times"])
    return out

def regression_target(batch, task, history_source="A"):
    if task not in TASKS or history_source not in ("A", "B"):
        raise ValueError("Unknown task or history source")
    h = SOURCES.index(history_source)
    z = batch["latents"]
    if task == "current":
        return z[:, 2].astype(np.float32)
    if task == "history":
        return z[:, h].astype(np.float32)
    return np.concatenate((z[:, h], z[:, 2]), axis=1).astype(np.float32)

def make_noise_pair(cfg, batch, pair_seed=0):
    """Hold all latents and times; independently redraw event measurement noise."""
    if pair_seed < 0 or batch["meta"]["config_hash"] != config_hash(cfg):
        raise ValueError("Pair seed or config mismatch")
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    noise = out["noise"]
    for row, sid in enumerate(batch["sample_ids"]):
        for j in range(3):
            g = np.random.default_rng(np.random.SeedSequence([
                cfg.data_seed, SPLITS[batch["meta"]["split"]], int(sid),
                2000+j, int(pair_seed)]))
            noise[row, j] = cfg.sigma*g.normal(size=cfg.sensory_dim)
    out["meta"] = dict(batch["meta"], pair_changed="measurement_noise", pair_seed=int(pair_seed))
    out["x"] = _render(cfg, out["latents"], noise, out["event_times"])
    return out

def classification_target(cfg, batch, task, history_source="A", direction_seed=902):
    y = regression_target(batch, task, history_source)
    g = np.random.default_rng(np.random.SeedSequence([direction_seed, TASKS.index(task)]))
    w = g.normal(size=y.shape[1])
    w /= np.linalg.norm(w)
    return (y @ w >= 0).astype(np.int64)

def same_target_certified(task, changed, history_source="A"):
    if task not in TASKS or changed not in SOURCES or history_source not in ("A", "B"):
        raise ValueError("Unknown task, source or history source")
    required = {"current": {"C"}, "history": {history_source},
                "joint": {history_source, "C"}}[task]
    return changed not in required

def model_payload(batch):
    return {"x": batch["x"]}

def observed_sequence_reference(cfg, batch, source):
    """Known-P reference from observed x/tags. Does not read latent truth."""
    j = SOURCES.index(source)
    tag = batch["x"][:, :, cfg.sensory_dim+j]
    if not np.all(tag.sum(1) == 1):
        raise ValueError("Exactly one observed event per source required")
    t = tag.argmax(1)
    noisy_z = batch["x"][np.arange(len(t)), t, :cfg.sensory_dim] @ projection(cfg)
    return noisy_z/(1+cfg.sigma**2)

