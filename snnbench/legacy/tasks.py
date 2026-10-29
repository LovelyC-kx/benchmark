"""Controlled generators. Every sequence is an independent statistical unit."""
from dataclasses import dataclass, asdict
import numpy as np

VECTOR_TASKS = ('repeat', 'stale', 'complement', 'order', 'interval', 'delay',
                'xor', 'crosslag', 'drift', 'binary_adding')
IMAGE_TASKS = ('image_repeat', 'image_complement', 'image_stale',
               'image_misaligned', 'image_rain', 'image_cloud')

@dataclass
class TaskConfig:
    name: str = 'repeat'
    objective: str = 'classification'
    T: int = 16
    latent_dim: int = 8
    input_dim: int = 16
    sigma: float = .25
    new_steps: int = 4
    delay: int = 3
    interval: int = 3
    image_size: int = 16
    repeated_noise: bool = False
    projection_seed: int = 901
    data_seed: int = 100
    shift: int = 2

    def validate(self):
        if self.name not in VECTOR_TASKS + IMAGE_TASKS: raise ValueError(self.name)
        if self.objective not in ('classification', 'regression', 'reconstruction'):
            raise ValueError(self.objective)
        if self.T < 4 or self.latent_dim < 2 or self.input_dim < self.latent_dim:
            raise ValueError('T>=4, latent_dim>=2, input_dim>=latent_dim required')
        if not 1 <= self.new_steps < self.T: raise ValueError('1<=new_steps<T')
        if self.sigma < 0 or self.delay < 0: raise ValueError('negative noise/delay')
        if self.name in ('order', 'interval', 'crosslag', 'delay'):
            if self.delay >= self.T-2: raise ValueError('delay leaves too little evidence')
        if self.name == 'order' and self.T-self.delay-2 <= 1:
            raise ValueError('order requires distinct cue times: T-delay-2 must exceed 1')
        if self.name == 'interval' and self.interval < 2: raise ValueError('interval>=2')
        if self.name == 'interval' and 1+self.interval+self.delay >= self.T:
            raise ValueError('interval/blank duration exceeds window')

def _projection(cfg):
    rng = np.random.default_rng(cfg.projection_seed)
    W = np.linalg.qr(rng.normal(size=(cfg.input_dim, cfg.latent_dim)))[0]
    return W.astype('float32')

def generate(cfg: TaskConfig, n: int, split='train', paired=False):
    cfg.validate()
    offset = {'train': 0, 'val': 1, 'test': 2, 'heldout': 3}.get(split)
    if offset is None: raise ValueError('unknown split')
    rng = np.random.default_rng(np.random.SeedSequence([cfg.data_seed, offset]))
    if cfg.name.startswith('image_'): return _images(cfg, n, rng, paired)
    W = _projection(cfg)
    z = rng.normal(size=(n, cfg.latent_dim)).astype('float32')
    q = rng.normal(size=z.shape).astype('float32')
    hidden = np.repeat(z[:, None], cfg.T, axis=1)
    visible = np.ones_like(hidden)
    labels = (z[:, 0] >= 0).astype('int64')
    meta = {'time_axis': 'discrete_observation', 'task': asdict(cfg),
            'label_semantics': 'sign(current target coordinate 0)',
            'oracle_boundary_available_to_model': False}
    if cfg.name == 'stale': hidden[:, :-cfg.new_steps] = q[:, None]
    elif cfg.name == 'drift':
        weights = np.linspace(0, 1, cfg.T)[None, :, None]
        hidden = (1-weights)*q[:, None]+weights*z[:, None]
    elif cfg.name == 'complement':
        visible[:] = 0
        for t in range(cfg.T): visible[:, t, t%2::2] = 1
        visible[:, -1] = 0; visible[:, -1, 1::2] = 1
        # terminal lacks one half; past is essential for a full-vector target.
    elif cfg.name == 'delay':
        hidden[:, cfg.T-cfg.delay:] = 0
        visible[:, cfg.T-cfg.delay:] = 0
    elif cfg.name in ('order', 'interval', 'crosslag'):
        hidden[:] = 0; visible[:] = 1
        labels = rng.integers(0, 2, n)
        end = cfg.T-cfg.delay-2
        if cfg.name == 'order':
            for i in range(n):
                hidden[i, 1, labels[i]] = 1
                hidden[i, end, 1-labels[i]] = 1
            meta['label_semantics'] = 'which cue appeared first; reverse changes label'
        elif cfg.name == 'interval':
            for i in range(n):
                hidden[i, 1, 0] = 1
                hidden[i, 1+(1 if labels[i]==0 else cfg.interval), 1] = 1
            meta['label_semantics'] = 'short versus long interval; same cue bag'
        else:
            for i in range(n):
                t0=max(1, end-2)
                hidden[i, t0, labels[i]] = 1
                hidden[i, t0+1, 1-labels[i]] = 1
            meta['label_semantics'] = 'cross-channel lead/lag'
        z = hidden.reshape(n, -1).copy()
    elif cfg.name in ('xor', 'binary_adding'):
        a = rng.integers(0, 2, n); b = rng.integers(0, 2, n)
        hidden[:] = 0; hidden[:, 1, 0] = a; hidden[:, -2, 1] = b
        z = np.stack([a, b], 1).astype('float32')
        labels = (a ^ b) if cfg.name == 'xor' else a+b
        meta['label_semantics'] = ('XOR of two temporally separated bits' if cfg.name=='xor'
                                   else 'sum of two separated binary inputs (three classes); not original Binary Adding reproduction')
    noise = rng.normal(scale=cfg.sigma, size=(n, cfg.T, cfg.input_dim)).astype('float32')
    if cfg.repeated_noise: noise[:] = noise[:, :1]
    x = (hidden*visible) @ W.T + noise
    # Delay and relation tasks have exact common blanks; blank input is not noisy evidence.
    if cfg.name == 'delay' and cfg.delay: x[:, -cfg.delay:] = 0
    if cfg.name in ('order', 'interval', 'crosslag'):
        active = np.any(hidden != 0, axis=-1)
        x[~active] = 0
    if cfg.objective == 'classification': y = labels
    else: y = z
    out = dict(x=x.astype('float32'), y=y, target=z, distractor=q,
               mask=visible, timestamps=np.arange(cfg.T, dtype='float32'), meta=meta,
               classes=3 if cfg.name=='binary_adding' else 2)
    if paired:
        xp=x.copy()
        if cfg.name=='stale':
            q2=rng.normal(size=q.shape).astype('float32')
            xp[:, :-cfg.new_steps]=q2[:, None]@W.T+noise[:, :-cfg.new_steps]
        elif cfg.name=='complement':
            z2=rng.normal(size=hidden.shape).astype('float32')
            xp[:, :-1]=(z2[:, :-1]*visible[:, :-1])@W.T+noise[:, :-1]
            out['meta']['paired_labels_preserved']=False
        else:
            xp[:, :-1]=x[rng.permutation(n), :-1]
            out['meta']['paired_labels_preserved']=False
        out['x_pair']=xp
    return out

def _images(cfg, n, rng, paired):
    H=cfg.image_size
    if H<8: raise ValueError('image_size>=8')
    def scenes():
        a=np.zeros((n,1,H,H), 'float32')
        labels=rng.integers(0,2,n)
        for i in range(n):
            a[i]=rng.uniform(0,.1,(1,H,H))
            if labels[i]==0: a[i,0,H//4:3*H//4,H//3:2*H//3]=rng.uniform(.6,1)
            else: a[i,0,H//3:2*H//3,H//4:3*H//4]=rng.uniform(.6,1)
            # independent continuous details make exact reconstruction nontrivial
            a[i,0,:H//4,:H//4]=rng.uniform(.1,.8,(H//4,H//4))
        return a, labels
    z,labels=scenes(); q,_=scenes()
    scene=np.repeat(z[:,None],cfg.T,axis=1)
    if cfg.name=='image_stale': scene[:,:-cfg.new_steps]=q[:,None]
    mask=np.ones_like(scene)
    if cfg.name in ('image_complement','image_cloud'):
        for t in range(cfg.T):
            mask[:,t,:,:, (0 if t%2==0 else H//2):(H//2 if t%2==0 else H)] = 0
        if cfg.name=='image_cloud':
            for i in range(n):
                for t in range(cfg.T):
                    r,c=rng.integers(0,H//2,2); mask[i,t,0,r:r+H//3,c:c+H//3]=0
    noise=rng.normal(0,cfg.sigma,scene.shape).astype('float32')
    if cfg.repeated_noise: noise[:]=noise[:,:1]
    obs=scene+noise
    if cfg.name=='image_misaligned':
        for t in range(cfg.T-1):
            obs[:,t]=np.roll(obs[:,t], ((t%3-1)*cfg.shift, (t%2)*cfg.shift), axis=(-2,-1))
    if cfg.name=='image_rain':
        for i in range(n):
            for t in range(cfg.T):
                for c in rng.integers(0,H, max(1,H//6)): obs[i,t,0,:,c]+=.4
    obs=np.clip(obs,0,1)*mask
    # Visibility is supplied to ALL architectures; reconstruction target has one channel.
    x=np.concatenate([obs,mask],axis=2)
    y=labels if cfg.objective=='classification' else z
    out=dict(x=x.astype('float32'),y=y,target=z,distractor=q,mask=mask,
             timestamps=np.arange(cfg.T,dtype='float32'),classes=2,
             meta={'task':asdict(cfg),'time_axis':'discrete_observation',
                   'synthetic_not_real_remote_sensing':True,'mask_fairly_shared':True,
                   'misalignment_boundary':'wraparound synthetic translation'})
    if paired:
        xp=x.copy()
        if cfg.name=='image_stale':
            q2,_=scenes(); xp[:,:-cfg.new_steps,0:1]=np.clip(q2[:,None]+noise[:,:-cfg.new_steps],0,1)*mask[:,:-cfg.new_steps]
        else: xp[:,:-1]=x[rng.permutation(n),:-1]; out['meta']['paired_labels_preserved']=False
        out['x_pair']=xp
    return out
