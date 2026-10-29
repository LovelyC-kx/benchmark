"""Versioned paired four-band toy images: coverage and source freshness are separate axes."""
import hashlib
import numpy as np

VERSION='spatial_history45_v1'
TIMES=np.array([0.,1.,3.,6.,10.,15.])
REGIONS=('C','H','U_vis','changed_C')

def region_masks(size=16):
    if size<8 or size%2:raise ValueError('Even spatial size >=8')
    h=size//2;c=np.zeros((size,size),bool);c[:h,:h]=1;c[h:,:h]=1
    hist=np.zeros_like(c);hist[:h,h:]=1
    unseen=np.zeros_like(c);unseen[h:,h:]=1
    changed=np.zeros_like(c);changed[h:,:h]=1
    return dict(C=c,H=hist,U_vis=unseen,changed_C=changed)

def basis(size=8,family='id'):
    yy,xx=np.mgrid[:size,:size];x=(xx+.5)/size;y=(yy+.5)/size
    k=1 if family=='id' else 2 if family=='texture_ood' else None
    if k is None:raise ValueError('Known texture family')
    return np.stack([np.ones_like(x),np.cos(k*np.pi*x),np.cos(k*np.pi*y),(-1.)**(xx+yy)])

def render(z,size=16,family='id'):
    z=np.asarray(z,float)
    if z.ndim!=3 or z.shape[1:]!=(4,4):raise ValueError('N x four regions x four coefficients')
    h=size//2;b=basis(h,family)
    # Shared factors, distinct known spectral responses; range remains in [0,1].
    spectral=np.array([[1,.7,.8,.9],[.9,1,.6,.7],[.7,.8,1,.6],[.8,.6,.9,1.]])
    tiles=.5+.07*np.einsum('nqk,ck,khw->nqchw',z,spectral,b)
    out=np.zeros((len(z),4,size,size));out[:,:,:h,:h]=tiles[:,0];out[:,:,:h,h:]=tiles[:,1]
    out[:,:,h:,:h]=tiles[:,2];out[:,:,h:,h:]=tiles[:,3]
    return out.astype('float32')

def batch(n,first_id,*,family='id',noise=.015,motion=False):
    if not isinstance(n,int) or n<1 or noise<0:raise ValueError('Positive count and nonnegative noise')
    ids=np.arange(first_id,first_id+n,dtype=np.int64)
    rows=[];old=[];eps=[]
    for sid in ids:
        g=np.random.default_rng(int(sid));rows.append(g.uniform(-1,1,(4,4)));old.append(g.uniform(-1,1,(2,4)))
        eps.append(g.normal(0,noise,(6,4,16,16)))
    z=np.array(rows);stale=np.array(old);eps=np.array(eps);target=render(z,family=family)
    masks=region_masks();vis=np.zeros((6,1,16,16),bool)
    # Old observations at t0; latest H at t2; new C and changed-C at t5.
    vis[0,0]=masks['H']|masks['changed_C'];vis[2,0]=vis[0,0];vis[5,0]=masks['C']
    zold=z.copy();zold[:,1]=stale[:,0];zold[:,2]=stale[:,1]
    zhist=z.copy();zhist[:,2]=stale[:,1]
    scenes=np.repeat(target[:,None],6,axis=1);scenes[:,0]=render(zold,family=family);scenes[:,2]=render(zhist,family=family)
    visible=np.repeat(vis[None],n,axis=0);obs=np.where(visible,np.clip(scenes+eps,0,1),0.)
    shifts=np.array([(0,1),(0,0),(1,0),(0,0),(0,0),(0,0)] if motion else [(0,0)]*6)
    for t,(dy,dx) in enumerate(shifts):
        obs[:,t]=np.roll(obs[:,t],(dy,dx),axis=(-2,-1));visible[:,t]=np.roll(visible[:,t],(dy,dx),axis=(-2,-1))
    x=np.concatenate([obs,visible.astype('float32')],axis=2).astype('float32')
    return {'x':x,'y':target,'latents':z,'stale':stale,'scene_ids':ids,'masks':masks,'visible':visible,
        'noise':eps,'family':family,'shifts':shifts,'timestamps':TIMES.copy(),
        'contract':{'generator':VERSION,'model_fields':['x'],'target_time':5,'deadline':15.,'one_frame_one_update':True,
        'timestamps_are_analysis_only':True,'U_vis':'Current target never observed','U_src':'Past changed-C and old H: different content, not a coverage region',
        'motion':'known periodic integer shifts; analysis alignment oracle' if motion else 'aligned',
        'noise_std':noise,'synthetic_four_bands_are_not_Sentinel2_measurements':True}}

def replace_source(a,b,source='H',neutral=False):
    """Keep recipient noise/masks/current frame; rebuild only the named past source."""
    if a['family']!=b['family'] or len(a['x'])!=len(b['x']):raise ValueError('Matched family and count')
    out={k:v for k,v in a.items()};out['x']=a['x'].copy();z=a['latents'].copy();stale=a['stale'].copy()
    if source=='H':z[:,1]=0. if neutral else b['latents'][:,1];t=2;region='H'
    elif source=='U_src':stale[:,1]=0. if neutral else b['stale'][:,1];t=None;region='changed_C'
    elif source=='C':z[:,0]=0. if neutral else b['latents'][:,0];t=5;region='C'
    else:raise ValueError('Declared source')
    oldz=z.copy();oldz[:,1]=stale[:,0];oldz[:,2]=stale[:,1]
    histz=z.copy();histz[:,2]=stale[:,1]
    frame_map={0:render(oldz,family=a['family']),2:render(histz,family=a['family']),5:render(z,family=a['family'])}
    ts=(0,2) if t is None else (t,)
    mask=a['masks'][region]
    if source=='C':mask=mask&~a['masks']['changed_C']
    for tt in ts:
        dy,dx=a['shifts'][tt];r=np.roll(mask,(dy,dx),axis=(-2,-1))
        obs=np.roll(np.clip(frame_map[tt]+a['noise'][:,tt],0,1),(dy,dx),axis=(-2,-1))
        valid=a['visible'][:,tt,0]&r
        out['x'][:,tt,:4]=np.where(valid[:,None],obs,out['x'][:,tt,:4])
    out['source_operation']=source;out['original_recipient_target_retained']=True
    return out

def align_observations(a):
    x=a['x'].copy()
    for t,(dy,dx) in enumerate(a['shifts']):x[:,t]=np.roll(x[:,t],(-dy,-dx),axis=(-2,-1))
    return x

def latest_visible(x):
    x=np.asarray(x);pred=np.full((len(x),4,16,16),.5,dtype='float32')
    for t in range(x.shape[1]):pred=np.where(x[:,t,4:5]>0,x[:,t,:4],pred)
    return pred

def labels(a):return np.concatenate([a['latents'].reshape(len(a['x']),-1),a['stale'].reshape(len(a['x']),-1)],axis=1)
def input_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
