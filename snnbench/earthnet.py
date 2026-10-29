"""Causal current-target protocol from U-TILISE-converted EarthNet2021 HDF5.

Source schema pinned to prs-eth/U-TILISE commit
1ea50fe1e4f1840a14b3f23062334077fc5f943b, toolbox converter and dataset loader.
We reuse the schema and author `idx_good_frames`, not their seq2seq permission.
Artificial rectangular gaps are our controlled corruption, not real clouds.
"""
import datetime as dt
from pathlib import Path
import numpy as np
from PIL import Image
from .prepare import write_records
from .utils import seeded,file_hash


def prepare_earthnet(train_file,test_file,out,T=6,size=32,group_unit='tile',seed=20261029):
    import h5py
    if T<2 or size<8 or group_unit not in ('tile','cube'):raise ValueError('T>=2, size>=8, explicit tile/cube grouping')
    files={'train':Path(train_file),'test':Path(test_file)};handles={k:h5py.File(v,'r') for k,v in files.items()};records=[];skipped=0
    try:
        for split,h in handles.items():
            paths=[]
            def visit(name,obj):
                if isinstance(obj,h5py.Group) and 'highresdynamic' in obj and 'idx_good_frames' in obj:paths.append(name)
            h.visititems(visit)
            if not paths:raise ValueError('Author-converted highresdynamic + idx_good_frames schema required')
            for path in sorted(paths):
                group=h[path];good=np.asarray(group['idx_good_frames'],int)
                if len(good)<T:skipped+=1;continue
                parts=path.split('/');tile=parts[-2];cube=parts[-1]
                # Same tile on both sides fails closed under default grouping.
                # Use OOD test for geographic transfer; IID overlap needs a separate audited cube protocol.
                group_id=tile if group_unit=='tile' else tile+'/'+cube
                records.append({'official_split':split,'group_id':group_id,'scene_id':split+':'+path,'path':path,'indices':good[:T].tolist()})
        def loader(r):
            group=handles[r['official_split']][r['path']];cube=group['highresdynamic'];indices=r['indices']
            if len(cube.shape)!=4 or cube.shape[2]<5:raise ValueError('Expected H,W,C,T EarthNet cube')
            frames=np.asarray(cube[:,:,:4,:]).transpose(3,2,0,1)[indices].astype('float32')
            if not np.isfinite(frames).all() or frames.min()<0 or frames.max()>1:raise ValueError('Cloud-free target values must be finite released reflectance in [0,1]; inspect bad/flagged frames rather than silently replace NaNs')
            if frames.shape[-2:]!=(size,size):
                frames=np.stack([np.stack([np.asarray(Image.fromarray(band).resize((size,size),resample=Image.Resampling.BOX),'float32') for band in frame]) for frame in frames])
            visible=np.ones((T,1,size,size),bool);rng=seeded(seed,'earthnet_mask',r['scene_id']);half=size//2
            visible[:,0,half:,half:]=False  # never observed by this protocol; no independent-texture claim
            visible[-1,0,:half,half:]=False
            for t in range(T-1):
                y=int(rng.integers(max(1,size//2)));x=int(rng.integers(max(1,size//2)));visible[t,0,y:y+half,x:x+half]=False
            x=np.concatenate([np.where(visible,frames,0),visible],1).astype('float32');target=frames[-1]
            # Author filename encodes meteorological start; Sentinel2 offset is four days, nominal grid five days.
            tokens=Path(r['path']).name.split('_')
            if len(tokens)<3:raise ValueError('EarthNet date-bearing cube name required')
            start=dt.date.fromisoformat(tokens[1])+dt.timedelta(days=4)
            dates=[start+dt.timedelta(days=5*int(j)) for j in indices];days=np.array([(v-dates[0]).days for v in dates],'float32')
            current=visible[-1];past=visible[:-1].any(0);ever=visible.any(0)
            return {'x':x,'y':target,'timestamps':days,'availability_times':days,'target_deadline':float(days[-1]),
                'region_C':current,'region_H':past&~current,'region_U_vis':~ever}
        contract={'dataset_id':'D94_earthnet_causal_rectangles_v1','name':'EarthNet2021 causal clean-target rectangular gaps',
            'target_mode':'continuous','classes':None,'output_shape':[4,size,size],'group_unit':group_unit,
            'author_schema_commit':'1ea50fe1e4f1840a14b3f23062334077fc5f943b','bands':['B2','B3','B4','B8'],
            'target_quality':'author idx_good_frames; clean-frame selection bias explicitly retained','native_seq2seq_reproduction':False,
            'mask_protocol':'independent synthetic rectangles; not natural cloud-mask simulation','skipped_short_cubes':skipped,
            'timing':'retained original frame indices × five days; no future observations; zero preprocessing latency assumed',
            'geographic_overlap_audit_complete':False,'raw_sources':[{'path':str(v),'sha256':file_hash(v)} for v in files.values()]}
        return write_records(records,loader,out,contract,seed)
    finally:
        for h in handles.values():h.close()
