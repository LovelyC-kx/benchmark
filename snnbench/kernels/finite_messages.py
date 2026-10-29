"""Finite legal messages, collisions, precision and exact Bayes references."""
import numpy as np
from scipy.special import ndtr


def binary_words(bits):
    if isinstance(bits,bool) or int(bits)!=bits or not 1<=bits<=16:raise ValueError('Explicit small binary alphabet, <=16 bits')
    bits=int(bits)
    return ((np.arange(2**bits)[:,None]>>np.arange(bits))&1).astype(np.uint8)


def exact_message_risk(messages,targets,weights=None):
    """Finite declared support, not a population estimate from observed samples."""
    x,y=np.asarray(messages),np.asarray(targets,float)
    if x.ndim==1:x=x[:,None]
    if y.ndim==1:y=y[:,None]
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite support with aligned targets')
    p=np.ones(len(x))/len(x) if weights is None else np.asarray(weights,float)
    if p.shape!=(len(x),) or p.min()<0 or not np.isclose(p.sum(),1):raise ValueError('Declared support probabilities')
    _,ids=np.unique(x,axis=0,return_inverse=True);groups=int(ids.max())+1
    masses=np.bincount(ids,weights=p,minlength=groups);means=np.zeros((groups,y.shape[1]));positive=np.zeros_like(means)
    for k in range(y.shape[1]):
        np.divide(np.bincount(ids,weights=p*y[:,k],minlength=groups),masses,out=means[:,k],where=masses>0)
        positive[:,k]=np.bincount(ids,weights=p*(y[:,k]>0),minlength=groups)
    mse=np.sum(p[:,None]*(y-means[ids])**2,axis=0)
    errors=np.minimum(positive,masses[:,None]-positive).sum(0)
    entropy=-np.sum(masses[masses>0]*np.log2(masses[masses>0]))
    return {'support_points':len(x),'distinct_messages':groups,'message_entropy_bits':float(entropy),
            'Bayes_MSE_by_target':mse,'Bayes_sign_error_by_target':errors,
            'scope':'Exact finite support; target sign uses >0. Not a learned decoder or empirical SNN capacity estimate.'}


def equally_spaced_error(symbols,spacing,sigma):
    if symbols<2 or spacing<=0 or sigma<0:raise ValueError('Uniform one-dimensional constellation with Gaussian additive noise')
    return 0. if sigma==0 else float(2*(1-1/symbols)*ndtr(-spacing/(2*sigma)))


def admissible_binary(messages):
    x=np.asarray(messages)
    if x.ndim!=3 or not np.isfinite(x).all() or not np.isin(x,[0,1]).all():raise ValueError('An actual binary edge override must stay binary')
    return x


def four_geometry_weights(source_direction,target_direction,source_basis,target_basis):
    """Four orthogonal components of one outer-product modulation term."""
    x,y,p,q=map(lambda v:np.asarray(v,float),(source_direction,target_direction,source_basis,target_basis))
    if x.ndim!=1 or y.ndim!=1 or not np.allclose(p.T@p,np.eye(p.shape[1])) or not np.allclose(q.T@q,np.eye(q.shape[1])):raise ValueError('Directions and orthonormal reference subspaces')
    sx=float(np.sum((p.T@x)**2)/np.sum(x*x));sy=float(np.sum((q.T@y)**2)/np.sum(y*y))
    return {'ss':sx*sy,'ns':(1-sx)*sy,'sn':sx*(1-sy),'nn':(1-sx)*(1-sy)}
