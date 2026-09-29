"""Pure definitions: sigma and N are independent; infeasible labels stay absent."""
import numpy as np

SIGMAS=np.arange(11,dtype=np.float64)/10
DELTA=.01


def initialize(previous,epsilon,sigma):
    p,e=np.asarray(previous,dtype=np.float32),np.asarray(epsilon,dtype=np.float32)
    if p.shape!=(32,7) or e.shape!=(32,7) or not np.isfinite(p).all() or not np.isfinite(e).all():
        raise ValueError('Expected finite previous and epsilon32x7')
    if not 0<=sigma<=1:
        raise ValueError('sigma outside[0,1]')
    x=e.copy()
    x[:16]=(1-float(sigma))*p[16:]+float(sigma)*e[:16]
    return x


def rmse(actions,reference):
    a,b=np.asarray(actions,dtype=np.float32),np.asarray(reference,dtype=np.float32)
    if a.shape[-2:]!=(16,7) or b.shape!=(16,7) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite executed16x7 actions')
    return np.sqrt(np.mean(np.square(a-b),axis=(-2,-1),dtype=np.float32))


def conservative_index(errors,delta=DELTA):
    e=np.asarray(errors)
    if e.ndim!=1 or not len(e) or not np.isfinite(e).all() or delta<=0:
        raise ValueError('Invalid errors or threshold')
    suffix=np.logical_and.accumulate((e<delta)[::-1])[::-1]
    hits=np.flatnonzero(suffix)
    return int(hits[0]) if len(hits) else None


def sigma_star(errors):
    if np.asarray(errors).shape!=(11,):
        raise ValueError('Need11 sigma candidates')
    index=conservative_index(errors)
    if index is None:
        raise ValueError('Fresh-reference endpoint must be feasible')
    return index


def n_star(errors):
    if np.asarray(errors).shape!=(10,):
        raise ValueError('Need10 N candidates')
    index=conservative_index(errors)
    return None if index is None else index+1


def integer_allocation(prediction):
    p=np.asarray(prediction,dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite prediction')
    return np.ceil(np.clip(p,1,10)).astype(np.int64)
