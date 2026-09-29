"""Training-task-only monotone conservative sigma mapping and pair requirements."""
import numpy as np


def weighted_quantile(values,weights,q=.9):
    values,weights=np.asarray(values),np.asarray(weights,dtype=np.float64)
    if values.ndim!=1 or weights.shape!=values.shape or not len(values) or not 0<q<=1:
        raise ValueError('Invalid weighted quantile inputs')
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or not (weights>0).all():
        raise ValueError('Invalid weighted quantile data')
    order=np.argsort(values,kind='stable')
    cumulative=np.cumsum(weights[order])
    index=min(int(np.searchsorted(cumulative,q*cumulative[-1],side='left')),len(values)-1)
    return int(values[order[index]])


def fit_g(s,sigma_indices,tasks):
    s,y,tasks=np.asarray(s,dtype=np.float64),np.asarray(sigma_indices),np.asarray(tasks)
    if s.ndim!=1 or y.shape!=s.shape or tasks.shape!=s.shape or len(s)<2:
        raise ValueError('Invalid mapping shapes')
    if not np.isfinite(s).all() or not np.isfinite(y).all() or not ((y>=0)&(y<=10)&(y==y.astype(int))).all():
        raise ValueError('Invalid signal or sigma label')
    weights=np.array([1./np.sum(tasks==t) for t in tasks])
    boundaries=np.quantile(s,[.2,.4,.6,.8])
    bins=np.searchsorted(boundaries,s,side='right')
    fallback=weighted_quantile(y,weights)
    values,support=[],[]
    for b in range(5):
        mask=bins==b
        values.append(weighted_quantile(y[mask],weights[mask]) if mask.any() else fallback)
        support.append(dict(states=int(mask.sum()),tasks=int(len(set(tasks[mask])))))
    return dict(boundaries=boundaries.tolist(),raw_sigma_indices=values,
        sigma_indices=np.maximum.accumulate(values).tolist(),support=support,
        quantile=.9,bin_count=5,training_tasks=sorted(set(tasks)),
        weighting='one total weight per training task; renormalized within bins')


def predict_g(mapping,s):
    s=np.asarray(s,dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite signal')
    bins=np.searchsorted(mapping['boundaries'],s,side='right')
    return np.asarray(mapping['sigma_indices'],dtype=np.int64)[bins]


def outer_mappings(s,sigma_indices,tasks):
    s,y,tasks=np.asarray(s),np.asarray(sigma_indices),np.asarray(tasks)
    unique=sorted(set(tasks))
    if len(unique)<3:
        raise ValueError('Need at least3 tasks')
    rows=[]
    requirements=set()
    for task in unique:
        training=tasks!=task
        mapping=fit_g(s[training],y[training],tasks[training])
        # Intentionally the SAME outer-trained g for both training and test rows.
        selected=predict_g(mapping,s)
        rows.append(dict(test_task=str(task),train_indices=np.flatnonzero(training).tolist(),
            test_indices=np.flatnonzero(~training).tolist(),mapping=mapping,
            sigma_indices_all_states=selected.tolist()))
        requirements.update((int(i),int(j)) for i,j in enumerate(selected) if j>0)
    return rows,sorted(requirements)
