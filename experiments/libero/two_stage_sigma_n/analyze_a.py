"""Phase A associations, task-held-out g, and label-blind Phase B pair requirements."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from definitions import DELTA,sigma_star
from mapping import outer_mappings
from io_utils import read,save,sha

HERE=Path(__file__).resolve().parent


def correlation(x,y):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or len(np.unique(x))<2 or len(np.unique(y))<2:
        return None
    return float(spearmanr(x,y).statistic)


def demean(x,tasks):
    result=np.asarray(x,dtype=np.float64).copy()
    for task in set(tasks):
        mask=tasks==task
        result[mask]-=result[mask].mean()
    return result


def mapping_metrics(indices,oracle,errors,tasks):
    chosen_error=errors[np.arange(len(indices)),indices]
    rows=[]
    for task in sorted(set(tasks)):
        mask=tasks==task
        reuse=mask&(indices==0)
        rows.append(dict(task=str(task),states=int(mask.sum()),
            sigma_mae=float(np.abs(indices[mask]-oracle[mask]).mean()/10),
            mean_selected_sigma=float(indices[mask].mean()/10),
            below_suffix_oracle_rate=float((indices[mask]<oracle[mask]).mean()),
            action_threshold_failure_rate=float((chosen_error[mask]>=DELTA).mean()),
            direct_reuse_rate=float(reuse.sum()/mask.sum()),
            direct_reuse_failed_states=int((chosen_error[reuse]>=DELTA).sum())))
    return dict(pooled_sigma_mae=float(np.abs(indices-oracle).mean()/10),
        macro_task_sigma_mae=float(np.mean([r['sigma_mae'] for r in rows])),
        mean_selected_sigma=float(indices.mean()/10),below_suffix_oracle_rate=float((indices<oracle).mean()),
        action_threshold_failure_rate=float((chosen_error>=DELTA).mean()),
        macro_task_action_threshold_failure_rate=float(np.mean([r['action_threshold_failure_rate'] for r in rows])),
        direct_reuse_rate=float((indices==0).mean()),direct_reuse_states=int((indices==0).sum()),
        direct_reuse_failed_states=int(((indices==0)&(chosen_error>=DELTA)).sum()),per_task=rows)


def analyze(errors,signal,tasks,s_bins):
    errors=np.asarray(errors,dtype=np.float32)
    s=np.asarray(signal,dtype=np.float64)
    tasks=np.asarray(tasks);bins=np.asarray(s_bins)
    if errors.shape!=(len(s),11) or tasks.shape!=s.shape or bins.shape!=s.shape:
        raise ValueError('Invalid analysis shapes')
    if not np.isfinite(errors).all() or not np.isfinite(s).all() or (errors<0).any() or (errors[:,-1]!=0).any():
        raise ValueError('Invalid errors/reference/signal')
    oracle=np.array([sigma_star(row) for row in errors],dtype=np.int64)
    folds,pairs=outer_mappings(s,oracle,tasks)
    selected=np.full(len(s),-1,dtype=np.int64)
    for fold in folds:
        test=np.array(fold['test_indices'],dtype=int)
        candidate=np.array(fold['sigma_indices_all_states'])
        assert (selected[test]==-1).all()
        selected[test]=candidate[test]
        training=np.array(fold['train_indices'],dtype=int)
        fold['training_metrics']=mapping_metrics(candidate[training],oracle[training],errors[training],tasks[training])
    assert (selected>=0).all()
    y=oracle/10.
    task_rows=[]
    for task in sorted(set(tasks)):
        mask=tasks==task
        task_rows.append(dict(task=str(task),states=int(mask.sum()),rho=correlation(s[mask],y[mask]),
            mean_sigma_star=float(y[mask].mean()),median_sigma_star=float(np.median(y[mask])),
            sigma_star_distribution=np.bincount(oracle[mask],minlength=11).tolist(),
            conservative_direct_reuse_rate=float((oracle[mask]==0).mean()),
            direct_action_pass_rate=float((errors[mask,0]<DELTA).mean())))
    values=[r['rho'] for r in task_rows if r['rho'] is not None]
    bin_rows=[]
    for b in range(5):
        mask=bins==b
        cells=[dict(task=str(t),states=int((mask&(tasks==t)).sum()),
                    mean_sigma_star=float(y[mask&(tasks==t)].mean()))
               for t in sorted(set(tasks)) if (mask&(tasks==t)).sum()>=3]
        means=[c['mean_sigma_star'] for c in cells]
        bin_rows.append(dict(s_bin=b,states=int(mask.sum()),
            sigma_star_mean=float(y[mask].mean()) if mask.any() else None,
            sigma_star_variance=float(y[mask].var()) if mask.any() else None,
            eligible_task_cells=cells,task_mean_spread=max(means)-min(means) if len(means)>1 else None))
    result=dict(states=len(s),tasks=len(set(tasks)),delta=DELTA,
        pooled_rho=correlation(s,y),task_demeaned_rho=correlation(demean(s,tasks),demean(y,tasks)),
        per_task_signs=dict(positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),
            zero=sum(v==0 for v in values),undefined=len(task_rows)-len(values)),per_task=task_rows,
        sigma_star_distribution=np.bincount(oracle,minlength=11).tolist(),
        mean_sigma_star=float(y.mean()),median_sigma_star=float(np.median(y)),
        conservative_direct_reuse_rate=float((oracle==0).mean()),
        direct_action_pass_rate=float((errors[:,0]<DELTA).mean()),
        error_nonmonotonic_states=int((np.diff(errors,axis=1)>0).any(axis=1).sum()),
        error_nonmonotonic_states_above_1e_6=int((np.diff(errors,axis=1)>1e-6).any(axis=1).sum()),
        heldout_g=mapping_metrics(selected,oracle,errors,tasks),within_s_bins=bin_rows,
        phase_b_required_positive_pairs=len(pairs),phase_b_nominal_shared_first_nfe=46*len(pairs),
        phase_b_preflight_warmup_cost_not_yet_included=True,
        verdict='PENDING_FULL_TWO_STAGE_REVIEW',overall_goal_complete=False)
    return result,folds,pairs,oracle,selected


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_analysis(output,errors,entries,provenance):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Preserve completed analysis; choose a new output')
    s=np.array([e['signal'] for e in entries])
    tasks=np.array([f"{e['key'][0]}/task{e['key'][1]}" for e in entries])
    bins=np.array([e['signal_bin'] for e in entries])
    summary,folds,pairs,oracle,selected=analyze(errors,s,tasks,bins)
    output.mkdir(parents=True)
    # Freeze outer mappings/pair selection before subsequent heldout comparisons.
    save(output/'FOLDS.json',dict(folds=folds,provenance=provenance))
    save(output/'PHASE_B_PAIRS.json',dict(pairs=[dict(state_id=entries[i]['id'],key=entries[i]['key'],
        sigma_index=j,start_sigma=j/10.) for i,j in pairs],
        nominal_measured_nfe=46*len(pairs),source_folds_sha256=sha(output/'FOLDS.json'),
        gpu_authorized=False,provenance=provenance))
    np.savez_compressed(output/'LABELS_AND_OOF.npz',errors=errors,s=s,tasks=tasks,
        sigma_star_indices=oracle,selected_sigma_indices=selected,s_bins=bins,
        keys=np.array([str(e['key']) for e in entries]))
    summary.update(provenance=provenance,folds_sha256=sha(output/'FOLDS.json'),
                   arrays_sha256=sha(output/'LABELS_AND_OOF.npz'))
    save(output/'SUMMARY.json',summary)
    write_csv(output/'per_task.csv',[{k:v for k,v in r.items() if k!='sigma_star_distribution'} for r in summary['per_task']])
    write_csv(output/'heldout_g_per_task.csv',summary['heldout_g']['per_task'])
    write_csv(output/'states.csv',[dict(id=e['id'],key=str(e['key']),signal=e['signal'],
        sigma_star=oracle[i]/10.,selected_sigma=selected[i]/10.,
        selected_action_error=float(errors[i,selected[i]]),direct_reuse=bool(selected[i]==0))
        for i,e in enumerate(entries)])
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from review_a import load_reviewed_phase_a
    errors,entries,evidence=load_reviewed_phase_a(args.raw,args.plan)
    evidence['analysis_code']={name:sha(HERE/name) for name in ('analyze_a.py','mapping.py','definitions.py','review_a.py')}
    result=write_analysis(args.output,errors,entries,evidence)
    print('PHASE_A_ANALYSIS_COMPLETE',result['pooled_rho'],result['phase_b_required_positive_pairs'],flush=True)


if __name__=='__main__':
    main()
