"""Independent Phase A artifact validation; no model or environment inference."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from io_utils import read,save,sha,validate_plan


def close(actual,expected):
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-10)


def suffix_reference(errors):
    # Match the scalar threshold's cast in the specified float32 error comparison.
    threshold=np.asarray(.01,dtype=np.asarray(errors).dtype).item()
    for index in range(len(errors)):
        if all(float(value)<threshold for value in errors[index:]):
            return index
    return None


def verify_state(raw,row,entry):
    raw=Path(raw)
    assert row['id']==entry['id'] and row['key']==entry['key']
    assert row['file']==f"states/state{entry['id']:05d}.npz"
    assert sha(raw/row['file'])==row['sha256']
    for which in ('previous','current'):
        assert sha(entry[which]['file'])==entry[which]['sha256']==row[which+'_sha256']
    with np.load(entry['previous']['file'],allow_pickle=False) as a:
        previous=a['normalized_action'].copy()
    with np.load(entry['current']['file'],allow_pickle=False) as a:
        epsilon=a['initial_noise'].copy()
        assert a['normalized_proprio'].shape==(1,8)
    np.testing.assert_array_equal(previous[16:],np.asarray(entry['prior_normalized_16x7'],dtype=np.float32))
    with np.load(raw/row['file'],allow_pickle=False) as a:
        arrays={key:a[key].copy() for key in a.files}
    shapes=dict(executed_actions=(11,16,7),full_positive_actions=(10,32,7),errors=(11,),
        initial_states=(10,32,7),first_network_inputs=(10,32,7),epsilon=(32,7))
    assert set(arrays)==set(shapes)
    for name,shape in shapes.items():
        assert arrays[name].shape==shape and arrays[name].dtype==np.float32 and np.isfinite(arrays[name]).all(),name
    np.testing.assert_array_equal(arrays['epsilon'],epsilon)
    np.testing.assert_array_equal(epsilon,torch.randn((1,32,7),generator=torch.Generator().manual_seed(42))[0].numpy())
    np.testing.assert_array_equal(arrays['executed_actions'][0],previous[16:])
    np.testing.assert_array_equal(arrays['executed_actions'][1:],arrays['full_positive_actions'][:,:16])
    for j,trace in enumerate(row['traces'],1):
        sigma=j/10.
        assert trace['sigma_index']==j and trace['start_sigma']==sigma
        assert trace['N']==trace['actual_nfe']==10 and trace['dtype']=='torch.bfloat16'
        assert trace['branch']==('original fresh' if j==10 else 'masked prior')
        expected=epsilon.copy()
        expected[:16]=(1-sigma)*previous[16:]+sigma*epsilon[:16]
        np.testing.assert_array_equal(arrays['initial_states'][j-1],expected)
        cast=torch.from_numpy(expected).to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(arrays['first_network_inputs'][j-1],cast)
        u=np.linspace(sigma/(5-4*sigma),0,11,dtype=np.float64)
        nodes=5*u/(1+4*u)
        t,delta=np.array(trace['timesteps']),np.array(trace['deltas'])
        assert t.shape==delta.shape==(10,) and (np.diff(t)<0).all() and (delta<0).all()
        np.testing.assert_allclose(t/1000,nodes[:-1],rtol=.004,atol=1e-7)
        np.testing.assert_allclose(delta,np.diff(nodes),rtol=.004,atol=1e-7)
    assert len(row['traces'])==10 and row['observed_nfe']==100
    a=arrays['executed_actions']
    direct=np.sqrt(np.mean((a-a[-1])**2,axis=(1,2),dtype=np.float32))
    np.testing.assert_array_equal(arrays['errors'],direct)
    double=np.array([np.sqrt(np.mean((chunk.astype(np.float64)-a[-1].astype(np.float64))**2)) for chunk in a])
    np.testing.assert_allclose(direct,double,rtol=2e-6,atol=1e-8)
    assert direct[-1]==0 and row['sigma_star_index']==suffix_reference(direct)
    return direct


def load_reviewed_phase_a(raw,plan_path):
    raw,plan_path=Path(raw),Path(plan_path)
    plan=validate_plan(plan_path)
    assert plan['gpu_authorized'] and raw.resolve()==Path(plan['output']).resolve()
    complete=read(raw/'PHASE_A_COMPLETE.json')
    assert complete['plan_sha256']==sha(plan_path)
    assert complete['inputs_sha256']==sha(plan_path.parent/'INPUTS.json')
    assert complete['runtime_sha256']==sha(raw/'runtime.json')
    assert complete['counts']==dict(warmup_nfe=30,measured_nfe=77200,states=772,environment_steps=0)
    runtime=read(raw/'runtime.json')
    assert runtime['phase']=='A' and runtime['sigma_shift']==5. and runtime['compiled']
    assert runtime['dtype']=='bfloat16' and runtime['environment_steps']==0 and runtime['final_allocator_training_steps']==0
    assert runtime['source_manifest_sha256']==sha(Path(plan['source'])/'SOURCE_MANIFEST.json')
    assert runtime['plan_sha256']==sha(plan_path)
    catalog=read(plan_path.parent/'INPUTS.json')
    entries=catalog['entries']
    assert len(entries)==len(complete['states'])==772 and catalog['tasks']==28
    assert [e['id'] for e in entries]==list(range(772))
    rows=[]
    for i,(row,entry) in enumerate(zip(complete['states'],entries)):
        rows.append(verify_state(raw,row,entry))
        if i%100==0:
            print('PHASE_A_RAW_REVIEW',i,flush=True)
    return np.stack(rows),entries,dict(raw=str(raw),raw_complete_sha256=sha(raw/'PHASE_A_COMPLETE.json'),
        plan=str(plan_path),plan_sha256=sha(plan_path),all_772_states_independently_reviewed=True,
        reviewer_sha256=sha(__file__),measured_nfe=77200,warmup_nfe=30)


def rho_reference(a,b):
    if len(a)<3 or len(set(a))<2 or len(set(b))<2:
        return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


def mapping_reference(s,y,tasks):
    edges=np.quantile(s,[.2,.4,.6,.8])
    weights=np.array([1/np.sum(tasks==t) for t in tasks],dtype=np.float64)
    bins=np.searchsorted(edges,s,side='right')
    def quantile(mask):
        yy,ww=y[mask],weights[mask]
        # Independent discrete-grid CDF, not the production sort/search routine.
        for v in range(11):
            if ww[yy<=v].sum()>=.9*ww.sum()-1e-14:
                return v
        raise AssertionError('No weighted quantile')
    raw=[quantile(bins==b) if (bins==b).any() else quantile(np.ones(len(s),dtype=bool)) for b in range(5)]
    monotone=[max(raw[:i+1]) for i in range(5)]
    return edges,raw,monotone


def verify_metrics(metrics,chosen_indices,oracle,errors,tasks):
    selected=chosen_indices;chosen=errors[np.arange(len(errors)),selected]
    close(metrics['pooled_sigma_mae'],np.abs(selected-oracle).mean()/10)
    close(metrics['macro_task_sigma_mae'],np.mean([np.abs(selected[tasks==t]-oracle[tasks==t]).mean()/10 for t in set(tasks)]))
    close(metrics['mean_selected_sigma'],selected.mean()/10)
    close(metrics['below_suffix_oracle_rate'],(selected<oracle).mean())
    close(metrics['action_threshold_failure_rate'],(chosen>=.01).mean())
    close(metrics['macro_task_action_threshold_failure_rate'],np.mean([(chosen[tasks==t]>=.01).mean() for t in set(tasks)]))
    close(metrics['direct_reuse_rate'],(selected==0).mean())
    assert metrics['direct_reuse_states']==int((selected==0).sum())
    assert metrics['direct_reuse_failed_states']==int(((selected==0)&(chosen>=.01)).sum())
    assert len(metrics['per_task'])==len(set(tasks))
    for row in metrics['per_task']:
        mask=tasks==row['task'];reuse=mask&(selected==0)
        assert row['states']==mask.sum()
        close(row['sigma_mae'],np.abs(selected[mask]-oracle[mask]).mean()/10)
        close(row['mean_selected_sigma'],selected[mask].mean()/10)
        close(row['below_suffix_oracle_rate'],(selected[mask]<oracle[mask]).mean())
        close(row['action_threshold_failure_rate'],(chosen[mask]>=.01).mean())
        close(row['direct_reuse_rate'],reuse.sum()/mask.sum())
        assert row['direct_reuse_failed_states']==int((chosen[reuse]>=.01).sum())


def review_analysis(output):
    output=Path(output)
    summary=read(output/'SUMMARY.json');frozen=read(output/'FOLDS.json');pair_data=read(output/'PHASE_B_PAIRS.json')
    assert summary['folds_sha256']==sha(output/'FOLDS.json')==pair_data['source_folds_sha256']
    assert summary['arrays_sha256']==sha(output/'LABELS_AND_OOF.npz')
    with np.load(output/'LABELS_AND_OOF.npz',allow_pickle=False) as a:
        data={k:a[k].copy() for k in a.files}
    e,s,tasks=data['errors'],data['s'],data['tasks']
    expected=np.array([suffix_reference(row) for row in e],dtype=int)
    np.testing.assert_array_equal(data['sigma_star_indices'],expected)
    selected=np.full(len(s),-1,dtype=int)
    pairs=set()
    assert len(frozen['folds'])==len(set(tasks))
    for fold in frozen['folds']:
        mask=tasks!=fold['test_task'];test=~mask
        np.testing.assert_array_equal(fold['train_indices'],np.flatnonzero(mask))
        np.testing.assert_array_equal(fold['test_indices'],np.flatnonzero(test))
        assert fold['mapping']['training_tasks']==sorted(set(tasks[mask]))
        assert fold['mapping']['quantile']==.9 and fold['mapping']['bin_count']==5
        edges,raw,monotone=mapping_reference(s[mask],expected[mask],tasks[mask])
        close(edges,fold['mapping']['boundaries'])
        np.testing.assert_array_equal(raw,fold['mapping']['raw_sigma_indices'])
        np.testing.assert_array_equal(monotone,fold['mapping']['sigma_indices'])
        training_bins=np.searchsorted(edges,s[mask],side='right')
        for bin_id,support in enumerate(fold['mapping']['support']):
            cell=training_bins==bin_id
            assert support['states']==cell.sum() and support['tasks']==len(set(tasks[mask][cell]))
        all_pred=np.array([monotone[np.searchsorted(edges,value,side='right')] for value in s])
        np.testing.assert_array_equal(all_pred,fold['sigma_indices_all_states'])
        verify_metrics(fold['training_metrics'],all_pred[mask],expected[mask],e[mask],tasks[mask])
        selected[test]=all_pred[test]
        pairs.update((i,int(j)) for i,j in enumerate(all_pred) if j>0)
    np.testing.assert_array_equal(selected,data['selected_sigma_indices'])
    actual_pairs={(r['state_id'],r['sigma_index']) for r in pair_data['pairs']}
    for row in pair_data['pairs']:
        assert str(row['key'])==data['keys'][row['state_id']]
        assert row['start_sigma']==row['sigma_index']/10.
    assert actual_pairs==pairs and len(pair_data['pairs'])==len(pairs)
    assert pair_data['nominal_measured_nfe']==summary['phase_b_nominal_shared_first_nfe']==46*len(pairs)
    assert not pair_data['gpu_authorized'] and not summary['overall_goal_complete']
    y=expected/10.
    a,b=s.copy(),y.copy()
    for task in set(tasks):
        mask=tasks==task;a[mask]-=s[mask].mean();b[mask]-=y[mask].mean()
    for key,value in [('pooled_rho',rho_reference(s,y)),('task_demeaned_rho',rho_reference(a,b))]:
        if value is None:
            assert summary[key] is None
        else:
            close(summary[key],value)
    np.testing.assert_array_equal(summary['sigma_star_distribution'],np.bincount(expected,minlength=11))
    close(summary['mean_sigma_star'],y.mean());close(summary['median_sigma_star'],np.median(y))
    close(summary['conservative_direct_reuse_rate'],(expected==0).mean())
    close(summary['direct_action_pass_rate'],(e[:,0]<.01).mean())
    signs=dict(positive=0,negative=0,zero=0,undefined=0)
    for row in summary['per_task']:
        mask=tasks==row['task'];value=rho_reference(s[mask],y[mask])
        assert row['states']==mask.sum()
        if value is None:
            assert row['rho'] is None;signs['undefined']+=1
        else:
            close(row['rho'],value);signs['positive' if value>0 else 'negative' if value<0 else 'zero']+=1
        close(row['mean_sigma_star'],y[mask].mean());close(row['median_sigma_star'],np.median(y[mask]))
        np.testing.assert_array_equal(row['sigma_star_distribution'],np.bincount(expected[mask],minlength=11))
        close(row['conservative_direct_reuse_rate'],(expected[mask]==0).mean())
        close(row['direct_action_pass_rate'],(e[mask,0]<.01).mean())
    assert signs==summary['per_task_signs']
    verify_metrics(summary['heldout_g'],selected,expected,e,tasks)
    assert summary['error_nonmonotonic_states']==int((np.diff(e,axis=1)>0).any(axis=1).sum())
    assert summary['error_nonmonotonic_states_above_1e_6']==int((np.diff(e,axis=1)>1e-6).any(axis=1).sum())
    for row in summary['within_s_bins']:
        mask=data['s_bins']==row['s_bin']
        assert row['states']==mask.sum()
        if mask.any():
            close(row['sigma_star_mean'],y[mask].mean());close(row['sigma_star_variance'],y[mask].var())
        else:
            assert row['sigma_star_mean'] is None and row['sigma_star_variance'] is None
        eligible=sorted(t for t in set(tasks) if (mask&(tasks==t)).sum()>=3)
        assert [c['task'] for c in row['eligible_task_cells']]==eligible
        means=[]
        for cell in row['eligible_task_cells']:
            support=mask&(tasks==cell['task'])
            assert cell['states']==support.sum();close(cell['mean_sigma_star'],y[support].mean())
            means.append(y[support].mean())
        if len(means)>1:
            close(row['task_mean_spread'],max(means)-min(means))
        else:
            assert row['task_mean_spread'] is None
    save(output/'ANALYSIS_REVIEW.json',dict(passed=True,states=len(s),tasks=len(set(tasks)),
        mappings_independently_recomputed=True,outer_tasks_excluded=True,pair_union_verified=True,
        suffix_labels_and_associations_verified=True,mapping_metrics_and_bin_spread_verified=True,
        overall_goal_complete=False,
        summary_sha256=sha(output/'SUMMARY.json'),reviewer_sha256=sha(__file__)))
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',required=True,type=Path)
    args=parser.parse_args();review_analysis(args.analysis)
    print('PHASE_A_ANALYSIS_REVIEW_PASSED',flush=True)
