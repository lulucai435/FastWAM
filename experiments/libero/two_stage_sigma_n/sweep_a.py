"""Phase A uses the unmodified production inference call with independent sigma."""
import numpy as np
import torch
from definitions import initialize,rmse,sigma_star


class Observe:
    def __init__(self,model,compiled):
        self.model=model
        self.name='_denoise_action_with_video_cache'+('_compiled' if compiled else '')
        self.original=getattr(model,self.name)
        self.instance=self.name in vars(model)
        self.timesteps=[]
        self.first=None

    def call(self,**kw):
        self.timesteps.append(kw['timestep_action'].detach().cpu().float().clone())
        if self.first is None:
            self.first=kw['latents_action'].detach().cpu().float().clone()
            self.dtype=kw['latents_action'].dtype
        return self.original(**kw)

    def __enter__(self):
        setattr(self.model,self.name,self.call)
        return self

    def __exit__(self,*exc):
        if self.instance:
            setattr(self.model,self.name,self.original)
        else:
            delattr(self.model,self.name)


@torch.no_grad()
def run_phase_a_state(model,conditioning,previous,epsilon,compiled=True):
    if set(conditioning)-{'prompt','input_image','proprio','tiled','context','context_mask'}:
        raise ValueError('Conditioning cannot override sampler controls')
    previous=np.asarray(previous,dtype=np.float32)
    epsilon=np.asarray(epsilon,dtype=np.float32)
    initialize(previous,epsilon,1.)
    prior=torch.zeros((32,7),dtype=torch.float32)
    prior[:16]=torch.from_numpy(previous[16:])
    mask=torch.arange(32)<16
    actions=[previous[16:].copy()]
    full,initials,firsts,traces=[],[],[],[]
    for index in range(1,11):
        sigma=index/10.
        kw=dict(conditioning,action_horizon=32,num_inference_steps=10,sigma_shift=5.,
            seed=42,rand_device='cpu',compile_action_infer=compiled,
            initial_noise=torch.from_numpy(epsilon.copy()),start_sigma=sigma,
            return_sampling_metadata=True)
        if index<10:
            kw.update(action_prior=prior.clone(),action_prior_mask=mask.clone())
        # At sigma1 this is exactly the standard fresh N10 inference API branch.
        with Observe(model,compiled) as observed:
            result=model.infer_action(**kw)
        m=result['sampling']
        assert len(observed.timesteps)==m['actual_nfe']==m['requested_nfe']==10
        assert m['start_sigma']==sigma
        expected=torch.from_numpy(initialize(previous,epsilon,sigma))
        torch.testing.assert_close(m['initial_noise'],torch.from_numpy(epsilon),rtol=0,atol=0)
        torch.testing.assert_close(m['initial_flow_state'],expected,rtol=0,atol=0)
        torch.testing.assert_close(observed.first[0],expected.to(observed.dtype).float(),rtol=0,atol=0)
        ts=torch.cat(observed.timesteps)
        delta=m['deltas'].cpu().float()
        reference_t,reference_delta=model.infer_action_scheduler.build_inference_schedule(
            10,device=model.device,dtype=observed.dtype,shift_override=5.,start_sigma=sigma)
        torch.testing.assert_close(ts,m['timesteps'].cpu().float(),rtol=0,atol=0)
        torch.testing.assert_close(ts,reference_t.cpu().float(),rtol=0,atol=0)
        torch.testing.assert_close(delta,reference_delta.cpu().float(),rtol=0,atol=0)
        assert (delta<0).all() and (ts[:-1]>ts[1:]).all()
        action=result['action'].cpu().float().numpy().copy()
        assert action.shape==(32,7) and np.isfinite(action).all()
        actions.append(action[:16].copy());full.append(action)
        initials.append(m['initial_flow_state'].cpu().numpy().copy())
        firsts.append(observed.first[0].numpy().copy())
        traces.append(dict(sigma_index=index,start_sigma=sigma,N=10,actual_nfe=10,
            timesteps=ts.tolist(),deltas=delta.tolist(),dtype=str(observed.dtype),
            branch='original fresh' if index==10 else 'masked prior'))
    actions=np.stack(actions)
    errors=rmse(actions,actions[-1])
    assert errors[-1]==0
    return dict(executed_actions=actions,full_positive_actions=np.stack(full),errors=errors,
        initial_states=np.stack(initials),first_network_inputs=np.stack(firsts),
        epsilon=epsilon.copy(),sigma_star_index=sigma_star(errors),observed_nfe=100,traces=traces)
