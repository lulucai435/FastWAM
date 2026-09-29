"""Phase B continuation core. The caller supplies an ALREADY EXECUTED first NFE.

No model loading, Video prefill, first-forward probe, or resource operation here.
The model adapter/baseline-equivalence preflight remains a separate required step.
"""
import torch


def layer29_dispersion(hidden,layer=29):
    if layer!=29 or hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[1]!=32:
        raise ValueError('Frozen layer29 signal requires batch1/all32 action tokens')
    if not torch.isfinite(hidden).all():
        raise ValueError('Nonfinite first-step hidden state')
    h=hidden[0].detach().float()
    return (h-h.mean(dim=0,keepdim=True)).square().mean(dim=0).sqrt().mean()


@torch.no_grad()
def continue_all_n(scheduler,initial,first_timestep,first_velocity,first_hidden,denoise_rest,*,sigma):
    """Independent proper N=1..10; shared first NFE +45 continuation evaluations.

    denoise_rest(x,t) must invoke the ordinary denoiser using the SAME conditioning
    and cache as the first result. The GPU adapter must establish and trace this.
    d is reduced from that already-produced first hidden state, without a call.
    """
    if not 0<sigma<=1:
        raise ValueError('Sigma0 is direct reuse, not a refinement sweep')
    if initial.shape!=(1,32,7) or first_velocity.shape!=initial.shape:
        raise ValueError('Expected model-dtype batch1 H32 D7 tensors')
    if first_velocity.dtype!=initial.dtype or first_velocity.device!=initial.device:
        raise ValueError('Initial state and first velocity must share dtype/device')
    if not torch.isfinite(initial).all() or not torch.isfinite(first_velocity).all():
        raise ValueError('Nonfinite sampler inputs')
    x0=initial.detach().clone();v0=first_velocity.detach().clone()
    d=float(layer29_dispersion(first_hidden).cpu())
    actions,traces=[],[]
    continued=0
    for n in range(1,11):
        ts,ds=scheduler.build_inference_schedule(n,device=x0.device,dtype=x0.dtype,
            shift_override=5.,start_sigma=float(sigma))
        torch.testing.assert_close(ts[0].reshape(-1),first_timestep.reshape(-1).to(ts),rtol=0,atol=0)
        # First velocity is not a prefix checkpoint: apply this N's OWN first delta.
        x=scheduler.step(v0,ds[0],x0.clone())
        for t,delta in zip(ts[1:],ds[1:]):
            velocity=denoise_rest(x,t.unsqueeze(0))
            continued+=1
            assert velocity.shape==x.shape and velocity.dtype==x.dtype
            x=scheduler.step(velocity,delta,x)
        assert torch.isfinite(x).all()
        actions.append(x[0].detach().cpu().float().clone())
        traces.append(dict(N=n,start_sigma=float(sigma),logical_nfe=n,
            reused_first_nfe=True,new_continuation_nfe=n-1,
            timesteps=ts.detach().cpu().float().tolist(),deltas=ds.detach().cpu().float().tolist()))
    assert continued==45
    torch.testing.assert_close(initial,x0,rtol=0,atol=0)
    torch.testing.assert_close(first_velocity,v0,rtol=0,atol=0)
    return dict(actions=torch.stack(actions),d_layer29=d,traces=traces,
        reused_first_nfe=1,new_continuation_nfe=45,total_physical_nfe_including_first=46,
        sum_logical_candidate_nfe=55)
