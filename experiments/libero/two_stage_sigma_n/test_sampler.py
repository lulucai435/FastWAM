import importlib.util
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from sweep_a import run_phase_a_state

SOURCE=Path(os.environ.get('TWO_STAGE_SOURCE',str(Path(__file__).resolve().parents[3])))
spec=importlib.util.spec_from_file_location('two_stage_pinned_scheduler',SOURCE/'src/fastwam/models/wan22/schedulers/scheduler_continuous.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Scheduler=module.WanContinuousFlowMatchScheduler


class Toy:
    def __init__(self):
        self.device=torch.device('cpu');self.infer_action_scheduler=Scheduler();self.calls=0;self.invocations=[]

    def _denoise_action_with_video_cache(self,**kw):
        self.calls+=1
        return .125*kw['latents_action']+.01*kw['timestep_action'].reshape(1,1,1)

    def infer_action(self,**kw):
        self.invocations.append(kw)
        e=kw['initial_noise'].clone();sigma=kw['start_sigma']
        initial=e.clone()
        if 'action_prior' in kw:
            mask=kw['action_prior_mask']
            initial[mask]=(1-sigma)*kw['action_prior'][mask]+sigma*e[mask]
        x=initial.to(torch.bfloat16).unsqueeze(0)
        ts,ds=self.infer_action_scheduler.build_inference_schedule(kw['num_inference_steps'],self.device,
            x.dtype,shift_override=kw['sigma_shift'],start_sigma=sigma)
        for t,d in zip(ts,ds):
            v=self._denoise_action_with_video_cache(latents_action=x,timestep_action=t.unsqueeze(0))
            x=self.infer_action_scheduler.step(v,d,x)
        return dict(action=x[0].float(),sampling=dict(initial_noise=e,initial_flow_state=initial,
            actual_nfe=len(ts),requested_nfe=kw['num_inference_steps'],start_sigma=sigma,timesteps=ts.float(),deltas=ds.float()))


class SamplerTests(unittest.TestCase):
    def test_phase_a_ten_independent_n10_and_free_direct_reuse(self):
        model=Toy()
        p=np.arange(224,dtype=np.float32).reshape(32,7)/200
        eps=torch.randn((32,7),generator=torch.Generator().manual_seed(42)).numpy()
        r=run_phase_a_state(model,{'input_image':torch.zeros(1,3,32,32)},p,eps,compiled=False)
        self.assertEqual(model.calls,100)
        self.assertEqual(len(model.invocations),10)
        self.assertEqual(r['observed_nfe'],100)
        self.assertTrue(all(k['num_inference_steps']==10 for k in model.invocations))
        self.assertNotIn('action_prior',model.invocations[-1])
        np.testing.assert_array_equal(r['executed_actions'][0],p[16:])
        np.testing.assert_array_equal(r['initial_states'][-1],eps)
        self.assertEqual(float(r['errors'][-1]),0.)

    def test_baseline_schedule_bitwise_unchanged(self):
        scheduler=Scheduler()
        for dtype in (torch.float32,torch.bfloat16):
            actual=scheduler.build_inference_schedule(10,torch.device('cpu'),dtype,start_sigma=1.,shift_override=5.)
            default=scheduler.build_inference_schedule(10,torch.device('cpu'),dtype)
            u=torch.linspace(1,0,11,dtype=torch.float32)
            nodes=5*u/(1+4*u)
            expected=((nodes[:-1]*1000).to(dtype),torch.diff(nodes).to(dtype))
            for a,b,c in zip(actual,default,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                torch.testing.assert_close(a,c,rtol=0,atol=0)

    def test_sigma_n_independent_shared_first_and_proper_endpoints(self):
        scheduler=Scheduler()
        for sigma in np.arange(1,11)/10:
            first=[]
            for n in range(1,11):
                t,d=scheduler.build_inference_schedule(n,torch.device('cpu'),torch.bfloat16,
                    shift_override=5.,start_sigma=float(sigma))
                self.assertEqual(len(t),n);self.assertTrue((d<0).all())
                first.append(t[0])
                # Independent ideal flow nodes, allowing the declared BF16 casts.
                u=np.linspace(sigma/(5-4*sigma),0,n+1)
                nodes=5*u/(1+4*u)
                self.assertEqual(nodes[-1],0.)
                np.testing.assert_allclose(t.float().numpy()/1000,nodes[:-1],rtol=.004,atol=1e-7)
                np.testing.assert_allclose(d.float().numpy(),np.diff(nodes),rtol=.004,atol=1e-7)
            for t in first:
                torch.testing.assert_close(t,first[0],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
