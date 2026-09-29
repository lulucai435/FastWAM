import unittest
import torch
from refinement import continue_all_n,layer29_dispersion
from test_sampler import Scheduler


class RefinementTests(unittest.TestCase):
    def test_shared_first_matches_all_independent_proper_candidates(self):
        scheduler=Scheduler()
        generator=torch.Generator().manual_seed(19)
        initial=torch.randn((1,32,7),generator=generator)
        counts={'calls':0}
        def ordinary(x,t):
            counts['calls']+=1
            return .05*x.square()+.1*x+torch.sin(t.reshape(1,1,1)/1000.)*.2
        for sigma in (.1,.4,1.):
            ts,_=scheduler.build_inference_schedule(10,initial.device,initial.dtype,start_sigma=sigma)
            first=ordinary(initial,ts[0:1])
            hidden=torch.cat([initial,initial.square()],dim=-1)
            counts['calls']=1
            result=continue_all_n(scheduler,initial,ts[0:1],first,hidden,ordinary,sigma=sigma)
            self.assertEqual(counts['calls'],46)
            self.assertEqual(result['sum_logical_candidate_nfe'],55)
            counts['calls']=0
            for n in range(1,11):
                t,delta=scheduler.build_inference_schedule(n,initial.device,initial.dtype,start_sigma=sigma)
                x=initial.clone()
                for step,change in zip(t,delta):
                    x=scheduler.step(ordinary(x,step.unsqueeze(0)),change,x)
                torch.testing.assert_close(result['actions'][n-1],x[0],rtol=0,atol=0)
            self.assertEqual(counts['calls'],55)

    def test_rejects_wrong_first_timestep_and_direct_reuse(self):
        x=torch.zeros(1,32,7);h=torch.ones(1,32,3)
        with self.assertRaises(AssertionError):
            continue_all_n(Scheduler(),x,torch.tensor([999.]),x,h,lambda x,t:x,sigma=1.)
        with self.assertRaises(ValueError):
            continue_all_n(Scheduler(),x,torch.tensor([0.]),x,h,lambda x,t:x,sigma=0.)

    def test_frozen_statistic_all32_tokens_float32(self):
        h=torch.arange(32*5).reshape(1,32,5).to(torch.bfloat16)
        expected=h[0].float().std(dim=0,unbiased=False).mean()
        torch.testing.assert_close(layer29_dispersion(h),expected,rtol=1e-6,atol=1e-6)
        with self.assertRaises(ValueError):
            layer29_dispersion(h[:,:16])
        with self.assertRaises(ValueError):
            layer29_dispersion(h,layer=28)


if __name__=='__main__':
    unittest.main()
