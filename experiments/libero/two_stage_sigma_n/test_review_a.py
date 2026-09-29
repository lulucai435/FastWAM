from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from io_utils import sha
from sweep_a import run_phase_a_state
from review_a import verify_state
from test_sampler import Toy


class RawReviewTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);(root/'states').mkdir()
        previous=np.arange(224,dtype=np.float32).reshape(32,7)/200
        epsilon=torch.randn((1,32,7),generator=torch.Generator().manual_seed(42))[0].numpy()
        np.savez(root/'previous.npz',normalized_action=previous)
        np.savez(root/'current.npz',initial_noise=epsilon,normalized_proprio=np.zeros((1,8),dtype=np.float32))
        result=run_phase_a_state(Toy(),{'input_image':torch.zeros(1,3,16,16)},previous,epsilon,compiled=False)
        arrays={k:result[k] for k in ('executed_actions','full_positive_actions','errors',
            'initial_states','first_network_inputs','epsilon')}
        file=root/'states/state00000.npz';np.savez(file,**arrays)
        entry=dict(id=0,key=['synthetic',0,0,1],
            previous=dict(file=str(root/'previous.npz'),sha256=sha(root/'previous.npz')),
            current=dict(file=str(root/'current.npz'),sha256=sha(root/'current.npz')),
            prior_normalized_16x7=previous[16:].tolist())
        row=dict(id=0,key=entry['key'],file='states/state00000.npz',sha256=sha(file),
            current_sha256=entry['current']['sha256'],previous_sha256=entry['previous']['sha256'],
            traces=result['traces'],sigma_star_index=result['sigma_star_index'],observed_nfe=100)
        return row,entry,arrays

    def test_raw_state_roundtrip_and_independent_errors(self):
        with tempfile.TemporaryDirectory(prefix='sigma-raw-test-') as root:
            row,entry,arrays=self.fixture(root)
            np.testing.assert_array_equal(verify_state(root,row,entry),arrays['errors'])

    def test_detects_wrong_error_even_with_rehashed_bundle(self):
        with tempfile.TemporaryDirectory(prefix='sigma-raw-test-') as root:
            row,entry,arrays=self.fixture(root)
            arrays['errors'][0]+=1
            file=Path(root)/row['file'];np.savez(file,**arrays);row['sha256']=sha(file)
            with self.assertRaises(AssertionError):
                verify_state(root,row,entry)

    def test_detects_schedule_and_nfe_corruption(self):
        with tempfile.TemporaryDirectory(prefix='sigma-raw-test-') as root:
            row,entry,_=self.fixture(root)
            row['traces'][0]['timesteps'][0]=1000
            with self.assertRaises(AssertionError):
                verify_state(root,row,entry)
            row['traces'][0]['timesteps'][0]=100
            row['observed_nfe']=99
            with self.assertRaises(AssertionError):
                verify_state(root,row,entry)


if __name__=='__main__':
    unittest.main()
