"""Run the actual model's initialization block on CPU without loading weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from definitions import initialize

MODEL = Path(__file__).resolve().parents[3] / 'src/fastwam/models/wan22/fastwam.py'


class ModelInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(MODEL.read_text())
        method, = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'infer_action']
        def assignment(name):
            return next(i for i, node in enumerate(method.body)
                        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == name)
        cls.code = compile(ast.Module(body=method.body[assignment('action_shape'):assignment('initial_flow_state')+1],
                                     type_ignores=[]), str(MODEL), 'exec')

    def run_initialization(self, **overrides):
        scope = dict(torch=torch, self=SimpleNamespace(action_expert=SimpleNamespace(action_dim=7)),
                     action_horizon=32, start_sigma=1., initial_noise=None, rand_device='cpu',
                     seed=42, action_prior=None, action_prior_mask=None)
        scope.update(overrides)
        exec(self.code, scope)
        return scope['initial_flow_state'][0]

    def test_original_fresh_seed_is_preserved(self):
        expected = torch.randn((1,32,7), generator=torch.Generator().manual_seed(42))[0]
        torch.testing.assert_close(self.run_initialization(), expected, rtol=0, atol=0)

    def test_actual_masked_initialization_all_sigmas(self):
        g = torch.Generator().manual_seed(42)
        epsilon, previous = torch.randn((32,7), generator=g), torch.randn((32,7), generator=g)
        prior = torch.zeros_like(previous); prior[:16] = previous[16:]
        mask = torch.arange(32) < 16
        for sigma in np.arange(1,11)/10:
            result = self.run_initialization(initial_noise=epsilon, action_prior=prior,
                                             action_prior_mask=mask, start_sigma=float(sigma))
            np.testing.assert_array_equal(result.numpy(), initialize(previous.numpy(), epsilon.numpy(), sigma))
            torch.testing.assert_close(result[16:], epsilon[16:], rtol=0, atol=0)

    def test_unmasked_legacy_formula_and_mask_validation(self):
        prior = torch.ones(32,7); epsilon = torch.full_like(prior,2.)
        result = self.run_initialization(initial_noise=epsilon, action_prior=prior, start_sigma=.4)
        torch.testing.assert_close(result, .6*prior+.4*epsilon, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.run_initialization(action_prior=prior, action_prior_mask=torch.ones(32))
        with self.assertRaises(ValueError):
            self.run_initialization(action_prior_mask=torch.ones(32,dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
