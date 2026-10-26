"""Synthetic mathematical checks, explicitly separate from empirical results."""
import copy
import sys
import tempfile
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vendor/LHM'))
from research_assets import fuse_bundles, refine_graph, validate, load_bundle


def fixture():
    n = 4
    return {'schema': 'lhm.canonical.v1',
            'query_points': torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]),
            'betas': torch.zeros((1, 10)), 'metadata': {},
            'gs': [{'offset_xyz': torch.zeros(n, 3), 'opacity': torch.ones(n, 1)*.5,
                    'rotation': torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
                    'scaling': torch.ones(n, 3), 'shs': torch.arange(12.).reshape(n, 1, 3),
                    'use_rgb': True}]}


class FusionInvariants(unittest.TestCase):
    def test_quaternion_antipodes_do_not_cancel(self):
        a, b = fixture(), fixture()
        b['gs'][0]['rotation'] *= -1
        f = fuse_bundles([a, b])
        self.assertTrue(torch.allclose(f['gs'][0]['rotation'], a['gs'][0]['rotation']))

    def test_anchor_mismatch_rejected(self):
        a, b = fixture(), fixture()
        b['query_points'][0, 0, 0] += .001
        with self.assertRaises(ValueError):
            fuse_bundles([a, b])

    def test_shape_mismatch_rejected(self):
        a, b = fixture(), fixture()
        b['betas'][0, 0] += .001
        with self.assertRaises(ValueError):
            fuse_bundles([a, b])

    def test_single_source_not_called_fusion(self):
        with self.assertRaises(ValueError):
            fuse_bundles([fixture()])

    def test_scale_geometric_average(self):
        a, b = fixture(), fixture()
        b['gs'][0]['scaling'] *= 4
        f = fuse_bundles([a, b])
        self.assertTrue(torch.allclose(f['gs'][0]['scaling'], torch.ones(4, 3)*2))

    def test_high_confidence_graph_color_unchanged(self):
        f = fuse_bundles([fixture(), fixture()])
        g = refine_graph(f, amount=.25)
        self.assertTrue(torch.equal(f['gs'][0]['shs'], g['gs'][0]['shs']))

    def test_serialization_preserves_attributes(self):
        f = fixture()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'asset.pt'
            torch.save(f, p)
            g = load_bundle(p)
            self.assertTrue(torch.equal(f['gs'][0]['offset_xyz'], g['gs'][0]['offset_xyz']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
