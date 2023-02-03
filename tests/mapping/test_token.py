import neurockt as ckt
import unittest
import torch


class TestToken(unittest.TestCase):
    def test_multicategory_embedder(self):
        N, D = 3, 512
        x = torch.eye(N)
        y, mask = ckt.MultiCategoryEmbedder(N, D)(x)
        self.assertEqual(y.size(), torch.zeros(N, 1, D).size())
        self.assertEqual(mask.size(), torch.zeros(N, 1).size())
