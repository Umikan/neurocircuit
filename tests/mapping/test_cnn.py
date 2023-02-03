import neurockt as ckt
import unittest
import torch


class TestMapping(unittest.TestCase):
    def test_gap(self):
        x = torch.randn(2, 3, 224, 224)
        y = ckt.GlobalAveragePooling()(x)
        self.assertEqual(y.size(), torch.zeros(2, 3).size())
