import neurockt as ckt
import unittest
import torch


class TestCommon(unittest.TestCase):
    def test_feedforward(self):
        D = 128
        x = torch.randn(2, 2, D)
        y = ckt.FeedForward(D)(x)
        self.assertEqual(y.size(), x.size())

    def test_projection(self):
        x = torch.randn(2, 2, 128)
        ys = ckt.Projection(*([128] * 4))(*([x] * 4))
        for y in ys:
            self.assertEqual(y.size(), x.size())

    def test_classifier(self):
        x = torch.randn(2, 128)
        y = ckt.ClassifierHead(128, 10)(x)
        self.assertEqual(y.size(), torch.zeros(2, 10).size())
