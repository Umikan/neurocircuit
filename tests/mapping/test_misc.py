import neurockt as ckt
import unittest
import torch


class TestMisc(unittest.TestCase):
    def test_image_to_token(self):
        x = torch.randn(2, 3, 224, 224)
        model = ckt.ImageToToken()
        y = model(x)
        z = model.reverse(y)
        self.assertEqual(x.size(), z.size())
