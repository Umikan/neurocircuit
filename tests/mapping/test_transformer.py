import neurockt as ckt
import unittest
import torch


class TestTransformer(unittest.TestCase):
    def test_qkv_attention(self):
        q = k = v = torch.randn(2, 10, 512)
        out = ckt.ScaleDotProductAttention(512)(q, k, v)
        self.assertEqual(q.size(), out.size())

    def test_transformer_block(self):
        x = torch.randn(2, 10, 256)
        y = ckt.TransformerBlock(256, 256, 256, 256)(x)
        self.assertEqual(y.size(), torch.zeros(2, 10, 256).size())
