from neurockt.module import Module
from ..models.sample import Net
import unittest


class TestModule(unittest.TestCase):
    def test_load_module(self):
        kwargs = {
            '_path': 'tests.models.classifier',
            '_cls': 'Net'
        }
        params = {
            "n_classes": 10,
            "dim": 512
        }
        kwargs.update(params)
        model = Module(**kwargs)()
        self.assertEqual(type(model), Net)
