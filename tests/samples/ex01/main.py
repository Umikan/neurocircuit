import unittest

import torch.nn as nn
from .model import Net as Model
from .dataset import SampleDataset as Dataset
from .dataset import SampleDatablock as Block
from .trainer import TrainClassifier


class TestModule(unittest.TestCase):
    def test_ex01(self):
        path = "testpath"
        dataset = Dataset(path)
        block = Block(dataset, dataset.predict_function)
        params = {"dim": 512}
        model = block.fit_model(Model, **params)
        model.replace(nn.ReLU, nn.Sigmoid)
        trainer = TrainClassifier(model=model,
                                  datablock=block(),
                                  loss=nn.CrossEntropyLoss(),
                                  bs=64,
                                  cb_func=None)
        trainer.train(n_epoch=10)
