import pandas as pd
from importlib import import_module
from abc import abstractmethod
from logging import getLogger
logger = getLogger(__name__)


class Module:
    def __init__(self, _path, _cls, **params):
        self.params = params
        self.module = getattr(import_module(_path), _cls)
        logger.info(f'Successfully loaded a module: {_path}.{_cls}')

    def __call__(self, *args, **kwargs):
        self.params.update(kwargs)
        return self.module(*args, **self.params)


class Dataset:
    def __init__(self):
        pass

    def for_task(self, task_code):
        assert isinstance(
            self.df, pd.DataFrame), "No pd.DataFrame is set. Make sure that you set self.df in __init__() function."

        inputs, labels = task_code.split("->")
        inputs = inputs.split(",")
        labels = labels.split(",")
        n_inputs = len(inputs)

        for key in (inputs + labels):
            assert key in self.df, f"No Values of (key: {key}) is set. Make sure that this task is valid."

        def get_items(row):
            return tuple([getattr(row, key) for key in (inputs + labels)])

        return n_inputs, get_items

    def dataframe(self):
        return self.df

    def at(self, k):
        return self.df.iloc[k]


class DataBlock:
    def __init__(self, dataset, task_code, *args):
        df = dataset.dataframe()
        n_inp, get_items = dataset.for_task(task_code)
        self.block = self.make_block(task_code, df, n_inp, get_items, *args)

    def __call__(self):
        assert self.block is not None, "Make sure that self.block is set in DataBlock"
        return self.block

    def fit_model(self, model_cls, *args, **kwargs):
        params = {}
        params.update(self.kwargs_to_pass())
        params.update(kwargs)
        return model_cls(*args, **params)

    @abstractmethod
    def make_block(self, task_code, df, n_inp, get_items, *args):
        raise NotImplementedError

    @abstractmethod
    def kwargs_to_pass(self) -> dict:
        raise NotImplementedError
