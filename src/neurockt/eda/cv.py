from abc import abstractmethod
from datetime import datetime
from sklearn import model_selection
import statistics as stats
import numpy as np


class KFold:
    def __init__(self, df, n_splits, y_col=None):
        self.df = df
        self.n_splits = n_splits
        if y_col:
            kf = model_selection.StratifiedKFold(n_splits=n_splits)
            self.indices = kf.split(df, df[y_col])
        else:
            kf = model_selection.KFold(n_splits=n_splits)
            self.indices = kf.split(df)

        self.indices = list(self.indices)
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == len(self):
            raise StopIteration()

        train, test = self.indices[self._i]
        self._i += 1
        return train, test

    def __len__(self):
        return self.n_splits


class ChooseBestModel:
    def __init__(self, mode="max"):
        self.model_id = datetime.now().strftime('%Y%m%d%H%M%S')
        self.best_valid = []
        self.mode = mode
        assert mode == "max" or mode == "min"

    def get_name(self, i):
        return f"{self.model_id}_{i}"

    @abstractmethod
    def load_model(self, i):
        raise NotImplementedError

    @abstractmethod
    def learn(self, fold, save_name) -> float:
        raise NotImplementedError

    def __call__(self, folds):
        for i, fold in enumerate(folds):
            v = self.learn(fold, self.get_name(i))
            self.best_valid.append(v)

    def get_best_model(self):
        print("CV mean: {}".format(stats.mean(self.best_valid)))
        print("CV variance: {}".format(stats.variance(self.best_valid)))

        if self.mode == "max":
            i = np.argmax(self.best_valid)
        elif self.mode == "min":
            i = np.argmin(self.best_valid)

        print(f"Best model: {i}")
        return self.load_model(i)
