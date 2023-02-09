from torch.utils.data import DataLoader
from .merge import Merge


class DataPipe:
    def __init__(self, df):
        self.df = df
        self.dsets = {}
        self.n_inp = {}
        self.args = {}
        self.__init_storage()

    def __init_storage(self):
        self.inputs = []
        self.targets = []    

    def bunch(self, name):
        self.dsets[name] = self.inputs + self.targets
        self.n_inp = len(self.inputs)
        self.__init_storage()
        return self

    def get_args(self):
        return self.args
        
    def select(self, bunch: tuple, idx: tuple):
        self.cursor = []
        for name, i in zip(bunch, idx):
            self.cursor.append(Merge(i, *self.dsets[name]))
            
        return self
    
    def __call__(self, bs, shuffle, num_workers, drop_last, pin_memory=True):
        def dataloader(dataset):
            return DataLoader(dataset,
                                  batch_size=bs,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=drop_last,
                                  pin_memory=pin_memory
                                )
        return [dataloader(dset) for dset in self.cursor]

    def X(self, column, data_type, *args, **kwargs):
        dset = data_type(self.df[column], *args, **kwargs)
        self.inputs.append(dset)
        self.last = self.inputs[-1]
        return self
    
    def Y(self, column, data_type, *args, **kwargs):
        dset = data_type(self.df[column], *args, **kwargs)
        self.targets.append(dset)
        self.last = self.targets[-1]
        return self  

    def arg(self, name, method):        
        self.args[name] = method(self.last)
        return self
