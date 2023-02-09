from torch.utils.data import Dataset


class Merge(Dataset):
    def __init__(self, indices, *datasets):
        self.datasets = datasets
        self.length = len(self.datasets[0])
        for dset in self.datasets[1:]:
            assert self.length == len(dset), "To merge datasets, the count of samples has to be the same."

        self.indices = list(range((self.length))) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return tuple(dset[idx] for dset in self.datasets)