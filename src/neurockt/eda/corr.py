from glob import glob
import pandas as pd
import os
import matplotlib.pyplot as plt


class ModelCorrelation:
    def __init__(self, glob_ptr):
        paths = glob(glob_ptr)
        name = [os.path.basename(path) for path in paths]

        def get_df(i):
            col = f"{name[i]}"
            return pd.read_csv(paths[i], names=["ID", col])[col]

        dfs = [get_df(i) for i in range(len(paths))]
        self.df = pd.concat(dfs, axis=1)

    def remove(self, *cols):
        for col in cols:
            if col in self.df:
                del self.df[col]

    def plot(self, figsize=(7, 7), good_ones=[], bad_ones=[]):
        num_df = self.df.select_dtypes(['number'])
        labels = num_df.columns
        labels = [col.upper() if col in good_ones else col for col in labels]
        labels = ["__" if col in bad_ones else col for col in labels]

        fig = plt.figure(figsize=figsize)
        plt.matshow(self.df.corr(), fignum=fig.number)
        plt.xticks(range(num_df.shape[1]), labels, fontsize=14, rotation=45)
        plt.yticks(range(num_df.shape[1]), labels, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)
        return plt
