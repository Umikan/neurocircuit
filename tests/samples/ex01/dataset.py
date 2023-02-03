import neurockt.module as module
import pandas as pd
from fastai.data.all import RegressionBlock, CategoryBlock, RandomSplitter, DataBlock


class SampleDataset(module.Dataset):
    predict_function = "x->y"

    def __init__(self, path):
        super().__init__()
        print(f"The path of this dataset is {path}.")
        points = list(range(600))
        self.df = pd.DataFrame({
            "x": [float(point) for point in points],
            "y": [str(int(point / 200)) for point in points]
        })


class SampleDatablock(module.DataBlock):
    def make_block(self, task_code, df, n_inp, get_items):
        self.classes = df['y'].unique()
        category = CategoryBlock(vocab=list(self.classes))
        if task_code == SampleDataset.predict_function:
            blocks = (RegressionBlock, category)
        else:
            assert False, "Check task_code is valid."

        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        block = DataBlock.from_columns(blocks=blocks,
                                       n_inp=n_inp,
                                       get_items=get_items,
                                       splitter=splitter)
        return block.datasets(df)

    def kwargs_to_pass(self) -> dict:
        return {"n_classes": len(self.classes)}
