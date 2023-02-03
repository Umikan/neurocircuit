from fastai.metrics import accuracy
from fastai.vision.all import Learner, ToTensor


class TrainClassifier:
    def __init__(self, model, datablock, loss, bs, cb_func):
        dls = datablock.dataloaders(bs=bs, after_item=[ToTensor], after_batch=[])
        self.trainer = Learner(dls, model, metrics=accuracy, loss_func=loss, cbs=cb_func)

    def train(self, n_epoch=30):
        lr = self.trainer.lr_find()
        self.trainer.fit_one_cycle(n_epoch=n_epoch, lr_max=lr)
