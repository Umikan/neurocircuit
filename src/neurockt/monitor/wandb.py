import wandb
from .recorder import Recorder


class WandbRecorder(Recorder):
    def log(self, log_dict):
        wandb.log(log_dict)