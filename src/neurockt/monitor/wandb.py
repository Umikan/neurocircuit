import wandb
from .recorder import Recorder


class WandbRecorder(Recorder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.run = wandb.init(*args, **kwargs)
        self.__define_metrics()
        self.step_count = 1
        self.epoch_count = 1

    def train_step_name(self):
        return f"{self.train_group}/step"

    def test_step_name(self):
        return f"{self.epoch_group}/step"

    def get_run(self):
        return self.run

    def __define_metrics(self):
        train_group = self.train_step_name().split("/")[0]
        test_group = self.test_step_name().split("/")[0]
        wandb.define_metric(self.train_step_name())
        wandb.define_metric(f"{train_group}/*", step_metric=self.train_step_name())
        wandb.define_metric(self.test_step_name())
        wandb.define_metric(f"{test_group}/*", step_metric=self.test_step_name())

    def log(self, log_dict):
        super().log(log_dict)

        if self._train:
            log_dict[self.train_step_name()] = self.step_count
            self.step_count += 1
        else:
            log_dict[self.test_step_name()] = self.epoch_count
            self.epoch_count += 1
        wandb.log(log_dict)