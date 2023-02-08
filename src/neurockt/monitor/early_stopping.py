from logging import getLogger
logger = getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=10, mode="max"):
        self.patience = patience
        self.n_epoch = 0
        self.mode = mode

        self.count = 0
        self.best_value = None
        self.prev = None
        
    def __call__(self, current):
        self.n_epoch += 1
        if self.prev is None:
            self.prev = float("-inf") if self.mode == "max" else float("inf")
            self.best_value = self.prev

        less_acc = self.mode == "max" and self.prev > current
        more_err = self.mode == "min" and self.prev < current
        if less_acc or more_err:
            self.count += 1
        else:
            self.prev = current
            self.count = 0
            logger.info(f"Better model found at epoch {self.n_epoch}: {current}")

        has_exceeded = self.count > self.patience
        return has_exceeded