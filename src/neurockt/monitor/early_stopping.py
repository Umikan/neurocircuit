from logging import getLogger
logger = getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=10, mode="max"):
        self.patience = patience
        self.n_epoch = 0
        self.mode = mode
        self.count = 0
        self.prev = None
        self.hooks = []

    def add_hook(self, func):
        self.hooks.append(func)

    def __call__(self, current):
        self.n_epoch += 1
        if self.prev is None:
            self.prev = float("-inf") if self.mode == "max" else float("inf")

        more_acc = self.mode == "max" and self.prev <= current
        less_err = self.mode == "min" and self.prev >= current
        if more_acc or less_err:
            self.prev = current
            self.count = 0
            logger.info(f"Better model found at epoch {self.n_epoch}: {current}")
            for hook in self.hooks:
                hook()
        else:
            self.count += 1

        has_exceeded = self.count > self.patience
        if has_exceeded:
            logger.info("EarlyStopping: Exceeded the maximum count of patience")

        return has_exceeded
