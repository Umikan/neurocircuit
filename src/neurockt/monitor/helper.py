from .recorder import Recorder


class RecorderHelper:
    def __init__(self, recorder: Recorder):
        self.recorder = recorder

    def metric(self, accum_func, **metrics):
        def apply(col_name, func):
            return self.recorder.metric(col_name, accum_func)(func)
        return {k: apply(k, v) for k, v in metrics.items()}
    
    def track_scalar(self, accum_func, **funcs):
        def apply(col_name, func):
            return self.recorder.track_scalar(col_name, accum_func)(func)
        return {k: apply(k, v) for k, v in funcs.items()}
    