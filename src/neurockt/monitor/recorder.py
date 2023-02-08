from statistics import mean


class Recorder:
    def __init__(self, name):
        self.__steps, self.__metrics = {}, {}
        self.name = name 
        self._train = True
        self.accum_funcs = {}
        
    def train(self, flag=True):
        self._train = flag

    def __track__(self, storage, col_name):
        storage[col_name] = []
        def call_func(func):
            def pass_args(*args, **kwargs):
                value = func(*args, **kwargs)
                storage[col_name].append(value)
                return value
            return pass_args
        return call_func

    def track_scalar(self, col_name, accum_func=mean):
        col_name = f"step/{col_name}"
        self.accum_funcs[col_name] = accum_func
        return self.__track__(self.__steps, col_name)
                
    def metric(self, col_name, accum_func=mean):
        col_name = f"metrics/{col_name}"
        self.accum_funcs[col_name] = accum_func
        return self.__track__(self.__metrics, col_name)

    def next_step(self):
        if self._train:
            self.__advance__(self.__steps)
        self.clear()

    def next_epoch(self):
        if not self._train:
            self.__advance__(self.__metrics)
        self.clear()
            
    def clear(self):
        for key in self.__steps.keys():
            self.__steps[key] = []
        for key in self.__metrics.keys():
            self.__metrics[key] = []

    def __advance__(self, storage):
        log_dict = {}
        for key in storage.keys():
            value = storage[key]
            if type(value) == list:
                value = self.accum_funcs[key](value)
            log_dict[key] = value
        self.log(log_dict)

    def log(self, log_dict):
        print(log_dict)
