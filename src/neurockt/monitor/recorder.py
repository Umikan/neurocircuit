from statistics import mean


class Recorder:
    train_group = "Training"
    epoch_group = "Metric"
 
    def __init__(self):
        self.__steps, self.__metrics = {}, {} 
        self._train = True
        self.accum_funcs = {}

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
        col_name = f"{self.train_group}/{col_name}"
        self.accum_funcs[col_name] = accum_func
        return self.__track__(self.__steps, col_name)
                
    def metric(self, col_name, accum_func=mean):
        col_name = f"{self.epoch_group}/{col_name}"
        self.accum_funcs[col_name] = accum_func
        return self.__track__(self.__metrics, col_name)

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
        self.clear()

    def log(self, log_dict):
        if not self._train:
            self.__print_log(log_dict, not hasattr(self, "_first"))
            self._first = False

    def __print_log(self, log_dict, is_first):
        def col_format(col):
            align = lambda x: "{:14s}".format(x)
            if type(col) == str:
                return align(col)
            return align("{:.6s}".format(str(col)))
            
        columns, values = [], []
        for k, v in log_dict.items():
            k = k.split("/")[-1]
            columns.append(col_format(k))
            values.append(col_format(v))

        if is_first:
            print("\n\t", "".join(columns))
        print("\t", "".join(values))

    def __enter__(self):
        self.clear()

    def __call__(self, train: bool):
        self._train = train
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self._train:
            self.__advance__(self.__steps)
        else:
            self.__advance__(self.__metrics)