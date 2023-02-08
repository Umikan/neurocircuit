# DO NOT USE this module for concurrent process
import contextvars


ctx_vars = {}


def autofill(cls):
    str_cls = str(cls)
    name_args = cls.__init__.__code__.co_varnames

    def filter_args(kwargs):
        return {k: v for k, v in kwargs.items() if k in name_args}

    def constructor(self, *args, **kwargs):
        params = {}
        if str_cls in list(ctx_vars.keys()):
            params = ctx_vars[str_cls].get().copy()
        params = filter_args(params)
        return cls.__old_init__(self, *args, **params, **kwargs)

    cls.__old_init__ = cls.__init__
    cls.__init__ = constructor 
    return cls


def pass_kwargs(*sublayers):
    def get_class(cls):
        def constructor(self, *args, **kwargs):
            for sublayer in sublayers:
                sublayer = str(sublayer)
                ctx_vars[sublayer] = contextvars.ContextVar(sublayer)
                ctx_vars[sublayer].set(kwargs)
            return cls.__old_init__(self, *args, **kwargs)
        cls.__old_init__ = cls.__init__
        cls.__init__ = constructor
        return cls
    return get_class
