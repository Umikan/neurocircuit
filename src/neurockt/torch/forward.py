from inspect import signature
from tqdm import tqdm


def forward(dl, model):
    sig = signature(model.forward)
    n_inp = len(sig.parameters)
    device = next(model.parameters()).device

    for data in tqdm(dl, leave=False):
        data =  [d.to(device) for d in data]
        inputs, targets = data[:n_inp], data[n_inp:] 
        preds = model(*inputs)
        yield preds, targets
