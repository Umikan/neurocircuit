---

icon: device-desktop
order: -2

---


# Monitor

## Overview
Monitor is a powerful utility that enables us to log loss values, metrics in brief code.\
Monitor's output is like this:

```
	 valid_loss    acc           
	 0.0812        0.9753        
	 0.0562        0.9845        
	 0.0536        0.9804        
	 0.0427        0.9857  
```


Firstly, you have to import `Recorder` module. We also provide `WandbRecorder` for someone using W&B. 
```python
from neurockt.monitor import Recorder
# from neurockt.monitor.wandb import WandbRecorder

recorder = Recorder()
```

## Usage

### With Python`s decorator
Decorators such as `@recorder.track_scalar` and `@recorder.metric` can be used to track output values of decorated functions. The following code logs loss values when training or validating our model.

```python
def torch_mean(x):
    return torch.stack(x).mean().item()

class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    # Add here!
    @recorder.track_scalar("train_loss", torch_mean)
    @recorder.metric("valid_loss", torch_mean)
    def forward(self, inp, targ):
        return self.loss(inp, targ)
```

Then, call `recorder(training=True)` or `recorder(training=False)` using `with` statement.\
This makes sure that all values are logged at each iteration or epoch.


+++ 1. Training
```python
model.train()
for preds, labels in forward(train_dl, model):
    # At each training step, recorder aggregates loss values.
    with recorder(train=True):  
        (loss := criterion(preds, labels[0])).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```
+++ 2. Validation
```python
model.eval()
# At the end of validation, recorder aggregates loss values.
with recorder(train=False), torch.inference_mode(): 
    for preds, labels in forward(valid_dl, model):
        loss = criterion(preds, labels[0])
```
+++

### With RecorderHelper

In some situations where you can't use decorator syntax, we provide `RecorderHelper` to notify our recorder of the values you want to log.

```python
from neurockt.monitor import RecorderHelper
from torchmetrics.classification import MulticlassAccuracy

device = "cuda"
helper = RecorderHelper(recorder)
metric = helper.metric(torch_mean, acc=MulticlassAccuracy(10).to(device))
# At this point, an instance of MultiClassAccuracy is stored in metric["acc"]
```

## SaaS Integration
### WandbRecorder

You can use `WandbRecorder` for tracking your experiment.

```python
use_wandb = False
project = {
    'project': 'Your project name',
    'tags': [],  # write tags here
    'notes': 'a brief description of your experiment'
}
if use_wandb:
  from neurockt.monitor.wandb import WandbRecorder
  recorder = WandbRecorder(**project)
else:
  recorder = Recorder()
```

!!!
If you want to know the details about W&B, please visit [W&B documentations](https://docs.wandb.ai/quickstart).
!!!

## What is `torch_mean`?

`torch_mean` is an aggregation method of `torch.Tensor`. By default, you can omit `torch_mean` when output type is primitive one such as `int` or `float`.    