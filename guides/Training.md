---
icon: graph
order: -3

---

# Training

Here is several utilities to write your training code. 
***

### neurockt.eda.KFold
KFold strategy reduces your variance of accuracies so that it comes easier to choose the best model.\
KFold is an iterator which returns train/valid indices.

Variable        |Type   | Details { class="compact" }
:--             |:--    |:-- 
df              |int    | Pandas' Dataframe
n_splits        |int    | A number of folds
y_col           |str     | If specified, it applys scikit-learn's `StratifiedKFold` to your dataset.



[!ref target="blank" text="Source Code"](https://github.com/Umikan/neurocircuit/blob/main/src/neurockt/eda/cv.py)

### neurockt.monitor.EarlyStopping

EarlyStopping is useful when you want to finishing training at the middle of epochs.\
If a better model is found, it calls hook functions that were registed by `EarlyStopping.add_hook()`.

```python
early_stopping = EarlyStopping(patience=5)
exceeded = early_stopping(acc)  # True if exceeding maximum count of patience 
```

Variable        |Type   | Details { class="compact" }
:--             |:--    |:-- 
patience        |int    | Maximum count of patience 
mode            |Category| If `mode="max/min"`, The count will be reset when current value is higher/smaller than the previous best value.

[!ref target="blank" text="Source Code"](https://github.com/Umikan/neurocircuit/blob/main/src/neurockt/monitor/early_stopping.py)

***
### neurockt.torch.forward
This function automatically deals with your data stream and returns predictions and labels.\
It inspects the number of model inputs so that your model feeds only inputs.

```python
for preds, labels in forward(train_dl, model):
    # write your training step
```

[!ref target="blank" text="Source Code"](https://github.com/Umikan/neurocircuit/blob/main/src/neurockt/torch/forward.py)

***
### neurockt.torch.Stack

This function stacks what it feeds. It is useful when calculating over all batches.

```python
stack = Stack()
for preds, labels in forward(valid_dl, model):
    loss = criterion(preds, labels[0])
    stack.add(preds=preds, labels=labels[0])
acc = metric["acc"](stack('preds'), stack('labels'))
```

[!ref target="blank" text="Source Code"](https://github.com/Umikan/neurocircuit/blob/main/src/neurockt/torch/batch.py)

