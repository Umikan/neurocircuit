---
icon: columns
order: -1
title: DataPipe
---

# DataPipe
## Overview
In machine learning, dataset management is important since we take numerous tasks into consideration with single dataset. Let's see our entire example.

```python
from neurockt.data import DataPipe, Multiclass, Multilabel, Image

image = ("Image", Image)
target = ("target", Multiclass)

# df is Pandas' Dataframe
pipe = (
    DataPipe(df)
    .X(*image, train_transforms)
    .Y(*target).arg("num_classes", Multiclass.num_classes)
    .bunch("train")
    .X(*image, common_transforms)
    .Y(*target)
    .bunch("valid")
)

dls = pipe.select(bunch=("train", "valid"), idx=(train_idx, valid_idx))
train_dl, valid_dl = dls(bs=64, shuffle=True, num_workers=8, drop_last=False)
```

Our DataPipe works with Pandas' Dataframe, since it's easy to handle each task which differs from other tasks in terms of input/output formats.

All you have to do with DataPipe is below:

* Specify inputs and targets of your Dataframe
* Apply settings (such as transformations) to training/inference phase 
* Store your parameters (such as category size) to DataPipe
* Create and select bunches of DataPipe

## Usage

!!!
All instance methods returns an instance of itself so that you can chain those methods.
!!!

***
### `DataPipe()`
Create a pipeline that can see the given dataframe.

Variable        |Type   | Details { class="compact" }
:--             |:--    |:-- 
df              |Dataframe | An object of Dataframe

***
### `DataPipe.X()`
Add an information of model's input. 

Variable        |Type   | Details { class="compact" }
:--             |:--    |:-- 
column          |str    | The column name of dataframe 
data_type       |Category| The column type of dataframe 
*args, **kwargs |       | arguments of Category object 

***
### `DataPipe.Y()`
Add an information of model's output. 

Variable        |Type   | Details { class="compact" }
:--             |:--    |:-- 
column          |str    | The column name of dataframe 
data_type       |Category| The column type of Dataframe 
*args, **kwargs |       | Arguments of Category object 

***
### `DataPipe.args()`
Retrieve a parameter from the Category object which was previously added. 

Variable        |Type               | Details { class="compact" }
:--             |:--                |:-- 
name            |str                | A name of constant variable 
method          |Category's method  | A Category's method to retrieve a parameter from

***
### `DataPipe.bunch()`
Create a bunch of the dataset in this pipeline.

Variable        |Type               | Details { class="compact" }
:--             |:--                |:-- 
name            |str                | A bunch's name

***
### `DataPipe.select()`
Select multiple bunches with the given indices.

Variable        |Type               | Details { class="compact" }
:--             |:--                |:-- 
bunch            |tuple                | Tuple of bunch name 
idx              |tuple                | Tuple of indices

***
### `DataPipe.__call__()`
Get `Dataloader` instances from the bunches you selected.

Variable        |Type               | Details { class="compact" }
:--             |:--                |:-- 
bs              |int                | Batch size 
shuffle         |bool                | Shuffle the dataset
num_workers     |int                | A number of workers
drop_last       |bool               | Drop last mini-batch.
pin_memory      |bool               | Pin memory

***
### `DataPipe.get_args()`
Get arguments stored in DataPipe.
