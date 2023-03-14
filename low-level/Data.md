---

icon: triangle-right
order: -1

---

# Data

***
### Multiclass
Create multi-class targets.

Variable        |Type   | Details { class="compact" }
:--             |:--    |:-- 
df              |Dataframe | A column of Dataframe

***
### Multilabel
Create multi-label targets.

Variable        |Type   | Details { class="compact" }
:--             |:--    |:-- 
df              |Dataframe | A column of Dataframe

***
### Image
Create image inputs.

Variable        |Type   | Details { class="compact" }
:--             |:--    |:-- 
df              |Dataframe | A column of Dataframe
transform       |Any       | Image transformation

!!!
Supported types are `str` (path) and `torch.Tensor` (raw data).
!!!

***
### Merge
Merge multple datasets. Use when all inputs/outputs are individually defined.

Variable        |Type   | Details { class="compact" }
:--             |:--    |:-- 
indices         |Dataframe | Indices of dataset samples
*datasets       |list       | List of datasets
