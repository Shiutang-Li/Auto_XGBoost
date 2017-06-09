# Auto_XGBoost
Tune XGBoost parameters automatically. User needs to prepare a training set and a validation set to get it working.

## Usage
In your python script:
```
from auto_parameter_tuner import auto_parameter_tuner
params, model = auto_parameter_tuner(DM_train, DM_valid, 
                                     objective  = 'reg:linear',
                                     eval_metric ='rmse',
                                     show_details = True,
                                     seed = 0)
```
## How it works

Instead of performing grid search (which is pretty time consuming), the algorithm tunes parameters by a few steps:

**Step 1.** Tune eta + max_depth simulaneously. eta = 0.03, 0.05, 0.1, max_depth = 3, 4, 5, 6, 7. (Do 15 searches) If the best max_depth is 7, it will continue to search max_depth = 8, 9, ... until eval_metric cannot get better.

**Step 2.** Fix the best eta + max_depth values obtained from Step 1, tune lambda, gamma simulaneously. lambda = 0.5, 1, 2, 4, 8, gamma = 0, 0.5, 1. (Do 15 searches) 

**Step 3.** Tune colsample_bytree, subsample simulaneously. values for both parameters are picked from 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1 (Do 49 searches) 

**Step 4.** Fine tune lambda, gamma, colsample_bytree, subsample.
