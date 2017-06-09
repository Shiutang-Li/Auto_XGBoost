# Auto_XGBoost
Tune XGBoost parameters automatically.

## Usage
from auto_parameter_tuner import auto_parameter_tuner
params, model = auto_parameter_tuner(DM_train, DM_valid, 
                                     objective  = 'reg:linear',
                                     eval_metric ='rmse',
                                     show_details = True,
                                     seed = 0)

## How it works

Instead of performing grid search (which is pretty time consuming), the algorithm tunes parameters by a few steps:

** Step 1 **

