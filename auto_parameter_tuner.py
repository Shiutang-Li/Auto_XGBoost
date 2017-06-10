#!/usr/bin/python

# coding=utf8
# description     : Tune XGBoost parameters automatically
# author          : Shiu-Tang Li
# last update     : 06/09/2017
# version         : 0.1
# python_version  : 3.5.2

import pandas as pd
import numpy as np
import xgboost as xgb
from copy import deepcopy

def auto_parameter_tuner(DM_train, DM_valid, 
                         objective  = 'reg:linear',
                         eval_metric ='rmse',
                         show_details = True,
                         seed = 0):

### GOAL: tune XGBoost parameters:
#         eta, max_depth, lambda, gamma, colsample_bytree, subsample


### PARAMETERS:
# DM_train       training set,   converted to xgboost DM_matrix 
# DM_train       validation set, converted to xgboost DM_matrix 
# objective      regression: 'reg:linear'
#                binary classification: 'binary:logistic'
#                ... etc
# eval_metric    'rmse', 'mae', 'logloss', ... etc
# show_details   print statistics when training the model 
# seed           random seed


### OUTPUT:      
# best_param     dictionary recording the parameters used to get best model
# final_model    the model with the best score


### EXAMPLE:
# Suppose train, valid, y_train, y_valid, test are ready (in the form of pandas dataframes)
#
# DM_train = xgb.DMatrix(data=train, label=y_train)
# DM_valid = xgb.DMatrix(data=valid, label=y_valid)
# DM_test  = xgb.DMatrix(data=valid, label=y_valid)
#
# param, model = auto_parameter_tuner(DM_train = DM_train, 
#                                     DM_valid = DM_valid)
# DM_test  = xgb.DMatrix(data=test)
# y_test   = model.predict(DM_test) 
    
    num_boost_round = 10000
    early_stopping_rounds = 50
    watchlist = [(DM_train, 'train'), (DM_valid, 'valid')]
    y_train = DM_train.get_label()
    
    params = {'base_score':np.mean(y_train),
              'seed': seed,
              'eta': 0.1,
              'colsample_bytree': 1,
              'max_depth': 3,
              'subsample': 1,
              'lambda': 1,
              'gamma':0,
              'objective'   : objective,
              'eval_metric' : eval_metric
             } 

    best_score = np.Inf
    best_param = None
    best_model = None

    if show_details: 
        print('Start with: gamma = 0, lambda = 1, colsubsample = 1, subsample = 1.')
        print('')

    # Tune max depth and rough tune eta
    for eta in [0.1, 0.05, 0.03]:
        for max_depth in [3,4,5,6,7]:
            params['max_depth'] = max_depth
            params['eta'] = eta
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details: 
                print("   eta: " + str(eta).ljust(4) + ", max depth: " + str(max_depth) +
                      ", score: " +  str(score))
    
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)

    if best_param['max_depth'] == 7: 
        max_depth = 8
        flag = 0
        while (flag == 0) and (max_depth <= 15):
            params['max_depth'] = max_depth
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details: 
                print("   max depth: " + str(max_depth) + ", score: " +  str(score))

            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                max_depth += 1
            else:
                flag = 1
            
    if show_details: 
        print('')
        print("max_depth tuned. It's "+ str(best_param['max_depth']) + ".")
        print("eta tuned. It's "+ str(best_param['eta']) + ".")
        print('')


# Rough tune: gamma, lambda 

    for param_gamma in [0, 0.3, 0.7, 1]:
        for param_lambda in [0.5, 1, 2, 4, 8]:
            params = deepcopy(best_param)
            params['gamma']  =param_gamma
            params['lambda'] =param_lambda
                
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details: 
                print(("   gamma: " + str(param_gamma)  + ",").ljust(15) + 
                      ("lambda: " + str(param_lambda) +  ",").ljust(13) +
                       "score: " +  str(score))
    
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)

    if show_details:
        print('')
        print("gamma temporarily tuned.  It's "+ str(best_param['gamma']) + ".")
        print("lambda temporarily tuned. It's "+ str(best_param['lambda']) + ".")
        print('')


# Rough tune: colsample_bytree, subsample 

    for colsample_bytree in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]:
        for subsample in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]:
            params = deepcopy(best_param)
            params['colsample_bytree'] =colsample_bytree
            params['subsample'] =subsample
                
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   colsample: " + str(colsample_bytree)  + ",").ljust(20) + 
                      ("subsample: " + str(subsample) +  ",").ljust(17) +
                       "score: " +  str(score))
        
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)

    if show_details:
        print('')
        print("colsample_bytree temporarily tuned. It's "+ str(best_param['colsample_bytree']) + ".")
        print("subsample temporarily tuned.        It's "+ str(best_param['subsample']) + ".")
        print('')

### Fine tune gamma, lambda, colsample_bytree, subsample 
# fine tune gamma

    params = deepcopy(best_param)
    if best_param['gamma'] == 0: 
        flag = 0
        params['gamma'] = 0.1
        while flag == 0:
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   gamma: " + str(params['gamma'])  + ",").ljust(15) 
                        + "score: " +  str(score))

            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['gamma'] += 0.1
            else:
                flag = 1
    else:
        flag1 = 0
        flag2 = 0
        starting_gamma = best_param['gamma']
        params['gamma'] -= 0.1
    
        while (flag1 == 0) and (params['gamma'] > 0):
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   gamma: " + str(params['gamma'])  + ",").ljust(15) 
                        + "score: " +  str(score))

            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['gamma'] -= 0.1
            else:
                flag1 = 1
            
        params['gamma'] = starting_gamma + 0.1   
        while flag2 == 0:
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   gamma: " + str(params['gamma'])  + ",").ljust(15) 
                        + "score: " +  str(score))

            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['gamma'] += 0.1
            else:
                flag2 = 1
                
    if show_details:
        print('')
        print("gamma tuned. It's "+ str(best_param['gamma']) + ".")
        print('')
    
    
# fine tune lambda
    
    params = deepcopy(best_param)
    if (best_param['lambda'] == 0.5) or (best_param['lambda'] == 1): 
        flag1 = 0
        flag2 = 0
        starting_lambda = best_param['lambda']
        params['lambda'] -= 0.1
    
        while (flag1 == 0) and (params['lambda'] > 0):
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   lambda: " + str(params['lambda'])  + ",").ljust(17) 
                         + "score: " +  str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['lambda'] -= 0.1
            else:
                flag1 = 1
            
        params['lambda'] = starting_lambda + 0.1   
        while flag2 == 0:
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   lambda: " + str(params['lambda'])  + ",").ljust(17) 
                         + "score: " +  str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['lambda'] += 0.1
            else:
                flag2 = 1
    
    else: 
        flag1 = 0
        flag2 = 0
        starting_lambda = best_param['lambda']
        params['lambda'] -= 0.2
    
        while (flag1 == 0) and (params['lambda'] > 0):
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   lambda: " + str(params['lambda'])  + ",").ljust(17) 
                         + "score: " +  str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['lambda'] -= 0.2
            else:
                flag1 = 1
            
        params['lambda'] = starting_lambda + 0.2   
        while flag2 == 0:
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   lambda: " + str(params['lambda'])  + ",").ljust(17) 
                         + "score: " + str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['lambda'] += 0.2
            else:
                flag2 = 1
                
    if show_details:
        print('')
        print("lambda tuned. It's "+ str(best_param['lambda']) + ".")
        print('')

# fine tune col_subsample

    params = deepcopy(best_param)
    if (best_param['colsample_bytree'] == 1): 
        flag1 = 0
        starting_colsample = best_param['colsample_bytree']
        params['colsample_bytree'] -= 0.01
    
        while flag1 == 0:
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   colsample: " + str(params['colsample_bytree'])  + ",").ljust(20) 
                       + "score: " +  str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['colsample_bytree'] -= 0.01
            else:
                flag1 = 1

    else: 
        flag1 = 0
        flag2 = 0
        starting_colsample = best_param['colsample_bytree']
        params['colsample_bytree'] -= 0.01
    
        while flag1 == 0:
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   colsample: " + str(params['colsample_bytree'])  + ",").ljust(20) 
                      + "score: " +  str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['colsample_bytree'] -= 0.01
            else:
                flag1 = 1
            
        params['colsample_bytree'] = starting_colsample  + 0.01   
        while (flag2 == 0)  and (params['colsample_bytree'] <1):
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   colsample: " + str(params['colsample_bytree'])  + ",").ljust(20) 
                      + "score: " +  str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['colsample_bytree'] += 0.01
            else:
                flag2 = 1

    if show_details:
        print('')
        print("colsample_bytree tuned. It's "+ str(best_param['colsample_bytree']) + ".")
        print('')


# fine tune subsample

    params = deepcopy(best_param)
    if (best_param['subsample'] == 1): 
        flag1 = 0
        starting_subsample = best_param['subsample']
        params['subsample'] -= 0.01
    
        while flag1 == 0:
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   subsample: " + str(params['subsample'])  + ",").ljust(20) 
                  + "score: " +  str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['subsample'] -= 0.01
            else:
                flag1 = 1

    else: 
        flag1 = 0
        flag2 = 0
        starting_subsample = best_param['subsample']
        params['subsample'] -= 0.01
    
        while flag1 == 0:
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                            )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   subsample: " + str(params['subsample'])  + ",").ljust(20) 
                      + "score: " +  str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['subsample'] -= 0.01
            else:
                flag1 = 1
            
        params['subsample'] = starting_subsample  + 0.01   
        while (flag2 == 0)  and (params['subsample'] <1):
            model   = xgb.train(params=params,  
                                dtrain=DM_train, 
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=False
                                )
            score = float(model.attributes()['best_score'])
            if show_details:
                print(("   subsample: " + str(params['subsample'])  + ",").ljust(20) 
                      + "score: " +  str(score))
        
            if score < best_score:
                best_score = score
                best_param = deepcopy(params)
                best_model = deepcopy(model)
                params['subsample'] += 0.01
            else:
                flag2 = 1
                
    if show_details:
        print('')
        print("subsample tuned. It's "+ str(best_param['subsample']) + ".")
        print('')
        
# tuning best round       

    final_model = xgb.train(params=best_param,  
                            dtrain=DM_train, 
                            num_boost_round= int(best_model.attributes()['best_iteration']),
                            verbose_eval=False
                            )
      
    return best_param, final_model
