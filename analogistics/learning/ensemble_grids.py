from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
# %% CLASSIFICATION MODEL PARAMETERS

tuned_param_rf= [{'n_estimators':[10,50,100,200],
                  #'criterion':['gini','entropy'],
                  'max_features':['auto','sqrt','log2'],
                  'max_depth': range(1,20)                 
                  }]


tuned_param_ab = [{'n_estimators':[1, 10, 50, 100]               
                  }]


tuned_param_gb =  [{'loss':['deviance','exponential'],
                    'learning_rate':[1e-2, 1e-1, 1, 10],
                    'n_estimators':[1,2,10,50,100]
                  
                              
                  }]

tuned_param_bt = [{'n_estimators':[1, 10, 50, 100]               
                         }]
                                
models_classification = {
                            
                            
                            'random forest': {
                                   'estimator': RandomForestClassifier(), 
                                   'param': tuned_param_rf,
                            },
                            
                            'adaboost': {
                                   'estimator': AdaBoostClassifier(), 
                                   'param': tuned_param_ab,
                            },
                            
                            'gradient boosting': {
                                   'estimator': GradientBoostingClassifier(), 
                                   'param': tuned_param_gb,
                            },
                            
                            'bagging tree': {
                                   'estimator': BaggingClassifier(), 
                                   'param': tuned_param_bt,
                            },
                        }

# %% REGRESSION MODEL PARAMETERS

tuned_param_rf_regr= [{'n_estimators':[10,50,100,200],
                            #'criterion':['gini','entropy'],
                          'max_features':['auto','sqrt','log2'],
                          'max_depth': range(1,20)                 
                          }]

tuned_param_ab_regr  = [{'n_estimators':[1, 10, 50, 100]               
                         }]

tuned_param_gb_regr = [{'loss':['ls','lad','huber','quantile'],
                    'learning_rate':[1e-2, 1e-1, 1, 10],
                    'n_estimators':[1,2,10,50,100]
                  
                              
                      }]

tuned_param_bt_regr = [{'n_estimators':[1, 10, 50, 100]               
                         }]


models_regression = {
                            
                            
                            'random forest': {
                                   'estimator': RandomForestRegressor(), 
                                   'param': tuned_param_rf_regr,
                            },
                            
                            'adaboost': {
                                   'estimator': AdaBoostRegressor(), 
                                   'param': tuned_param_ab_regr,
                            },
                            
                            'gradient boosting': {
                                   'estimator': GradientBoostingRegressor(), 
                                   'param': tuned_param_gb_regr,
                            },
                            
                            'bagging tree': {
                                   'estimator': BaggingRegressor(), 
                                   'param': tuned_param_bt_regr,
                            },
                            
                            
                        }



