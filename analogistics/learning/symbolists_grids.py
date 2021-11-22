from sklearn import tree

# %% CLASSIFICATION MODEL PARAMETERS


tuned_param_dt= [{'criterion':['gini','entropy'],
                  'splitter':['best','random'],
                  'max_features':['auto','sqrt','log2'],
                  'max_depth': range(1,20)                 
                  }]


                                
models_classification = {
                            'decision tree': {
                                   'estimator': tree.DecisionTreeClassifier(), 
                                   'param': tuned_param_dt,
                            },
                            
                           
                        }



# %% REGRESSION MODEL PARAMETERS

tuned_param_dt_regr= [{'criterion':['mse','mae'],
                  'splitter':['best','random'],
                  'max_features':['auto','sqrt','log2'],
                  'max_depth': range(1,20)                 
                  }]


                                
models_regression = {
                            'decision tree': {
                                   'estimator': tree.DecisionTreeRegressor(), 
                                   'param': tuned_param_dt_regr,
                            },
                            
                           
                        }


