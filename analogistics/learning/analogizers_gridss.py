from sklearn import svm

# %% GRID PARAMETERS CLASSIFICATION

tuned_param_svm= [{'kernel': ['rbf'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000],                
                  }]

tuned_param_svm_linear= [{'penalty': ['l1','l2'], 
                          'C': [1, 10, 100, 1000],                
                  }]
                                
models_classification = {
                            'svm': {
                                   'estimator': svm.SVC(), 
                                   'param': tuned_param_svm,
                            },
                            
                            'svm_linear': {
                                   'estimator': svm.LinearSVC(), 
                                   'param': tuned_param_svm_linear,
                            },
                        }
# GRID PARAMETERS REGRESSION
# %% GRID PARAMETERS

tuned_param_regr= [{'kernel': ['rbf'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000],                
                  }]
                                
models_regression = {
                            'svm': {
                                   'estimator': svm.SVR(), 
                                   'param': tuned_param_regr,
                            },
                            
                            
                        }



