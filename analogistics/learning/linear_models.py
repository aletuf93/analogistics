import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import metrics

# %%
'''
questo modulo costruisce le griglie con i parametri per i modelli
lineari di regressione e classificazione
'''
# %% IMPOSTO MODELLI E PARAMETRI DI DEFAULT PER LA REGRESSIONE

tuned_param_linear_regression = [{'fit_intercept':[True,False],
                                'normalize':[True, False]}]
                                

tuned_param_ridge_regression =[{'fit_intercept':[True,False],
                                'normalize':[True, False],
                               'alpha': [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 1e1, 1e2, 1e3]}]

tuned_param_lasso_regression =[{'fit_intercept':[True,False],
                               'normalize':[True, False],
                               'alpha': [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 1e1, 1e2, 1e3]}]

tuned_param_elasticnet_regression =[{'fit_intercept':[True,False],
                                    'normalize':[True, False],
                               'alpha': [1e-5, 1e-4, 1e-3, 1e-2,1e-1, 1, 1e1, 1e2, 1e3]}]

tuned_param_lars_regression =[{'fit_intercept':[True,False],
                              'normalize':[True, False]}
                               ]



models_regression = {
    'regr_linear': {
           'estimator': LinearRegression(), 
           'param': tuned_param_linear_regression,
          },
    'regr_ridge': {
           'estimator': Ridge(), 
           'param': tuned_param_ridge_regression,
          },
    'regr_lasso': {
           'estimator': Lasso(), 
           'param': tuned_param_lasso_regression,
          },
    'regr_elasticnet': {
           'estimator': ElasticNet(), 
           'param': tuned_param_elasticnet_regression,
          },
    'regr_lars': {
           'estimator': Lars(), 
           'param': tuned_param_lars_regression,
          },
   
          }



# %% IMPOSTO MODELLI E PARAMETRI DI DEFAULT PER LA CLASSIFICAZIONE

tuned_param_lda = [{'solver':['svd','lsqr','eigen'],
                     'shrinkage':[None, 'auto']}]

tuned_param_qda = [{'priors':[None]}]

tuned_param_logistic = [{'penalty':['l1','l2','elasticnet','none'],
                         'dual':[True, False],
                         'fit_intercept':[True, False]
                         }]

models_classification = {
                            'linear_discriminant_analysis': {
                                   'estimator': LinearDiscriminantAnalysis(), 
                                   'param': tuned_param_lda,
                            },
                            
                            'quadratic_discriminant_analysis': {
                                   'estimator': QuadraticDiscriminantAnalysis(), 
                                   'param': tuned_param_qda,
                            },
                            
                            'logistic_regression': {
                                   'estimator': LogisticRegression(), 
                                   'param': tuned_param_logistic,
                            },
                        }


def fit_linear_reg(X: pd.DataFrame, Y: pd.DataFrame):
    """
    Fit linear regression model and return RSS and R squared values

    Args:
        X (pd.DataFrame): input pandas dataframe.
        Y (pd.DataFrame): input target label.

    Returns:
        RSS (float): output Root Squared sum.
        R_squared (float): output r squared.

    """
    model_k = LinearRegression(fit_intercept=True)
    model_k.fit(X, Y)
    RSS = metrics.mean_squared_error(Y, model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X, Y)
    return RSS, R_squared
