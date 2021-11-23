# %% CLASSIFICATION MODEL PARAMETERS
from sklearn.neural_network import MLPClassifier, MLPRegressor

from analogistics.learning.grids import GridSearch

tuned_param_perc = [{'hidden_layer_sizes': [1, 2, 10, 50, 100],
                     'activation': ['identity', 'logistic', 'tanh', 'relu'],
                     'learning_rate':['constant', 'invscaling', 'adaptive']
                     }]

tuned_param_perc_regr = [{'hidden_layer_sizes': [1, 2, 10, 50, 100],
                          'activation': ['identity', 'logistic', 'tanh', 'relu'],
                          'learning_rate': ['constant', 'invscaling', 'adaptive']
                          }]


class GridSearchConnectionist(GridSearch):
    def __init__(self):
        self.models_regression = {'perceptron single layer': {'estimator': MLPRegressor(),
                                                              'param': tuned_param_perc_regr,
                                                              },
                                  }

        self.models_classification = {'perceptron single layer': {'estimator': MLPClassifier(),
                                                                  'param': tuned_param_perc
                                                                  },
                                      }
