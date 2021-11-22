from sklearn import tree

from analogistics.learning.grids import GridSearch

tuned_param_dt = [{'criterion': ['gini', 'entropy'],
                   'splitter': ['best', 'random'],
                   'max_features': ['auto', 'sqrt', 'log2'],
                   'max_depth': range(1, 20)
                   }]

tuned_param_dt_regr = [{'criterion': ['mse', 'mae'],
                        'splitter': ['best', 'random'],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'max_depth': range(1, 20)
                        }]


class GridSearchSymbolist(GridSearch):
    def __init__(self):

        self.models_classification = {'decision tree': {'estimator': tree.DecisionTreeClassifier(),
                                                        'param': tuned_param_dt,
                                                        },
                                      }
        self.models_regression = {'decision tree': {'estimator': tree.DecisionTreeRegressor(),
                                                    'param': tuned_param_dt_regr,
                                                    },
                                  }
