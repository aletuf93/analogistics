# -*- coding: utf-8 -*-
from abc import ABC
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score, classification_report


class GridSearch(ABC):
    def __init__(self):
        self.models_regression = None
        self.models_classification = None

    def train_models_regression(self,
                                X: pd.DataFrame, y: pd.Series,
                                test_size: float = 0.33,
                                cv: int = 5) -> pd.DataFrame:
        """
        Train all the models of a regression grid using grid search with cross validation

        Args:
            X (pd.DataFrame): Input dataframe.
            y (pd.Series): Input target label series.
            test_size (float, optional): Percentage size of the test set. Defaults to 0.33.
            cv (int, optional): Number of folds of the cross-validation. Defaults to 5.

        Returns:
            D_trainingResults (pd.DataFrame): Output DataFrame with the results of the models.

        """
        # get regression grid
        models_regression_grid = self.models_regression

        # split into train and validation set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # define evaluation dataframe
        D_trainingResults = pd.DataFrame(columns=['MODEL_NAME', 'MODEL',
                                                  'PARAMS', 'SCORE_TEST',
                                                  'SCORE_VALIDATION'])

        scores = 'neg_mean_squared_error'
        for model in models_regression_grid.keys():
            estimator = models_regression_grid[model]['estimator']
            param_grid = models_regression_grid[model]['param']

            clf = GridSearchCV(estimator, param_grid, cv=cv, scoring=scores, verbose=10)
            clf.fit(X_train, y_train)
            # clf.cv_results_
            MODEL = clf.best_estimator_
            SCORE_TEST = clf.best_score_
            PARAMS = clf.best_params_

            # validation_set
            y_pred = MODEL.predict(X_test)
            # scorer=get_scorer(scores)
            SCORE_VALIDATION = - mean_squared_error(y_test, y_pred)
            D_trainingResults = D_trainingResults.append(pd.DataFrame([[model, MODEL, PARAMS, SCORE_TEST, SCORE_VALIDATION]],
                                                                      columns=D_trainingResults.columns))
        return D_trainingResults

    def train_models_classification(self, X: pd.DataFrame, y: pd.Series,
                                    test_size: float = 0.33, cv: int = 3,
                                    scores: str = 'accuracy') -> pd.DataFrame:
        """
        Train all the models of a classification grid using grid search with cross validation

        Args:
            X (pd.DataFrame): Input dataframe.
            y (pd.Series): Input target label series.
            test_size (float, optional): Percentage size of the test set. Defaults to 0.33.
            cv (int, optional): Number of folds of the cross-validation. Defaults to 3.
            scores (str, optional): Metric to minimise in the training procedure. Defaults to 'accuracy'.

        Returns:
            D_trainingResults (pd.DataFrame): Output DataFrame with the results of the models.

        """
        # get classification grid
        models_classification = self.models_classification
        # split into train and validation set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # define evaluation dataframe
        D_trainingResults = pd.DataFrame(columns=['MODEL_NAME', 'MODEL', 'PARAMS', 'SCORE_TEST',
                                                  'ACCURACY',
                                                  'PRECISION',
                                                  'RECALL',
                                                  'F1',
                                                  'REPORT'])

        for model in models_classification.keys():
            estimator = models_classification[model]['estimator']
            param_grid = models_classification[model]['param']

            clf = GridSearchCV(estimator, param_grid, cv=cv, scoring=scores, verbose=7, n_jobs=-1)
            clf.fit(X_train, y_train)
            # clf.cv_results_
            MODEL = clf.best_estimator_
            SCORE_TEST = clf.best_score_
            PARAMS = clf.best_params_

            # validation_set
            y_pred = MODEL.predict(X_test)
            # scorer=get_scorer(scores)
            ACCURACY = accuracy_score(y_test, y_pred)
            PRECISION = precision_score(y_test, y_pred, average='weighted')
            RECALL = recall_score(y_test, y_pred, average='weighted')
            F1 = f1_score(y_test, y_pred, average='weighted')

            # get classification report
            # print(y_test)
            # print(y_pred)
            # print(MODEL.classes_)
            # D_rep = classification_report(y_test, y_pred, target_names=MODEL.classes_)
            D_rep = pd.DataFrame()
            D_trainingResults = D_trainingResults.append(pd.DataFrame([[model, MODEL, PARAMS, SCORE_TEST, ACCURACY, PRECISION, RECALL, F1, D_rep]],
                                                                      columns=D_trainingResults.columns))
        return D_trainingResults
