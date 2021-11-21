import numpy as np
import pandas as pd


def dummyColumns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Transform categorical columns adding dummy columns with 0, 1 values

    Args:
        X (pd.DataFrame): input pandas dataframe.

    Returns:
        X (TYPE): output dataframe with dummy variables.

    """

    # The index of the dataframe is not modified
    try:
        for column in X.columns:
            try:
                if X[column].dtype == object:
                    dummyCols = pd.get_dummies(X[column])
                    X = pd.concat([X, dummyCols], axis=1)
                    del X[column]
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
    return X


def transformClockData(series: pd.Series):
    """
    use cosine and sine transformation to a series
    (e.g. indicating the hour of the days, or the minutes)

    Args:
        series (pd.Series): input datetime series.

    Returns:
        transformedDataCos (pd.Series): output series.
        transformedDataSin (pd.Series): output series.

    """

    transformedDataCos = np.cos(2 * np.pi * series / max(series))
    transformedDataSin = np.sin(2 * np.pi * series / max(series))
    return transformedDataCos, transformedDataSin
