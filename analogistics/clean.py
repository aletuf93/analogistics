# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.special import erfc


def chauvenet(y, mean=None, stdv=None) -> np.array:
    """
    Apply chauvenet criterion for data cleaning

    Args:
        y (TYPE): represent measured data.
        mean (TYPE, optional): DESCRIPTION. Defaults to None.
        stdv (TYPE, optional): DESCRIPTION. Defaults to None.

    Returns:
        TYPE: It returns a boolean array as filter.
        #         The False values correspond to the array elements
        #         that should be excluded.

    """

    if mean is None:
        mean = y.mean()           # Mean of incoming array y
    if stdv is None:
        stdv = y.std()            # Its standard deviation
    N = len(y)                   # Lenght of incoming arrays
    criterion = 1.0 / (2 * N)        # Chauvenet's criterion
    d = abs(y - mean) / stdv         # Distance of a value to mean in stdv's
    d /= 2.0**0.5                # The left and right tail threshold values
    prob = erfc(d)               # Area normal dist.
    filter = prob >= criterion   # The 'accept' filter array with booleans
    return filter                # Use boolean array outside this function


def cleanOutliers(df: pd.DataFrame, features: list):
    """
    Applies the chauvenet criterion to clean the data of a Pandas DataFrame

    Args:
        df (pd.DataFrame): input dataframe to be cleaned.
        features (list): list of feature to consider for the a[pplication of the Chauvenet criterion.

    Returns:
        df (TYPE): output cleaned dataframe.
        Perc (TYPE): percentage of good data in the initial dataframe.

    """
    good = np.ones(len(df))

    for i in range(0, len(features)):
        temp = df.loc[:, features[i]]
        values = chauvenet(temp)
        good = np.logical_and(good, values)

    df = df[good]
    Perc = np.around(float(len(df)) / len(df) * 100, 2)  # percentage of good data
    return df, Perc


def cleanUsingIQR(table: pd.DataFrame, features: list, capacityField: list = []):
    """
    Clean data using the interquartile range method (IQR)

    Args:
        table (pd.DataFrame): input dataframe.
        features (list): ordered list of features to consider for data cleaning. All the features are considered
        one at a time.
        capacityField (list, optional): Field of capacity associated to each record. It is used to calculate the covering
        statistics of the initial data. Defaults to [].

    Returns:
        TYPE: DESCRIPTION.

    """

    table = temp = table.reset_index(drop=True)
    for feature in features:

        if len(temp[feature]) > 0:
            q1, q3 = np.percentile(temp[feature].dropna(), [25, 75])  # percentile ignoring nan values
            if (q1 is not None) & (q3 is not None):
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                temp = temp[(temp[feature] <= upper_bound) & (temp[feature] >= lower_bound)]
                temp = temp.reset_index(drop=True)
    lineCoverage = len(temp) / len(table)
    qtyCoverage = np.nan
    if len(capacityField) > 0:
        qtyCoverage = np.nansum(temp[capacityField]) / np.nansum(table[capacityField])
    return temp, (lineCoverage, qtyCoverage)