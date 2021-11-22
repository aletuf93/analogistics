import numpy as np
import pandas as pd

from sklearn import metrics

from analytics.transform import dummyColumns


def BootstrapValues(X: pd.DataFrame, nboot: int) -> list:
    """
    randomly bootstrap values from a dataset X

    Args:
        X (pd.DataFrame): input dataframe.
        nboot (int): number of boots.

    Returns:
        listBoots (list): list of boots.

    """

    listBoots = []
    for boot_i in range(nboot):
        boot_tr = np.random.choice(X, size=len(X), replace=True)
        listBoots.append(boot_tr)
    return listBoots


def BootstrapLoop(nboot: int, model, X: pd.DataFrame, y: pd.Series):
    """
    estimate MSE using bootstrap

    Args:
        nboot (int): numer of boots.
        model (TYPE): DESCRIPTION.
        X (pd.DataFrame): input pandas dataframe.
        y (pd.Series): input target variable.

    Returns:
        scores_stat (TYPE): DESCRIPTION.

    """

    X = dummyColumns(X)

    scores_names = ["MSE"]
    scores_boot = np.zeros((nboot, len(scores_names)))
    # coefs_boot = np.zeros((nboot, X.shape[1]))
    orig_all = np.arange(X.shape[0])
    for boot_i in range(nboot):
        boot_tr = np.random.choice(orig_all, size=len(orig_all), replace=True)
        boot_te = np.setdiff1d(orig_all, boot_tr, assume_unique=False)
        Xtr, ytr = X.iloc[boot_tr, :], y[boot_tr]
        Xte, yte = X.iloc[boot_te, :], y[boot_te]
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte).ravel()
        scores_boot[boot_i, :] = metrics.mean_squared_error(yte, y_pred)
        # coefs_boot[boot_i, :] = model.coef_
    # Compute Mean, SE, CI
    scores_boot = pd.DataFrame(scores_boot, columns=scores_names)
    scores_stat = scores_boot.describe(percentiles=[.99, .95, .5, .1, .05, 0.01])
    # print("r-squared: Mean=%.2f, SE=%.2f, CI=(%.2f %.2f)" %\
    #      tuple(scores_stat.ix[["mean", "std", "5%", "95%"], "r2"]))
    # coefs_boot = pd.DataFrame(coefs_boot)
    # coefs_stat = coefs_boot.describe(percentiles=[.99, .95, .5, .1, .05, 0.01])
    # print("Coefficients distribution")
    # print(coefs_stat)
    return scores_stat


def sampleClassWithSameCardinality(df: pd.DataFrame, targetColumn: str,
                                   numRecordsPerclass: float = np.inf,
                                   minRecordPerclass: int = 100,
                                   includeLower: bool = False) -> pd.DataFrame:
    """
    resample a dataset based on the target label

    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        targetColumn (str): Input seriew with target label.
        numRecordsPerclass (float, optional): Number of records expected for each class of the target label. Defaults to np.inf.
        minRecordPerclass (int, optional): Minimum number of records for each class of the target label. Defaults to 100.
        includeLower (bool, optional): If true includes the minimum number of observation for the label with the minimum value count. Defaults to False.

    Returns:
        df_learning (TYPE): DESCRIPTION.

    """
    D_class_stat = df[targetColumn].value_counts()

    # if the minimum number of records is above the threshold
    if min(D_class_stat) > minRecordPerclass:
        numSample = min(numRecordsPerclass, min(D_class_stat))
        df_learning = pd.DataFrame()
        for target in set(df[targetColumn]):
            df_learning = df_learning.append(df[df[targetColumn] == target].sample(numSample))

    # if for some classes there are few records than minRecordPerclass
    else:
        # identify the classes with few records
        fewRecordLabelsList = []
        for target in set(df[targetColumn]):
            if D_class_stat[target] < minRecordPerclass:
                fewRecordLabelsList.append(target)

        D_class_stat_less = D_class_stat[fewRecordLabelsList]
        D_class_stat_greater = D_class_stat[[(i not in fewRecordLabelsList) for i in D_class_stat.index]]

        df_learning = pd.DataFrame()

        # same as before for the greater than
        numSample = min(numRecordsPerclass, min(D_class_stat_greater))
        for target in set(D_class_stat_greater.index):
            df_learning = df_learning.append(df[df[targetColumn] == target].sample(numSample))
        if includeLower:
            # the maximum possible for the lower
            numSample = min(D_class_stat_less)
            for target in set(D_class_stat_less.index):
                df_learning = df_learning.append(df[df[targetColumn] == target].sample(numSample))

    return df_learning
