
import pandas as pd
from analogistics.clean import cleanUsingIQR


def cleanDatatableLearningOrders(D_tem: pd.DataFrame, features: list):
    """
    Prepare warehouse learning table with cleaning data using IQR methods

    Args:
        D_tem (pd.dataFrame): Input dataframe.
        features (list): List of column name - features - to be cleaned.

    Returns:
        D_res_tem_IN (pd.DataFrame): Output inbound dataframe.
        D_res_tem_OUT (pd.DataFrame): Output outbound dataframe.
        perc_IN (float): Cleaning percentage cleaning inbound dataframe.
        perc_OUT (TfloatYPE): percentage cleaning outbound dataframe.

    """

    # set default error variable
    D_res_tem_IN = pd.DataFrame()
    D_res_tem_OUT = pd.DataFrame()
    perc_IN = []
    perc_OUT = []

    D_tem_IN = D_tem[D_tem['INOUT'] == '+']
    D_tem_OUT = D_tem[D_tem['INOUT'] == '-']

    # ############### DATA CLEANING #############

    if len(D_tem_IN) > 0:

        # INBOUND
        for feat in features:  # remove zero values
            D_tem_IN = D_tem_IN[(D_tem_IN[feat] != 0)]

        # clean using IQR
        if len(D_tem_IN) > 0:
            D_res_tem_IN, perc_IN = cleanUsingIQR(D_tem_IN, features)

    if len(D_tem_OUT) > 0:
        # OUTBOUND
        for feat in features:
            D_tem_OUT = D_tem_OUT[(D_tem_OUT[feat] != 0)]

        # clean using IQR
        if len(D_tem_OUT) > 0:
            D_res_tem_OUT, perc_OUT = cleanUsingIQR(D_tem_OUT, features)
    return D_res_tem_IN, D_res_tem_OUT, perc_IN, perc_OUT
