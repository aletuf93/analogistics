# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def buildLearningTablePickList(D_movements: pd.DataFrame):
    """
    Define a learning table based on the attributes of a movement picking list

    Args:
        D_movements (pd.DataFrame): Input dataframe with warehouse movements.

    Returns:
        D_learning (pd.dataFrame): Output dataframe with learning table.

    """
    D_learning = D_movements.groupby(['NODECODE', 'PICKINGLIST', 'INOUT']).agg({'QUANTITY': ['sum', 'size'],
                                                                                'TIMESTAMP_IN': ['max', 'min'],
                                                                                'RACK': ['max', 'min'],
                                                                                'BAY': ['max', 'min'],
                                                                                'LEVEL': ['max', 'min'],
                                                                                'LOCCODEX': ['max', 'min'],
                                                                                'LOCCODEY': ['max', 'min'],
                                                                                }).reset_index()

    D_learning.columns = ['NODECODE', 'PICKINGLIST', 'INOUT',
                          'sum_QUANTITY', 'count_QUANTITY',
                          'max_TIMESTAMP_IN', 'min_TIMESTAMP_IN',
                          'max_RACK', 'min_RACK',
                          'max_BAY', 'min_BAY',
                          'max_LEVEL', 'min_LEVEL',
                          'max_LOCCODEX', 'min_LOCCODEX',
                          'max_LOCCODEY', 'min_LOCCODEY']

    # clean results
    D_learning['TIMESEC_SPAN'] = D_learning['max_TIMESTAMP_IN'] - D_learning['min_TIMESTAMP_IN']
    D_learning['RACK_SPAN'] = D_learning['max_RACK'] - D_learning['min_RACK']
    D_learning['BAY_SPAN'] = D_learning['max_BAY'] - D_learning['min_BAY']
    D_learning['LEVEL_SPAN'] = D_learning['max_LEVEL'] - D_learning['min_LEVEL']
    D_learning['LOCCODEX_SPAN'] = D_learning['max_LOCCODEX'] - D_learning['min_LOCCODEX']
    D_learning['LOCCODEY_SPAN'] = D_learning['max_LOCCODEY'] - D_learning['min_LOCCODEY']

    D_learning = D_learning.drop(columns=['max_TIMESTAMP_IN', 'min_TIMESTAMP_IN', 'max_RACK', 'min_RACK',
                                          'max_BAY', 'min_BAY', 'max_LEVEL', 'min_LEVEL', 'max_LOCCODEX',
                                          'min_LOCCODEX', 'max_LOCCODEY', 'min_LOCCODEY'])

    D_learning['TIMESEC_SPAN'] = D_learning['TIMESEC_SPAN'].dt.seconds

    return D_learning


def histogramKeyVars(D_learning: pd.DataFrame):
    """
    Define histograms on the key variables

    Args:
        D_learning (pd.DataFrame): Input pandas dataframe with learning table.

    Returns:
        output_figure (dict): Dictionary with output figures.

    """
    output_figure = {}

    columnToAnalyse = list(D_learning.columns)
    columnToAnalyse.remove('NODECODE')
    columnToAnalyse.remove('PICKINGLIST')
    columnToAnalyse.remove('INOUT')

    # split inbound and outbound
    D_learning_positive = D_learning[D_learning['INOUT'] == '+']
    D_learning_negative = D_learning[D_learning['INOUT'] == '-']

    for col in columnToAnalyse:

        # inbound
        fig = plt.figure()
        plt.hist(D_learning_positive[col], color='orange')
        plt.title(f"Histogram: {col}, INBOUND")
        plt.xlabel(f"{col}")
        plt.ylabel("frequency")
        output_figure[f"{col}_inbound_histogram"] = fig

        # outbound
        fig = plt.figure()
        plt.hist(D_learning_negative[col], color='orange')
        plt.title(f"Histogram: {col}, OUTBOUND")
        plt.xlabel(f"{col}")
        plt.ylabel("frequency")
        output_figure[f"{col}_outbound_histogram"] = fig

    return output_figure


def exploreKeyVars(D_learning: pd.DataFrame):
    """
    Explore the key variables of a learning table

    Args:
        D_learning (pd.DataFrame): Input pandas dataFrame.

    Returns:
        output_figures (dict): Output dictionary containing figures.

    """
    output_figures = {}

    # pairplot
    fig = sn.pairplot(D_learning, hue='INOUT', diag_kind='hist')
    output_figures['pairplot'] = fig

    D_learning_positive = D_learning[D_learning['INOUT'] == '+']
    D_learning_negative = D_learning[D_learning['INOUT'] == '-']

    # inbound_correlation
    df_corr = D_learning_positive.drop(columns=['NODECODE', 'PICKINGLIST', 'INOUT'])
    corr_matrix = df_corr.corr()
    plt.figure()
    fig = sn.heatmap(corr_matrix, annot=True)
    fig = fig.get_figure()
    output_figures['correlation_inbound'] = fig

    # outboud_correlation
    df_corr = D_learning_negative.drop(columns=['NODECODE', 'PICKINGLIST', 'INOUT'])
    corr_matrix = df_corr.corr()
    plt.figure()
    fig = sn.heatmap(corr_matrix, annot=True)
    fig = fig.get_figure()
    output_figures['correlation_outbound'] = fig

    return output_figures
