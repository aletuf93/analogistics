# -*- coding: utf-8 -*-
from analogistics.statistics.time_series import timeStampToDays
from analogistics.supply_chain.P8_performanceAssessment.utilities_movements import getCoverageStats
from analogistics.graph.graph import plotGraph
import pandas as pd


def networkStatistics(D_mov: pd.dataFrame,
                      terminalfieldFrom: str = 'LOADING_NODE',
                      terminalfieldto: str = 'DISCHARGING_NODE',
                      capacityField: str = 'QUANTITY',
                      actual: bool = False,
                      timeColumns: dict = {}):
    """
    Plot the network connection on a graph

    Args:
        D_mov (pd.dataFrame): Input pandas dataframe with movements.
        terminalfieldFrom (str, optional): Column name with the origin location code. Defaults to 'LOADING_NODE'.
        terminalfieldto (str, optional): Column name with the destination location code. Defaults to 'DISCHARGING_NODE'.
        capacityField (str, optional): Column name with the transported quantity. Defaults to 'QUANTITY'.
        actual (bool, optional): If true, actual time windows are used. Defaults to False.
        timeColumns (dict, optional): Dictionary with time columns. Defaults to {}.

    Returns:
        outputFigure (dict): output dictionary containing figures.
        sailingTime (pd.DataFrame): output Pandas Dataframe.

    """

    outputFigure = {}
    sailingTime = pd.DataFrame()

    if actual == 'PROVISIONAL':
        accuracy, _ = getCoverageStats(D_mov, analysisFieldList=[timeColumns['dischargingpta'],
                                                                 timeColumns['loadingptd']],
                                       capacityField=capacityField)
        # Calculate distances
        D_mov['sailingTime'] = timeStampToDays(D_mov[timeColumns['dischargingpta']] - D_mov[timeColumns['loadingptd']])

    elif actual == 'ACTUAL':
        accuracy, _ = getCoverageStats(D_mov, analysisFieldList=[timeColumns['dischargingata'],
                                                                 timeColumns['loadingatd']],
                                       capacityField=capacityField)
        # Calculate distances
        D_mov['sailingTime'] = timeStampToDays(D_mov[timeColumns['dischargingata']] - D_mov[timeColumns['loadingatd']])

    D_filterActual = D_mov.dropna(subset=['sailingTime'])
    sailingTime = D_filterActual.groupby([terminalfieldFrom, terminalfieldto])['sailingTime'].mean().reset_index()

    sailingTime = D_filterActual.groupby([terminalfieldFrom, terminalfieldto]).agg({'sailingTime': ['mean', 'std',
                                                                                                    'size']}).reset_index()
    sailingTime.columns = list(map(''.join, sailingTime.columns.values))

    fig1 = plotGraph(sailingTime, terminalfieldFrom, terminalfieldto, 'sailingTimemean',
                     'sailingTimesize', 'Network flow', arcLabel=False)
    outputFigure[f"NetworkGraph_{actual}"] = fig1

    sailingTime['accuracy'] = [accuracy for i in range(0, len(sailingTime))]

    return outputFigure, sailingTime
