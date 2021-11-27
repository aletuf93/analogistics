# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def getCoverageStats(D_mov: pd.DataFrame, analysisFieldList: list, capacityField: str = 'QUANTITY'):
    """
    Calculate statistics of an input dataframe by counting the elements over nan values and the percentage coverage over the nan
    when a quantity field is specified

    Args:
        D_mov (pd.DataFrame): Input pandas dataFrame.
        analysisFieldList (list): List of analysis columns to calculate the coverages.
        capacityField (str, optional): Column name of the quantity field. Defaults to 'QUANTITY'.

    Returns:
        lineCoverage (float): Percentage coverage.
        float: Percentage coverage over the quantity.
        tot_lines (int): Numer of total lines.

    """

    n_lines = len(D_mov)
    tot_qties = np.nansum(D_mov[capacityField])

    tot_lines = len(D_mov[analysisFieldList].dropna())

    if isinstance(analysisFieldList, list):
        listCol = analysisFieldList
        listCol.append(capacityField)
        D_filtered_qties = D_mov[listCol].dropna()

    else:
        D_filtered_qties = D_mov[[analysisFieldList, capacityField]].dropna()

    lineCoverage = tot_lines / n_lines

    # Quantity coverage if specified in the input attributes
    if capacityField == analysisFieldList:
        qtyCoverage = 1
    else:
        qtyCoverage = np.nansum(D_filtered_qties[capacityField]) / tot_qties

    return (lineCoverage, qtyCoverage), tot_lines


def movementStatistics(D_mov: pd.DataFrame, capacityField: str = 'QUANTITY'):
    """
    Performs global analysis on the D_mov dataframe

    Args:
        D_mov (pd.DataFrame): Input pandas dataFrame.
        capacityField (str, optional): quantity field to calculate coverages on. Defaults to 'QUANTITY'.

    Returns:
        D_global (pd.DataFrame): dataframe with global statistics.

    """

    data = {}
    coverage_stats = {}

    for col in D_mov.columns:
        # calculate counting statistics
        coverage_stats[f"COUNT.{col}"], nrows = getCoverageStats(D_mov, col, capacityField)
        if any([isinstance(i, dict) for i in D_mov[col]]):
            data[f"COUNT.{col}"] = nrows
        else:
            data[f"COUNT.{col}"] = len(D_mov[col].unique())
        # if a number calculate sum statistics
        if (D_mov[col].dtypes == np.float) | (D_mov[col].dtypes == np.int):
            data[f"SUM.{col}"] = np.nansum(D_mov[col])
            coverage_stats[f"SUM.{col}"] = coverage_stats[f"COUNT.{col}"]

        # if a date identify first and last day, and the count of the included days
        if (D_mov[col].dtypes == np.datetime64) | (D_mov[col].dtypes == '<M8[ns]'):
            BookingDates = np.unique(D_mov[col].dt.date)
            beginningTimeHorizon = min(BookingDates)
            endTimeHorizon = max(BookingDates)
            NofBookingDays = len(BookingDates)

            data[f"N.OF.DAYS.{col}"] = NofBookingDays
            data[f"FIRST.DAY.{col}"] = beginningTimeHorizon
            data[f"LAST.DAY.{col}"] = endTimeHorizon

            coverage_stats[f"N.OF.DAYS.{col}"] = coverage_stats[f"COUNT.{col}"]
            coverage_stats[f"FIRST.DAY.{col}"] = coverage_stats[f"COUNT.{col}"]
            coverage_stats[f"LAST.DAY.{col}"] = coverage_stats[f"COUNT.{col}"]

    D_global = pd.DataFrame([data, coverage_stats]).transpose()
    D_global.columns = ['VALUE', 'ACCURACY']
    return  D_global