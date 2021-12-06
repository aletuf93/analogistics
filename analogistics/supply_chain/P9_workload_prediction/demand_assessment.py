# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analogistics.statistics import time_series as ts
from statsmodels.tsa.seasonal import seasonal_decompose
from analogistics.supply_chain.P8_performance_assessment.utilities_movements import getCoverageStats


def getAdvanceInPlanning(D_mov: pd.DataFrame, loadingptafield: str = 'LOADING_TIME_WINDOWS_PROVISIONAL_START'):
    """
    calculate the distribution of the advance in planning

    Args:
        D_mov (pd.DataFrame): Input movements dataFrame.
        loadingptafield (str, optional): column name with planning timestamp. Defaults to 'LOADING_TIME_WINDOWS_PROVISIONAL_START'.

    Returns:
        plt.Figure: output figure.
        pd.DataFrame: output dataframe.

    """

    output_figure = {}
    output_data = {}
    output_coverage = {}

    # filterNan
    D_mov_filtered = D_mov[['TIMESTAMP_IN', loadingptafield]].dropna()

    if len(D_mov_filtered) == 0:
        return output_figure, pd.DataFrame(['No PTA fields to perform this analysis'])
    if loadingptafield == 'TIMESTAMP_IN':  # when using the same column, get zero
        mean_advanceInPlanning = std_advanceInPlanning = 0
        advanceInPlanningDistribution = []
    else:
        advanceInPlanning = D_mov_filtered[loadingptafield] - D_mov_filtered['TIMESTAMP_IN']
        advanceInPlanningD = advanceInPlanning.dt.components['days']
        advanceInPlanningH = advanceInPlanning.dt.components['hours']
        advanceInPlanningM = advanceInPlanning.dt.components['minutes']
        advanceInPlanning = advanceInPlanningD + advanceInPlanningH / 24 + advanceInPlanningM / (60 * 24)
        advanceInPlanning = advanceInPlanning[advanceInPlanning > 0]
        mean_advanceInPlanning = np.mean(advanceInPlanning)
        std_advanceInPlanning = np.std(advanceInPlanning)
        advanceInPlanningDistribution = advanceInPlanning

    if len(advanceInPlanningDistribution) > 0:
        # Advance in planning
        fig_planningAdvance = plt.figure()
        plt.title('Days of advance in booking')
        plt.hist(advanceInPlanning, color='orange', bins=np.arange(0, max(advanceInPlanningDistribution), 1))
        plt.xlabel('days')
        plt.ylabel('N.ofBooks')

        # output_figure
        output_figure['ADVANCE_IN_PLANNING'] = fig_planningAdvance

    # output_data
    output_data['ADVANCE_PLANNING_MEAN'] = mean_advanceInPlanning
    output_data['ADVANCE_PLANNING_STD'] = std_advanceInPlanning
    output_data['SERIES'] = advanceInPlanningDistribution

    # get coverage
    output_coverage['ADVANCE_PLANNING_MEAN'] = getCoverageStats(D_mov, loadingptafield, capacityField='QUANTITY')
    output_coverage['ADVANCE_PLANNING_STD'] = output_coverage['ADVANCE_PLANNING_MEAN']
    output_coverage['SERIES'] = output_coverage['ADVANCE_PLANNING_MEAN']

    D_global = pd.DataFrame([output_data, output_coverage]).transpose()
    D_global.columns = ['VALUE', 'ACCURACY']

    return output_figure, D_global


def bookingStatistics(D_mov: pd.DataFrame, capacityField: str = 'QUANTITY',
                      timeVariable: str = 'TIMESTAMP_IN',
                      samplingInterval: list = ['day', 'week', 'month']):
    """
    trend analysis by month, wek and day

    Args:
        D_mov (pd.DataFrame): Inpur movements DataFrame.
        capacityField (str, optional): Column name with quantity. Defaults to 'QUANTITY'.
        timeVariable (str, optional): Column name with timestamp. Defaults to 'TIMESTAMP_IN'.
        samplingInterval (list, optional): List of sampling intervals to consider. Defaults to ['day', 'week', 'month'].

    Returns:
        imageResults (dict): Dictionary with output figures.
        dataframeResults (dict): Dictionary with output dataFrames.

    """

    # create dictionary results
    imageResults = {}
    dataframeResults = {}
    dataResults_trend = {}
    coverage_stats = {}

    # calculate coverages
    accuracy, _ = getCoverageStats(D_mov, analysisFieldList=timeVariable, capacityField=capacityField)

    D_OrderTrend = D_mov.groupby([timeVariable]).size().reset_index()
    D_OrderTrend.columns = ['DatePeriod', 'Orders']
    D_OrderTrend = D_OrderTrend.sort_values(['DatePeriod'])

    for spInterval in samplingInterval:
        if spInterval == 'month':
            timeSeries_analysis = ts.groupPerMonth(D_OrderTrend, 'DatePeriod', 'Orders', 'sum')

        elif spInterval == 'week':
            timeSeries_analysis = ts.groupPerWeek(D_OrderTrend, 'DatePeriod', 'Orders', 'sum')

        elif spInterval == 'day':
            timeSeries_analysis = D_OrderTrend.set_index('DatePeriod')
            timeSeries_analysis = timeSeries_analysis['Orders']

        # daily trend
        fig1 = plt.figure()
        plt.plot(timeSeries_analysis.index.values, timeSeries_analysis, color='orange')
        plt.title(f"TREND: {timeVariable} per {spInterval}")
        plt.xticks(rotation=30)
        imageResults[f"trend_{spInterval}"] = fig1

        # distribution
        fig2 = plt.figure()
        plt.hist(timeSeries_analysis, color='orange')
        plt.title(f"Frequency analysis of {timeVariable} per {spInterval}")
        plt.xlabel(f"{timeVariable}")
        plt.ylabel(f"{spInterval}")
        imageResults[f"pdf_{spInterval}"] = fig2
        # fig1.savefig(dirResults+'\\02-ContainerPDFDaily.png')

        daily_mean = np.mean(timeSeries_analysis)
        daily_std = np.std(timeSeries_analysis)

        # calculate the values
        dataResults_trend[f"{timeVariable}_{spInterval}_MEAN"] = daily_mean
        dataResults_trend[f"{timeVariable}_{spInterval}_STD"] = daily_std

        # assign coverages
        coverage_stats[f"{timeVariable}_{spInterval}_MEAN"] = accuracy
        coverage_stats[f"{timeVariable}_{spInterval}_STD"] = accuracy

    # save dataframe with results
    D_trend_stat = pd.DataFrame([dataResults_trend, coverage_stats]).transpose()
    D_trend_stat.columns = ['VALUE', 'ACCURACY']
    dataframeResults['trend_df'] = D_trend_stat

    # distribution per weekday
    D_grouped = ts.groupPerWeekday(D_OrderTrend, timeVariable='DatePeriod', groupVariable='Orders')
    D_grouped['accuracy'] = [accuracy for i in range(0, len(D_grouped))]
    dataframeResults['weekday_df'] = D_grouped

    fig3 = plt.figure()
    plt.bar(D_grouped.index.values, D_grouped['mean'], color='orange')
    plt.title(f"N.of {timeVariable} per day of the week")
    plt.xlabel('day of the week')
    plt.ylabel('Frequency')
    imageResults["pdf_dayOfTheWeek"] = fig3

    return imageResults, dataframeResults


def plotquantitytrend(D_temp: pd.DataFrame, date_field: str = 'TIMESTAMP_IN',
                      filterVariable: list = [], filterValue: list = [],
                      quantityVariable: str = 'sum_QUANTITY',
                      countVariable: str = 'count_TIMESTAMP_IN', titolo: str = ''):
    """
    the function return a figure with two subplots on for quantities the other for lines

    Args:
        D_temp (pd.DataFrame): input dataframe.
        date_field (str, optional): string with the column name for the date field. Defaults to 'TIMESTAMP_IN'.
        filterVariable (list, optional): string with the column name for filtering the dataframe. Defaults to [].
        filterValue (list, optional): value to filter the dataframe. Defaults to [].
        quantityVariable (str, optional): string with the column name for the sum of the quantities. Defaults to 'sum_QUANTITY'.
        countVariable (str, optional): string with the column name for the count. Defaults to 'count_TIMESTAMP_IN'.
        titolo (str, optional): title of the figure. Defaults to ''.

    Returns:
        fig (plt.figure): Output figure.

    """

    if len(filterVariable) > 0:
        D_temp = D_temp[D_temp[filterVariable] == filterValue]
    D_temp = D_temp.sort_values(date_field)
    D_temp = D_temp.reset_index(drop=True)
    D_temp = D_temp.dropna(subset=[date_field, quantityVariable])

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(titolo)

    # plot quantity
    axs[0].plot(D_temp[date_field], D_temp[quantityVariable])
    axs[0].set_title('Quantity trend')
    for tick in axs[0].get_xticklabels():
        tick.set_rotation(45)

    # plot lines
    axs[1].plot(D_temp[date_field], D_temp[countVariable])
    axs[1].set_title('Lines trend')
    for tick in axs[1].get_xticklabels():
        tick.set_rotation(45)

    return fig


def plotQuantityTrendWeeklyDaily(D_temp: pd.DataFrame, date_field: str = 'TIMESTAMP_IN',
                                 filterVariable: str = [], filterValue: str = [],
                                 quantityVariable: str = 'sum_QUANTITY',
                                 countVariable: str = 'count_TIMESTAMP_IN',
                                 titolo: str = ''):
    """
    the function return a figure with two subplots on for quantities the other for lines

    Args:
        D_temp (pd.dataFrame): input dataframe.
        date_field (str, optional): string with the column name for the date field. Defaults to 'TIMESTAMP_IN'.
        filterVariable (str, optional): string with the column name for filtering the dataframe. Defaults to [].
        filterValue (list, optional): value to filter the dataframe. Defaults to [].
        quantityVariable (str, optional): string with the column name for the sum of the quantities. Defaults to 'sum_QUANTITY'.
        countVariable (str, optional): string with the column name for the count. Defaults to 'count_TIMESTAMP_IN'.
        titolo (str, optional): title of the figure. Defaults to ''.

    Returns:
        fig (plt.figure): Output figure.

    """

    if len(filterVariable) > 0:
        D_temp = D_temp[D_temp[filterVariable] == filterValue]
    D_temp = D_temp.sort_values(date_field)
    D_temp = D_temp.reset_index(drop=True)
    D_temp = D_temp.dropna(subset=[date_field, quantityVariable])

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(titolo)

    # QUANTITIES
    # extract daily time series
    timeSeries = pd.DataFrame(D_temp[[date_field, quantityVariable]])
    timeSeries_day = timeSeries.set_index(date_field).resample('D').sum()

    # extract weekly time series
    timeSeries_week = ts.groupPerWeek(timeSeries, date_field, quantityVariable, 'sum')

    # extract monthly time series
    timeSeries_month = ts.groupPerMonth(timeSeries, date_field, quantityVariable, 'sum')

    # plot weekly-daily
    axs[0].plot(timeSeries_day)
    axs[0].plot(timeSeries_week)
    axs[0].plot(timeSeries_month)
    axs[0].set_title('Quantity trend')
    axs[0].legend(['daily time series', 'weekly time series', 'monthly time series'])
    for tick in axs[0].get_xticklabels():
        tick.set_rotation(45)

    # LINES

    # extract daily time series
    timeSeries = pd.DataFrame(D_temp[[date_field, countVariable]])
    timeSeries_day = timeSeries.set_index(date_field).resample('D').sum()

    # extract weekly time series
    timeSeries_week = ts.groupPerWeek(timeSeries, date_field, countVariable, 'sum')

    # extract monthly time series
    timeSeries_month = ts.groupPerMonth(timeSeries, date_field, countVariable, 'sum')

    # plot weekly-daily
    axs[1].plot(timeSeries_day)
    axs[1].plot(timeSeries_week)
    axs[1].plot(timeSeries_month)
    axs[1].set_title('Lines trend')
    axs[1].legend(['daily time series', 'weekly time series', 'monthly time series'])
    for tick in axs[1].get_xticklabels():
        tick.set_rotation(45)

    return fig


def decomposeTimeSeries(D_time: pd.DataFrame, seriesVariable: str,
                        samplingInterval: str = 'week', date_field: str = 'TIMESTAMP_IN',
                        decompositionModel: str = 'additive'):
    """
    defines a graph decomposing a time series

    Args:
        D_time (pd.DataFrame): input dataframe.
        seriesVariable (str): string with the name of the column containing the datetime series.
        samplingInterval (str, optional): if week it groups the series for week. Defaults to 'week'.
        date_field (str, optional): string with the name of the column containing the series. Defaults to 'TIMESTAMP_IN'.
        decompositionModel (str, optional): argument of seasonal_decompose (additive or multiplicative). Defaults to 'additive'.

    Returns:
        TYPE: DESCRIPTION.

    """

    # estract daily time series
    timeSeries = pd.DataFrame(D_time[[date_field, seriesVariable]])
    timeSeries_analysis = timeSeries.set_index(date_field).resample('D').sum()
    timeSeries_analysis[date_field] = timeSeries_analysis.index.values

    if samplingInterval == 'month':
        timeSeries_analysis = ts.groupPerMonth(timeSeries_analysis, date_field, seriesVariable, 'sum')
        frequency = min(12, len(timeSeries_analysis) - 1)  # search yearly frequency
    elif samplingInterval == 'week':
        timeSeries_analysis = ts.groupPerWeek(timeSeries_analysis, date_field, seriesVariable, 'sum')
        frequency = min(4, len(timeSeries_analysis) - 1)  # search monthly frequency
    elif samplingInterval == 'day':
        timeSeries_analysis = timeSeries_analysis[seriesVariable]
        frequency = min(7, len(timeSeries_analysis) - 1)  # search weekly frequency

    if len(timeSeries_analysis) < 2 * frequency:
        print(f"Not enough values to decompose series with sampling interval {samplingInterval}")
        return plt.figure()
    result = seasonal_decompose(timeSeries_analysis, model=decompositionModel, freq=frequency)
    fig = result.plot()
    return fig


def seasonalityWithfourier(D_time: pd.DataFrame, seriesVariable: str,
                           samplingInterval: str = 'week',
                           date_field: str = 'TIMESTAMP_IN', titolo: str = ''):
    """
    decompose the seasonal part of a time series using Fourier transform

    Args:
        D_time (pd.dataFrame): Input dataFrame.
        seriesVariable (str): string with the name of the column containing the series.
        samplingInterval (str, optional): if week it groups the series for week or day. Defaults to 'week'.
        date_field (str, optional): string with the name of the column containing the datetime series. Defaults to 'TIMESTAMP_IN'.
        titolo (str, optional): title of the figure. Defaults to ''.

    Returns:
        fig (TYPE): DESCRIPTION.

    """

    # extract time series
    timeSeries = pd.DataFrame(D_time[[date_field, seriesVariable]])
    timeSeries_analysis = timeSeries.set_index(date_field).resample('D').sum()
    timeSeries_analysis[date_field] = timeSeries_analysis.index.values

    if samplingInterval == 'month':
        timeSeries_analysis = ts.groupPerMonth(timeSeries_analysis, date_field, seriesVariable, 'sum')
    elif samplingInterval == 'week':
        timeSeries_analysis = ts.groupPerWeek(timeSeries_analysis, date_field, seriesVariable, 'sum')
    elif samplingInterval == 'day':
        timeSeries_analysis = timeSeries_analysis[seriesVariable]

    y = np.array(timeSeries_analysis)
    D = ts.fourierAnalysis(y)

    fig = plt.figure()
    plt.stem(1 / D['Frequency_domain_value'], D['Amplitude'])
    plt.title(f"Amplitude spectrum {titolo}")
    plt.xlabel(f"Time domain: {samplingInterval}")
    plt.ylabel('Amplitude')
    return fig
