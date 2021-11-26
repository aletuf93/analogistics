import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analogistics.statistics import time_series as ts
from analogistics.supply_chain.P8_performanceAssessment.utilities_movements import getCoverageStats
from analogistics.supply_chain.P8_performanceAssessment.vehicle_assessment import createTabellaMovimenti


def checkPlannedActual(D_mov: pd.DataFrame, locfrom: str = 'LOADING_NODE',
                       locto: str = 'DISCHARGING_NODE',
                       capacityField: str = 'QUANTITY',
                       voyagefield: str = 'VOYAGE_CODE',
                       vehiclefield: str = 'VEHICLE_CODE',
                       timeColumns: dict = {}):
    """
    analyse if planned routes have been respected in actual voyages

    Args:
        D_mov (pd.DataFrame): Input pandas dataframe with movements.
        locfrom (str, optional): Column name with the origin location code. Defaults to 'LOADING_NODE'.
        locto (str, optional): Column name with the destination location code. Defaults to 'DISCHARGING_NODE'.
        capacityField (str, optional): Column name with the transported quantity. Defaults to 'QUANTITY'.
        voyagefield (str, optional): Column name with the voyage code. Defaults to 'VOYAGE_CODE'.
        vehiclefield (str, optional): Column name with the vehicle code. Defaults to 'VEHICLE_CODE'.
        timeColumns (dict, optional): Set of the time columns. Defaults to {}.

    Returns:
        output_figure (dict): output dictionary containing figures.
        df_results (dict): output dictionary containing dataframes.

    """

    df_results = {}
    output_figure = {}

    D = createTabellaMovimenti(D_mov,
                               locfrom=locfrom,
                               locto=locto,
                               capacityField=capacityField,
                               timeColumns=timeColumns
                               )
    if any(column not in D.columns for column in ['PTA', 'PTD', 'ATA', 'ATD']):
        print("WARNING: no actual and provisional columns in D_mov")
        return output_figure, df_results
    accuracy, _ = getCoverageStats(D_mov, analysisFieldList=[locfrom, locto, voyagefield,
                                                             vehiclefield, *list(timeColumns.values())
                                                             ],
                                   capacityField='QUANTITY')

    D_movimenti = D.groupby([vehiclefield, 'Location', 'PTA',
                             'PTD', 'ATA', 'ATD', voyagefield])['Movementquantity'].sum().reset_index()
    D_movimenti['AsPlanned'] = True  # save the movements table if routes have been respected
    colsCheckRoute = ['VoyageCode', 'PlanPerformed']
    results_route = pd.DataFrame(columns=colsCheckRoute)

    colsCheckArcs = ['VoyageCode', 'plannedLocation', 'actualLocation']
    results_arcExchange = pd.DataFrame(columns=colsCheckArcs)

    # identify routes
    routeCode = np.unique(D_movimenti[voyagefield][~D_movimenti[voyagefield].isna()])
    for i in range(0, len(routeCode)):
        codiceRoute = routeCode[i]
        dataRoute = D_movimenti[D_movimenti[voyagefield] == codiceRoute]

        # order by PLANNED
        sortpl = dataRoute.sort_values(by='PTA')
        ordinePlanned = sortpl.index.values

        # order by ACTUAL
        sortact = dataRoute.sort_values(by='ATA')
        ordineActual = sortact.index.values

        check = all(ordineActual == ordinePlanned)

        if(check):  # the route has been performed as planned
            # update voyage table
            temp = pd.DataFrame([[codiceRoute, True]], columns=colsCheckRoute)
            results_route = results_route.append(temp)
        else:  # the route has NOT been performed as planned
            # update voyage table
            temp = pd.DataFrame([[codiceRoute, False]], columns=colsCheckRoute)
            results_route = results_route.append(temp)

            # update arc exchange table

            # identify indexes to update
            indexFrom = sortpl[~(ordineActual == ordinePlanned)].index.values
            indexTo = sortact[~(ordineActual == ordinePlanned)].index.values

            locFrom = dataRoute.Location[indexFrom]
            locTo = dataRoute.Location[indexTo]
            for j in range(0, len(locFrom)):
                temp = pd.DataFrame([[codiceRoute, locFrom.iloc[j], locTo.iloc[j]]], columns=colsCheckArcs)
                results_arcExchange = results_arcExchange.append(temp)

            # Update the table with the flag if the route has been performed as planned
            D_movimenti.loc[(D_movimenti[voyagefield] == codiceRoute) & (D_movimenti.Location.isin(locFrom)), 'AsPlanned'] = False

    # calculate statistics
    stat_exchange = results_arcExchange.groupby(['plannedLocation', 'actualLocation']).size().reset_index()
    stat_exchange.rename(columns={0: 'count'}, inplace=True)
    stat_exchange = stat_exchange.sort_values(by='count', ascending=False)

    stat_exchange['accuracy'] = [accuracy for i in range(0, len(stat_exchange))]
    results_route['accuracy'] = [accuracy for i in range(0, len(results_route))]

    df_results['routeExchange'] = stat_exchange
    df_results['routeExecutedAsPlanned'] = results_route

    # create pie chart with the percentage of respected routes

    sizes = results_route.groupby(['PlanPerformed']).size()
    labels = sizes.index.values
    explode = 0.1 * np.ones(len(sizes))

    fig1, ax1 = plt.subplots(figsize=(20, 10))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Route as planned')
    output_figure['routeAsPlannedPie'] = fig1

    # calculate the difference planned-actual, depending on the date of generation of the record
    D_movimenti['latenessTD'] = lateness_TD = ts.timeStampToDays(D_movimenti.ATD - D_movimenti.PTD)
    D_movimenti['tardinessTD'] = tardiness_TD = lateness_TD.clip(0, None)  # delete all values outside the range [0, +inf]
    lateness_TD_mean = np.mean(lateness_TD)
    tardiness_TD_mean = np.mean(tardiness_TD)

    lateness_TA = ts.timeStampToDays(D_movimenti.ATA - D_movimenti.PTA)
    tardiness_TA = lateness_TA.clip(0, None)
    lateness_TA_mean = np.mean(lateness_TA)
    tardiness_TA_mean = np.mean(tardiness_TA)

    gap_handling = ts.timeStampToDays((D_movimenti.ATD - D_movimenti.ATA) - (D_movimenti.PTD - D_movimenti.PTA))
    handling_gap_mean = np.mean(gap_handling)

    cols = ['mean lateness - dep.', 'mean lateness - arr.',
            'mean tardiness - dep.', 'mean tardiness - arr.', 'mean handling gap']
    schedule_results = pd.DataFrame([[lateness_TD_mean, lateness_TA_mean, tardiness_TD_mean,
                                      tardiness_TA_mean, handling_gap_mean]], columns=cols)
    schedule_results['accuracy'] = [accuracy for i in range(0, len(schedule_results))]

    df_results['schedule_results'] = schedule_results

    return output_figure, df_results


def travelTimedistribution(D_mov: pd.DataFrame,
                           capacityField: str = 'QUANTITY',
                           loadingTA: str = 'PTA_FROM',
                           loadingTD: str = 'PTD_FROM',
                           dischargingTA: str = 'PTA_TO',
                           dischargingTD: str = 'PTD_TO'
                           ):
    """
    calculate the average time spent by products on a vehicle

    Args:
        D_mov (pd.DataFrame): Input pandas dataframe with movements.
        capacityField (str, optional): Column name with the transported quantity. Defaults to 'QUANTITY'.
        loadingTA (str, optional): Column name with the planned time of arrival for loading. Defaults to 'PTA_FROM'.
        loadingTD (str, optional): Column name with the planned time of departure for loading. Defaults to 'PTD_FROM'.
        dischargingTA (str, optional): Column name with the planned time of arrival for offloading. Defaults to 'PTA_TO'.
        dischargingTD (str, optional): Column name with the planned time of departure for offloading. Defaults to 'PTD_TO'.

    Returns:
        imageResults (dict): output dictionary containing figures.
        df_traveltime (pd.DataFrame): output pandas dataFrame.

    """

    df_traveltime = pd.DataFrame(columns=['U_L_BOUND', 'TIME_MEAN', 'TIME_STD'])
    imageResults = {}

    # Get coverage
    accuracy_ub, _ = getCoverageStats(D_mov, analysisFieldList=[dischargingTD, loadingTA],
                                      capacityField=capacityField)

    # Expected travel time per container (UPPER BOUND)
    ExpectedTravelTime_ub = ts.timeStampToDays(D_mov[dischargingTD] - D_mov[loadingTA])
    ExpectedTravelTime_ub = ExpectedTravelTime_ub[ExpectedTravelTime_ub > 0]
    mean_ExpectedTravelTime = np.mean(ExpectedTravelTime_ub)
    std_ExpectedTravelTime = np.std(ExpectedTravelTime_ub)

    data = {'U_L_BOUND': 'upperBound',
            'TIME_MEAN': mean_ExpectedTravelTime,
            'TIME_STD': std_ExpectedTravelTime,
            'accuracy': str(accuracy_ub)}
    temp = pd.DataFrame(data, index=[0])
    df_traveltime = df_traveltime.append(temp)

    # get coverage
    accuracy_lb, _ = getCoverageStats(D_mov, analysisFieldList=[dischargingTA, loadingTD],
                                      capacityField=capacityField)
    # Expected travel time per container (LOWER BOUND)
    ExpectedTravelTime_lb = ts.timeStampToDays(D_mov[dischargingTA] - D_mov[loadingTD])
    ExpectedTravelTime_lb = ExpectedTravelTime_lb[ExpectedTravelTime_lb > 0]
    mean_ExpectedTravelTime = np.mean(ExpectedTravelTime_lb)
    std_ExpectedTravelTime = np.std(ExpectedTravelTime_lb)

    data = {'U_L_BOUND': 'lowerBound',
            'TIME_MEAN': mean_ExpectedTravelTime,
            'TIME_STD': std_ExpectedTravelTime,
            'accuracy': str(accuracy_lb)}
    temp = pd.DataFrame(data, index=[0])
    df_traveltime = df_traveltime.append(temp)

    # define unit of measure
    udm = 'days'
    value_ub = ExpectedTravelTime_ub
    value_lb = ExpectedTravelTime_lb
    if mean_ExpectedTravelTime < 1 / 24 / 60:  # use minutes
        udm = 'minutes'
        value_ub = ExpectedTravelTime_ub * 24 * 60
        value_lb = ExpectedTravelTime_lb * 24 * 60

    elif mean_ExpectedTravelTime < 1:  # use hours
        udm = 'hours'
        value_ub = ExpectedTravelTime_ub * 24
        value_lb = ExpectedTravelTime_lb * 24

    # plot figure
    fig1 = plt.figure()
    plt.hist(value_ub, color='orange')
    plt.hist(value_lb, color='blue', alpha=0.6)
    plt.title(f"Travel time ({udm})")
    plt.xlabel(f"{udm}")
    plt.ylabel('Quantity')
    plt.legend(['Upper bound', 'Lower bound'])
    imageResults["travel_time_per_movement"] = fig1

    return imageResults, df_traveltime


def calculateLoS(D_mov: pd.DataFrame,
                 capacityField: str = 'QUANTITY',
                 timeColumns: dict = {}
                 ):
    """
    define the level of Service

    Args:
        D_mov (pd.DataFrame): Input pandas dataframe with movements.
        capacityField (str, optional): Column name with the transported quantity. Defaults to 'QUANTITY'.
        timeColumns (dict, optional): Set of the time columns. Defaults to {}.

    Returns:
        output_figure (TYPE): output dictionary containing figures.
        coverages (TYPE): output tuple with coverage of the analysis.

    """

    output_figure = {}
    coverages = pd.DataFrame()

    if all(column in timeColumns.keys() for column in ['loadingptd', 'dischargingpta',
                                                       'loadingatd', 'dischargingata']):
        columnsNeeded = [timeColumns['loadingptd'], timeColumns['dischargingpta'],
                         timeColumns['loadingatd'], timeColumns['dischargingata']]

        accuracy, _ = getCoverageStats(D_mov, analysisFieldList=columnsNeeded,
                                       capacityField=capacityField)

        D_time = D_mov.dropna(subset=columnsNeeded)

        plannedTime = D_time[timeColumns['dischargingpta']] - D_time[timeColumns['loadingptd']]
        actualTime = D_time[timeColumns['dischargingata']] - D_time[timeColumns['loadingatd']]

        Los = actualTime < plannedTime
        D_res = Los.value_counts()

        fig1 = plt.figure()
        plt.pie(D_res, autopct='%1.1f%%', shadow=True, startangle=90, labels=D_res.index)
        plt.title('Level of Service')

        output_figure['level_of_service'] = fig1

        coverages = pd.DataFrame([accuracy])

    return output_figure, coverages
