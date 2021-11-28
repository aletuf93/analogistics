# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from analogistics.supply_chain.P8_performance_assessment.utilities_movements import getCoverageStats
from analogistics.supply_chain.P9_workload_prediction.demand_assessment import getAdvanceInPlanning
from analogistics.graph.graph import plotGraph


def createTabellaMovimenti(D_mov: pd.DataFrame,
                           locfrom: str = 'LOADING_NODE',
                           locto: str = 'DISCHARGING_NODE',
                           capacityField: str = 'QUANTITY',
                           timeColumns: dict = {}
                           ):
    """
    Create a table with the movements (one line for each movement)

    Args:
        D_mov (pd.dataFrame): Input pandas DataFrame.
        locfrom (str, optional): Column name containing origin nodes. Defaults to 'LOADING_NODE'.
        locto (str, optional): Column name containing destination nodes. Defaults to 'DISCHARGING_NODE'.
        capacityField (str, optional): Column name containing transported quantities. Defaults to 'QUANTITY'.
        timeColumns (dict, optional): Dict containing time windows column names. Defaults to {}.

    Returns:
        D (pd.DataFrame): Output movements DataFrame.

    """
    # Split movements into two rows (IN and OUT)

    print("**** DEFINE D MOV IN/OUT ****")
    # check which felds are available, and define grouping keys
    columnsCompleteFrom = ['loadingpta', 'loadingptd', 'loadingata', 'loadingatd']
    columnsCompleteTo = ['dischargingpta', 'dischargingptd', 'dischargingata', 'dischargingatd']

    columnsPresentFrom = [timeColumns[col] for col in list(timeColumns) if col in columnsCompleteFrom]
    columnsPresentTo = [timeColumns[col] for col in list(timeColumns) if col in columnsCompleteTo]

    selectColumnFrom = list(D_mov.columns)
    for col in [locto, *columnsPresentTo]:
        if col in selectColumnFrom:
            selectColumnFrom.remove(col)

    selectColumnTo = list(D_mov.columns)
    for col in [locfrom, *columnsPresentFrom]:
        if col in selectColumnTo:
            selectColumnTo.remove(col)

    # identify which coluns are present, and how to rename them
    allcolumnstorename = {'loadingpta': 'PTA',
                          'loadingptd': 'PTD',
                          'loadingata': 'ATA',
                          'loadingatd': 'ATD',
                          'dischargingpta': 'PTA',
                          'dischargingptd': 'PTD',
                          'dischargingata': 'ATA',
                          'dischargingatd': 'ATD'}

    renameDictionarycomplete = {locto: 'Location',
                                locfrom: 'Location'
                                }
    for col in allcolumnstorename.keys():
        if col in timeColumns.keys():
            renameDictionarycomplete[timeColumns[col]] = allcolumnstorename[col]

    # split and rename movements
    D1 = D_mov[selectColumnFrom]
    D1 = D1.rename(columns=renameDictionarycomplete)
    D1['InOut'] = 'IN'

    D2 = D_mov[selectColumnTo]
    D2 = D2.rename(columns=renameDictionarycomplete)
    D2['InOut'] = 'OUT'

    # Create movements table
    D = pd.concat([D1, D2])

    # Assign quantities and sign to the movements
    MovimentiIN = (D.InOut == 'IN') * 1
    MovimentiOUT = (D.InOut == 'OUT') * (-1)
    D['Movementquantity'] = MovimentiIN + MovimentiOUT
    D['Movementquantity'] = D.Movementquantity * D[capacityField]

    return D


def defineRouteTable(D: pd.DataFrame, agregationVariables: list = ['VEHICLE_CODE', 'VOYAGE_CODE'],
                     actual: str = 'PROVISIONAL'):
    """
    Define the route of a vessel, given its movements

    Args:
        D (pd.DataFrame): Input pandas dataFrame.
        agregationVariables (list, optional): Key of the route. Defaults to ['VEHICLE_CODE', 'VOYAGE_CODE'].
        actual (str, optional): If "ACTUAL" use the actual timestamp to define the route. Defaults to 'PROVISIONAL'.

    Returns:
        D_route (pd.DataFRame): Output dataframe with the route.
        timestartfield (str): Column name containing start time.
        timeendfield (str): Column name containing end time.

    """
    # import a dataframe D containing movements and defines a route dataframe
    print("**** DEFINE ROUTES  ****")
    aggregation_dictionary = {'Movementquantity': np.sum}
    if actual == 'PROVISIONAL':
        listCol = [*agregationVariables, 'Location', 'PTA', 'PTD', 'Movementquantity', '_id']
    elif actual == 'ACTUAL':
        listCol = [*agregationVariables, 'Location', 'ATA', 'ATD', 'Movementquantity', '_id']
    aggregation_columns = [col for col in D.columns if col not in listCol]
    for col in aggregation_columns:
        aggregation_dictionary[col] = lambda group_series: list(set(group_series.tolist()))

    # remove columns eventually containing dict
    # listKeys = aggregation_dictionary.keys()
    for col in list(aggregation_dictionary):
        if any([isinstance(i, dict) for i in D[col]]):
            print(col)
            aggregation_dictionary.pop(col)

    # Infer the actual route
    if actual == 'PROVISIONAL':
        D_route = D.groupby([*agregationVariables, 'Location', 'PTA', 'PTD']).agg(aggregation_dictionary).reset_index()
        timestartfield = 'PTA'
        timeendfield = 'PTD'

    elif actual == 'ACTUAL':
        D_route = D.groupby([*agregationVariables, 'Location', 'ATA', 'ATD']).agg(aggregation_dictionary).reset_index()
        timestartfield = 'ATA'
        timeendfield = 'ATD'
    return D_route, timestartfield, timeendfield


def voyageStatistics(D_mov: pd.DataFrame,
                     timefield: str = 'TIMESTAMP_IN',
                     locfrom: str = 'LOADING_NODE',
                     locto: str = 'DISCHARGING_NODE',
                     timeColumns: dict = {},
                     capacityField: str = 'QUANTITY',
                     censoredData: bool = False,
                     voyagefield: str = 'VOYAGE_CODE',
                     actual: str = 'PROVISIONAL'):
    """
    Estimate inventory values of the vessels on a route

    Args:
        D_mov (pd.DataFrame): Input dataframe.
        timefield (str, optional): column name with time. Defaults to 'TIMESTAMP_IN'.
        locfrom (str, optional): Column name with origin node. Defaults to 'LOADING_NODE'.
        locto (str, optional): column name with destination node. Defaults to 'DISCHARGING_NODE'.
        timeColumns (dict, optional): Dict with time windows columns names. Defaults to {}.
        capacityField (str, optional): Column name with transported quantities. Defaults to 'QUANTITY'.
        censoredData (bool, optional): If ture, considers censored data. Defaults to False.
        voyagefield (str, optional): Column name containing voyage code. Defaults to 'VOYAGE_CODE'.
        actual (str, optional): If "ACTUAL" consider actual timestamps. Otherwise, provisional. Defaults to 'PROVISIONAL'.

    Returns:
        D_route (pd.DataFrame): Output dataFrame with route.
        D_arcs_route (pd.DataFrame): Output dataFrame with route arcs.
        D_coverages (pd.DataFrame): Output dataFrame with statistical coverages.

    """

    # Initialise empty dataframes
    D_route = D_arcs_route = D_coverages = pd.DataFrame()

    # Calculate coverages and availability of data
    if actual == 'PROVISIONAL':
        colonneNecessarie = ['loadingpta', 'loadingptd', 'dischargingpta', 'dischargingptd']
        if all([column in timeColumns.keys() for column in colonneNecessarie]):
            allcolumns = [locfrom, locto, timeColumns['loadingpta'], timeColumns['loadingptd'], timeColumns['dischargingpta'], timeColumns['dischargingptd']]
            accuracy, _ = getCoverageStats(D_mov, analysisFieldList=allcolumns, capacityField='QUANTITY')
        else:
            colonneMancanti = [column for column in colonneNecessarie if column not in timeColumns.keys()]
            D_coverages = pd.DataFrame([f"NO columns {colonneMancanti} in timeColumns"])
    elif actual == 'ACTUAL':
        colonneNecessarie = ['loadingata', 'loadingatd', 'dischargingata', 'dischargingatd']
        if all([column in timeColumns.keys() for column in colonneNecessarie]):
            allcolumns = [locfrom, locto, timeColumns['loadingata'], timeColumns['loadingatd'], timeColumns['dischargingata'], timeColumns['dischargingatd']]
            accuracy, _ = getCoverageStats(D_mov, analysisFieldList=allcolumns, capacityField='QUANTITY')
        else:
            colonneMancanti = [column for column in colonneNecessarie if column not in timeColumns.keys()]
            D_coverages = pd.DataFrame([f"NO columns {colonneMancanti} in timeColumns"])
    # Assign accuracy
    D_coverages = pd.DataFrame(accuracy)

    D_arcs_route = pd.DataFrame()

    D = createTabellaMovimenti(D_mov=D_mov,
                               locfrom=locfrom,
                               locto=locto,
                               capacityField=capacityField,
                               timeColumns=timeColumns)

    # define routes
    D_route, timestartfield, timeendfield = defineRouteTable(D, agregationVariables=[voyagefield],
                                                             actual=actual)

    # identify voyages
    Voyages = np.unique(D_route[voyagefield])

    # identify first planning day
    firstPlanningDay = min(D_mov[timefield].dt.date)

    # Identify advance in planning
    _, df_advance = getAdvanceInPlanning(D_mov, loadingptafield=timeColumns['loadingpta'])
    mean_advanceInPlanning = df_advance.loc['ADVANCE_PLANNING_MEAN']['VALUE']
    std_advanceInPlanning = df_advance.loc['ADVANCE_PLANNING_STD']['VALUE']
    lowerBoundDataCensored = firstPlanningDay + pd.Timedelta(days=(mean_advanceInPlanning + std_advanceInPlanning))

    # Identify last planning day
    lastPlanningDay = max(D_mov[timefield].dt.date)

    # remove movements outside the reference time horizon
    if(not(censoredData)):  # if avoiding censored data
        D_route = D_route[(D_route[timestartfield] > pd.to_datetime(lowerBoundDataCensored)) & (D_route[timeendfield] < pd.to_datetime(lastPlanningDay))]
        D_route = D_route.reset_index(drop=True)

    # go on only if there are not censored data
    if len(D_route) == 0:
        D_route = pd.DataFrame(["No uncensored data"])
        return D_route, D_arcs_route, D_coverages
    D_route['inventory'] = np.nan

    print("**** INVENTORY ESTIMATE  ****")
    # scan each single voyage identifying the residual capacity
    for i in range(0, len(Voyages)):
        # i=0
        voyage = Voyages[i]
        route = D_route[D_route[voyagefield] == voyage]

        print(f"==estimate inventory voyage {voyage}, with {len(route)} movements")
        # if a route is given
        if len(route) > 0:

            # sort by time
            route = route.sort_values([timeendfield])

            # define planned movements
            counter = 0
            allIndex = []  # define a list of indexes
            for index, row in route.iterrows():  # same indexes of D_route
                if counter == 0:
                    D_route['inventory'].loc[index] = row['Movementquantity']
                    allIndex.append(index)

                else:
                    D_route['inventory'].loc[index] = row['Movementquantity'] + D_route['inventory'].loc[allIndex[counter - 1]]
                    allIndex.append(index)
                counter = counter + 1

            # calculate the estimate of the capacity and move on positive values (above zero)
            allCapacities = D_route[D_route[voyagefield] == voyage]['inventory']
            slack = np.double(- min(allCapacities))
            D_route['inventory'].loc[allIndex] = D_route[D_route[voyagefield] == voyage]['inventory'] + slack
            capMax = max(D_route['inventory'].loc[allIndex])

            # assign route value on the dataframe
            route = D_route[D_route[voyagefield] == voyage]
            route = route.sort_values([timeendfield])

            # scan the route to define from-to movement dataframe
            for k in range(0, len(route) - 1):
                # k=0

                # identify current and following movement
                currentMovement = route.iloc[k]
                nextMovement = route.iloc[k + 1]

                rowDictionary = {'arcFrom': currentMovement.Location,
                                 'arcTo': nextMovement.Location,
                                 'departureFromALAP': currentMovement[timeendfield],
                                 'arrivalToASAP': nextMovement[timestartfield],
                                 'inventory': currentMovement.inventory,
                                 'capacity': capMax - currentMovement.inventory}
                # append all the other FROM
                add_columns_from = [col for col in currentMovement.index if col not in ['Location', 'timeendfield', 'inventory']]
                for col in add_columns_from:
                    rowDictionary[f"{col}_from"] = currentMovement[col]

                # append all the other TO
                add_columns_to = [col for col in nextMovement.index if col not in ['Location', 'timestartfield', 'inventory']]
                for col in add_columns_to:
                    rowDictionary[f"{col}_to"] = nextMovement[col]

                # add restults to the final dataframe
                D_arcs_route = D_arcs_route.append(pd.DataFrame([rowDictionary]))

    return D_route, D_arcs_route, D_coverages


def returnFigureVoyage(D_route: pd.DataFrame, D_arcs_route: pd.DataFrame, lastPlanningDay: list = [],
                       lowerBoundDataCensored: list = [], filteringfield: str = 'VOYAGE_CODE', sortTimefield: str = 'PTD'):
    """
    Create a dictionary of figures with  a chart plot of the routes

    Args:
        D_route (pd.dataFrame): Input route dataframe.
        D_arcs_route (pd.DataFrame): Input route dataframe with arcs.
        lastPlanningDay (list, optional): DESCRIPTION. Defaults to [].
        lowerBoundDataCensored (list, optional): DESCRIPTION. Defaults to [].
        filteringfield (str, optional): DESCRIPTION. Defaults to 'VOYAGE_CODE'.
        sortTimefield (str, optional): column name to sort values by. Defaults to 'PTD'.

    Returns:
        figure_results (dict): Output dictionary containing figures.

    """

    figure_results = {}
    for voyage in set(D_route[filteringfield]):
        # voyage='Vessel 10'
        # generate inventory dataframe
        D_plannedRouteVessel = D_route[D_route[filteringfield] == voyage]

        if len(D_plannedRouteVessel) > 0:
            D_plannedRouteVessel = D_plannedRouteVessel.sort_values(by=sortTimefield)

            # Create capacity charts
            figure = plt.figure(figsize=(20, 10))
            plt.step(D_plannedRouteVessel[sortTimefield], D_plannedRouteVessel['inventory'], color='orange')
            plt.title(str(voyage) + ' inventory')
            plt.xticks(rotation=30)

            # track the capacity
            capMax = max(D_plannedRouteVessel['inventory'])
            plt.plot(D_plannedRouteVessel[sortTimefield], [capMax] * len(D_plannedRouteVessel), 'r--')
            plt.axvline(x=lastPlanningDay, color='red', linestyle='--')
            plt.axvline(x=lowerBoundDataCensored, color='red', linestyle='--')
            figure_results[f"{filteringfield}_{voyage}_inventory"] = figure

        # generate graph chart
        D_plannedRouteVessel_fromTo = D_arcs_route[D_arcs_route[f"{filteringfield}_from"] == voyage]

        if len(D_plannedRouteVessel_fromTo) > 0:
            # plot routes on a graph
            FlowAnalysis = D_plannedRouteVessel_fromTo.groupby(['arcFrom', 'arcTo']).size().reset_index()
            FlowAnalysis = FlowAnalysis.rename(columns={0: 'Trips'})

            fig1 = plotGraph(df=FlowAnalysis,
                             edgeFrom='arcFrom',
                             edgeTo='arcTo',
                             distance='Trips',
                             weight='Trips',
                             title=str(voyage),
                             arcLabel=True)
            figure_results[f"{filteringfield}_{voyage}_graph"] = fig1

    return figure_results


def graphClock(D_mov: pd.DataFrame,
               loadingNode: str = 'LOADING_NODE',
               dischargingNode: str = 'DISCHARGING_NODE',
               sortingField: str = 'PTA_FROM',
               vehicle: str = 'VEHICLE_CODE',
               capacityField: str = 'QUANTITY',
               timeColumns: str = {},
               actual: str = 'PROVISIONAL'):
    """
    Train style chart with time and space

    Args:
        D_mov (pd.DataFrame): Input pandas DataFrame.
        loadingNode (str, optional): Column name containing origin node. Defaults to 'LOADING_NODE'.
        dischargingNode (str, optional): Column name containing destination node. Defaults to 'DISCHARGING_NODE'.
        sortingField (str, optional): Column name containing attribute to sort by. Defaults to 'PTA_FROM'.
        vehicle (str, optional): Column name containing vehicle code. Defaults to 'VEHICLE_CODE'.
        capacityField (str, optional): Column name containing transported quantity. Defaults to 'QUANTITY'.
        timeColumns (str, optional): dict of column names containing time windows. Defaults to {}.
        actual (str, optional): If "ACTUAL" use actual timestamps, otherwise, "PROVISIONAL". Defaults to 'PROVISIONAL'.

    Returns:
        output_figure (TYPE): DESCRIPTION.
        output_df (TYPE): DESCRIPTION.

    """

    output_figure = {}
    output_df = {}

    # identify necessary columns and calculate coverages
    if actual == 'PROVISIONAL':
        colonneNecessarie = ['loadingptd', 'dischargingpta']
        if all([column in timeColumns.keys() for column in colonneNecessarie]):
            allcolumns = [loadingNode, dischargingNode, vehicle, timeColumns['loadingptd'], timeColumns['dischargingpta']]
            accuracy, _ = getCoverageStats(D_mov, analysisFieldList=allcolumns, capacityField='QUANTITY')
        else:
            colonneMancanti = [column for column in colonneNecessarie if column not in timeColumns.keys()]
            output_df['coverages'] = pd.DataFrame([f"NO columns {colonneMancanti} in timeColumns"])
            return output_figure, output_df
    elif actual == 'ACTUAL':
        colonneNecessarie = ['loadingatd', 'dischargingata']
        if all([column in timeColumns.keys() for column in colonneNecessarie]):
            allcolumns = [loadingNode, dischargingNode, vehicle, timeColumns['loadingatd'], timeColumns['dischargingata']]
            accuracy, _ = getCoverageStats(D_mov, analysisFieldList=allcolumns, capacityField='QUANTITY')
    output_df[f"coverages_{actual}"] = pd.DataFrame(accuracy)

    # identify all terminals
    terminal_dict = {}
    D_mov = D_mov.sort_values(by=[sortingField])
    terminals = list(set([*D_mov[loadingNode], *D_mov[dischargingNode]]))
    for i in range(0, len(terminals)):
        terminal_dict[terminals[i]] = i

    # identify movements
    D = createTabellaMovimenti(D_mov,
                               locfrom=loadingNode,
                               locto=dischargingNode,
                               capacityField=capacityField,
                               timeColumns=timeColumns
                               )
    D_route, timestartfield, timeendfield = defineRouteTable(D,
                                                             agregationVariables=[vehicle],
                                                             actual=actual)

    for vessel in set(D_route[vehicle]):
        D_mov_filtered = D_route[D_mov[vehicle] == vessel]
        D_mov_filtered = D_route.sort_values(by=timestartfield)
        D_mov_filtered = D_mov_filtered.dropna(subset=[timestartfield, 'Location'])

        # create graph on a temporal axis
        fig1 = plt.figure()
        for i in range(1, len(D_mov_filtered)):

            # plot voyage
            x_array = [D_mov_filtered[timeendfield].iloc[i - 1], D_mov_filtered[timestartfield].iloc[i]]
            y_array = [terminal_dict[D_mov_filtered['Location'].iloc[i - 1]], terminal_dict[D_mov_filtered['Location'].iloc[i]]]
            plt.plot(x_array, y_array, color='orange', marker='o')

            # plot waiting time
            x_array = [D_mov_filtered[timestartfield].iloc[i], D_mov_filtered[timeendfield].iloc[i]]
            y_array = [terminal_dict[D_mov_filtered['Location'].iloc[i]], terminal_dict[D_mov_filtered['Location'].iloc[i]]]
            plt.plot(x_array, y_array, color='orange', marker='o')
        plt.ylabel('Terminal')
        plt.xlabel('Time')
        output_figure[f"Train_chart_{vessel}_{actual}"] = fig1

        if actual == 'PROVISIONAL':
            time_from = timeColumns['loadingptd']
            time_to = timeColumns['dischargingpta']
        elif actual == 'ACTUAL':
            time_from = timeColumns['loadingatd']
            time_to = timeColumns['dischargingata']

        # plot graph grouping on a day
        D_train = D_mov[D_mov[vehicle] == vessel]
        D_train['hour_from'] = D_train[time_from].dt.time
        D_train['hour_to'] = D_train[time_to].dt.time
        D_graph = D_train.groupby([loadingNode, dischargingNode, 'hour_from', 'hour_to']).sum()[capacityField].reset_index()
        D_graph = D_graph.sort_values(by=capacityField, ascending=False)
        fig1 = plt.figure()
        for i in range(0, len(D_graph)):
            x_array = [D_graph['hour_from'].iloc[i], D_graph['hour_to'].iloc[i]]
            y_array = [terminal_dict[D_graph[loadingNode].iloc[i]], terminal_dict[D_graph[dischargingNode].iloc[i]]]

            my_day = datetime.date(1990, 1, 1)
            x_array = [datetime.datetime.combine(my_day, t) for t in x_array]

            plt.title(f"Train schedule chart VEHICLE: {vessel}")
            plt.plot(x_array, y_array, color='orange', marker='o', linewidth=np.log(D_graph[capacityField].iloc[i]))
        output_figure[f"Train_chart_daily_{vessel}_{actual}"] = fig1
        plt.close('all')
    return output_figure, output_df
