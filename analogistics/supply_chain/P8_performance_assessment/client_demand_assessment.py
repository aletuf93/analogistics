# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from analogistics.explore import paretoChart
from analogistics.supply_chain.P8_performance_assessment.utilities_movements import getCoverageStats


def clientStatistics(D_mov: pd.DataFrame,
                     clientfield: str = 'KLANT',
                     itemfamily: str = 'ContainerSize',
                     capacityfield: str = 'QUANTITY'):
    """
    Produce statistics on the clients, given a DataFrame of movements

    Args:
        D_mov (pd.dataFrame): Input pandas dataframe with movements.
        clientfield (str, optional): Column name with the client code. Defaults to 'KLANT'.
        itemfamily (str, optional): Column name with the category of the items. Defaults to 'ContainerSize'.
        capacityfield (str, optional): Column name with the item quantity. Defaults to 'QUANTITY'.

    Returns:
        imageResult (dict): output dictionary containing figures.
        df_results (pd.DataFrame): output pandas dataFrames.

    """

    imageResult = {}
    df_results = pd.DataFrame()

    accuracy, _ = getCoverageStats(D_mov, clientfield, capacityField='QUANTITY')
    D_OrderPerClient = D_mov.groupby([clientfield]).size().reset_index()
    D_OrderPerClient = D_OrderPerClient.rename(columns={0: 'TotalOrders'})
    D_OrderPerClient = D_OrderPerClient.sort_values([clientfield])

    # create pie chart
    labels = D_OrderPerClient[clientfield]
    sizes = D_OrderPerClient.TotalOrders
    explode = 0.1 * np.ones(len(sizes))

    fig1, ax1 = plt.subplots(figsize=(20, 10))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    imageResult['clients_pie'] = fig1

    # Count movement type per client
    D_movTypePerClient = D_mov.groupby([clientfield, itemfamily]).size().reset_index()
    D_movTypePerClient = D_movTypePerClient.rename(columns={0: 'TotalContainer'})
    D_movTypePerClient = D_movTypePerClient.pivot(index=clientfield, columns=itemfamily, values='TotalContainer')

    D = pd.merge(D_movTypePerClient, D_OrderPerClient,
                 left_on=[clientfield], right_on=[clientfield])
    D = D.fillna(0)

    df_results = D
    # add accuracy
    df_results['accuracy'] = [accuracy for i in range(0, len(df_results))]

    # Pareto curve on capacity per client
    D_capacityPerClient = D_mov.groupby([clientfield])[capacityfield].sum().reset_index()
    fig1 = paretoChart(D_capacityPerClient, clientfield, capacityfield, 'Pareto clients')

    imageResult['paretoClient'] = fig1
    return imageResult, df_results


def paretoNodeClient(D_mov: pd.DataFrame,
                     clientfield: str = 'KLANT',
                     locationfromfield: str = 'LOADING_NODE',
                     locationtofield: str = 'DISCHARGING_NODE',
                     vehiclefield: str = 'VEHICLE_CODE',
                     capacityField: str = 'QUANTITY'
                     ):
    """
    Built a pareto cumulating the quantity loaded or offloaded by the clients
    on the different locations (i.e. the nodes) of a supply chain network

    Args:
        D_mov (pd.DataFrame): Input pandas dataframe with movements.
        clientfield (str, optional): Column name with the client code. Defaults to 'KLANT'.
        locationfromfield (str, optional): Column name with the origin location code. Defaults to 'LOADING_NODE'.
        locationtofield (str, optional): Column name with the destination location code. Defaults to 'DISCHARGING_NODE'.
        vehiclefield (str, optional): Column name with the vehicle code. Defaults to 'VEHICLE_CODE'.
        capacityField (str, optional): Column name with the transported quantity. Defaults to 'QUANTITY'.

    Returns:
        outputfigure (dict): output dictionary containing figures.
        output_df (dict): output dictionary containing dataframes.

    """
    outputfigure = {}
    output_df = {}

    # if same field, it is not possible to cumulate values -> no analysis
    if (clientfield == locationfromfield) | (clientfield == locationtofield):
        print("Same field for client and location from/to. Cannot proceed")
        return outputfigure, output_df
    for barge in set(D_mov[vehiclefield]):

        # filter dataframe
        D_clNode = D_mov[D_mov[vehiclefield] == barge]
        if len(D_clNode) > 0:
            # Calculate coverages on count and quantities
            accuracy, _ = getCoverageStats(D_clNode, [clientfield, locationfromfield,
                                                      locationtofield, vehiclefield],
                                           capacityField=capacityField)

            D_clNode_from = pd.DataFrame(D_clNode.groupby([clientfield, locationtofield]).size()).reset_index()
            D_clNode_from = D_clNode_from.rename(columns={locationtofield: 'Location'})

            D_clNode_to = pd.DataFrame(D_clNode.groupby([clientfield, locationfromfield]).size()).reset_index()
            D_clNode_to = D_clNode_to.rename(columns={locationfromfield: 'Location'})

            D_clNode_all = pd.concat([D_clNode_from, D_clNode_to], axis=0)
            D_clNode_all = D_clNode_all.sort_values(by=0, ascending=False)
            D_clNode_all = D_clNode_all.dropna()
            D_clNode_all = D_clNode_all.reset_index(drop=True)

            # delete locations already visited
            setLocation = []
            for row in D_clNode_all.iterrows():
                index = row[0]
                rr = row[1]
                if str(rr.Location).lower().strip() in setLocation:
                    D_clNode_all = D_clNode_all.drop(index)
                else:
                    setLocation.append(str(rr.Location).lower().strip())

            # add the nodes not cumulating any values
            D_clNode_all = D_clNode_all.groupby([clientfield])['Location'].nunique()
            D_clNode_all = pd.DataFrame(D_clNode_all)
            for client in set(D_clNode[clientfield]):
                if client not in D_clNode_all.index.values:
                    # print(client)
                    temp = pd.DataFrame([0], index=[client], columns=['Location'])
                    D_clNode_all = pd.concat([D_clNode_all, temp])

            D_clNode_all = pd.DataFrame(D_clNode_all)
            D_clNode_all['Client'] = D_clNode_all.index.values
            D_clNode_all['accuracy'] = [accuracy for i in range(0, len(D_clNode_all))]

            titolo = f"Vehicle Code: {barge}"
            fig = paretoChart(D_clNode_all, 'Client', 'Location', titolo)
            outputfigure[f"pareto_vehicle_{barge}"] = fig
            output_df[f"pareto_vehicle_{barge}"] = D_clNode_all
    return outputfigure, output_df


def violinPlantTerminal(D_mov: pd.DataFrame, plantField: str = 'LOADING_NODE',
                        clientField: str = 'DISCHARGING_NODE',
                        capacityField: str = 'QUANTITY'):
    """
    Build a plot with a violin for each node of the fistribution network,
    indicating the deliered wuantitied towards each client

    Args:
        D_mov (pd.dataFrame): Input pandas dataframe with movements.
        plantField (str, optional): Column name with the plant node code. Defaults to 'LOADING_NODE'.
        clientField (str, optional): Column name with the client code. Defaults to 'DISCHARGING_NODE'.
        capacityField (str, optional): Column name with the transported quantit. Defaults to 'QUANTITY'.

    Returns:
        output_figure (dict): output dictionary containing figures.
        output_df (dict): output dictionary containing dataframes.

    """

    output_figure = {}
    output_df = {}

    accuracy, _ = getCoverageStats(D_mov, [clientField, plantField], capacityField=capacityField)
    df_out = pd.DataFrame([accuracy])

    D_clientTerminal = D_mov.groupby([plantField, clientField]).sum()[capacityField].reset_index()

    fig = plt.figure()
    sns.violinplot(x=plantField, y=capacityField,
                   data=D_clientTerminal, palette="muted")
    output_figure['violin_plant_client'] = fig
    output_df['violin_plant_client_coverages'] = df_out

    return output_figure, output_df
