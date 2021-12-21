import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def defineFromToTable(D_flows: pd.DataFrame, colFrom: str, colTo: str, colQty: str):
    """
    Generate a from-to table from a flows table

    Args:
        D_flows (pd.DataFrame): pandas dataframe containing the flows.
        colFrom (str): column name containing the origin point of the flow.
        colTo (str): column name containing the destination point of the flow.
        colQty (str): column name containing the entity of each flow.

    Returns:
        output_dataframe (dict): dictionary of pandas dataframe with the from-to (flow count and quantity).
        output_figures (dict): dictionary of figures with the heatmaps (flow count and quantity).

    """

    output_dataframe = {}
    output_figures = {}

    D_flows = D_flows.groupby([colFrom, colTo]).agg({colQty: ['sum', 'size']}).reset_index()
    D_flows.columns = [colFrom, colTo, 'flow_qty', 'flow_count']

    # % create the from-to dataframe
    columns = list(set(D_flows[colFrom]).union(D_flows[colTo]))
    columns = sorted(columns)

    df_fromTo_quantity = pd.DataFrame(index=columns, columns=columns)

    df_fromTo_count = pd.DataFrame(index=columns, columns=columns)

    for index, row in D_flows.iterrows():
        df_fromTo_count.at[row[colFrom], row[colTo]] = row['flow_count']
        df_fromTo_quantity.at[row[colFrom], row[colTo]] = row['flow_qty']

    df_fromTo_count = df_fromTo_count.fillna(0)
    df_fromTo_quantity = df_fromTo_quantity.fillna(0)

    # generate heatmap
    fig_count = plt.figure()
    plt.imshow(df_fromTo_count, cmap=plt.get_cmap('plasma'))
    plt.yticks(np.arange(0.5, len(df_fromTo_count.index), 1), df_fromTo_count.index, rotation=45)
    plt.xticks(np.arange(0.5, len(df_fromTo_count.columns), 1), df_fromTo_count.columns, rotation=45)
    plt.title("From-to matrix flows count")
    plt.xlabel('TO')  # dataframe columns
    plt.ylabel('FROM')  # dataframe rows
    plt.colorbar()
    plt.show()

    fig_quantity = plt.figure()
    plt.imshow(df_fromTo_quantity, cmap=plt.get_cmap('plasma'))
    plt.yticks(np.arange(0.5, len(df_fromTo_quantity.index), 1), df_fromTo_quantity.index, rotation=45)
    plt.xticks(np.arange(0.5, len(df_fromTo_quantity.columns), 1), df_fromTo_quantity.columns, rotation=45)
    plt.title("From-to matrix flows quantities")
    plt.xlabel('TO')  # dataframe columns
    plt.ylabel('FROM')  # dataframe rows
    plt.colorbar()
    plt.show()

    output_dataframe['fromToQuantity'] = df_fromTo_quantity
    output_dataframe['fromToCount'] = df_fromTo_count
    output_figures['fromToHeatmapQuantity'] = fig_quantity
    output_figures['fromToHeatmapCount'] = fig_count
    return output_dataframe, output_figures
