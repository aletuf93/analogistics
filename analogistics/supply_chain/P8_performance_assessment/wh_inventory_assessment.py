
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def updateGlobalInventory(D_SKUs: pd.DataFrame, inventoryColumn: str):
    """
    Update the global inventory of the warehouse

    Args:
        D_SKUs (pd.DataFrame): Input SKUs dataframe.
        inventoryColumn (str): column name with the inventory.

    Returns:
        D_inventory (pd.DataFrame): Output DataFrame with inventory values.

    """

    D_inventory = pd.DataFrame([], columns=['WH_INVENTORY_VOLUME', 'WH_INVENTORY_NORMALISED'])
    givenVolumes = 0  # count the number of SKUs with a given volume
    for i in range(0, len(D_SKUs)):
        # i=33159
        volume = D_SKUs.iloc[i]['VOLUME']
        list_days = D_SKUs.iloc[i]['INVENTORY_DAYS']

        # go on only if an inventory has been saved
        if isinstance(list_days, list):
            list_inventory = np.array(D_SKUs.iloc[i][inventoryColumn])
            list_inventory = np.nan_to_num(list_inventory)  # convert nan to 0
            list_inventory_volume = list_inventory * volume
            list_inventory_normalised = (list_inventory - min(list_inventory)) / (max(list_inventory) - min(list_inventory))

            D_temp = pd.DataFrame(list_inventory_normalised, index=list_days, columns=['SKU_INVENTORY_NORMALISED'])
            D_inventory = pd.concat([D_temp, D_inventory], axis=1, sort=False)
            D_inventory = D_inventory.fillna(0)
            D_inventory['WH_INVENTORY_NORMALISED'] = D_inventory['WH_INVENTORY_NORMALISED'] + D_inventory['SKU_INVENTORY_NORMALISED']
            D_inventory = D_inventory.drop(columns=['SKU_INVENTORY_NORMALISED'])

            if str(volume) != 'nan':  # if volume is not nan
                D_temp = pd.DataFrame(list_inventory_volume, index=list_days, columns=['SKU_INVENTORY_VOLUME'])
                D_inventory = pd.concat([D_temp, D_inventory], axis=1, sort=False)
                D_inventory = D_inventory.fillna(0)
                D_inventory['WH_INVENTORY_VOLUME'] = D_inventory['WH_INVENTORY_VOLUME'] + D_inventory['SKU_INVENTORY_VOLUME']
                D_inventory = D_inventory.drop(columns=['SKU_INVENTORY_VOLUME'])
                givenVolumes = givenVolumes + 1

    return D_inventory


def _cumulativeFunction(ser: pd.Series):
    ser = ser.sort_values()
    cum_dist = np.linspace(0., 1., len(ser))
    ser_cdf = pd.Series(cum_dist, index=ser)
    return ser_cdf


def inventoryAnalysis(D_global_inventory: pd.DataFrame):
    """
    Perform analysis on the inventory values

    Args:
        D_global_inventory (pd.DataFrame): Input Dataframe with inventory values.

    Returns:
        dict: Output dict containing figures.

    """

    output_figures = {}

    # plot histogram
    fig1 = plt.figure()
    plt.hist(D_global_inventory['WH_INVENTORY_VOLUME'], color='orange')
    plt.xlabel('Inventory values')
    plt.ylabel('Frequency')
    plt.title('Inventory histogram')

    output_figures['INVENTORY_HIST'] = fig1

    fig2 = plt.figure()
    plt.hist(D_global_inventory['WH_INVENTORY_NORMALISED'], color='orange')
    plt.xlabel('Normalised Inventory values')
    plt.ylabel('Frequency')
    plt.title('Normalised inventory histogram')
    output_figures['INVENTORY_NORM_HIST'] = fig2

    # plot trend
    fig3 = plt.figure()
    plt.plot(D_global_inventory.index, D_global_inventory['WH_INVENTORY_VOLUME'], color='orange')
    plt.xlabel('Timeline')
    plt.ylabel('Inventory values')
    plt.title('Inventory time series')
    plt.xticks(rotation=45)
    output_figures['INVENTORY_TS'] = fig3

    fig4 = plt.figure()
    plt.plot(D_global_inventory.index, D_global_inventory['WH_INVENTORY_NORMALISED'], color='orange')
    plt.xlabel('Timeline')
    plt.ylabel('Normalised inventory values')
    plt.title('Normalised inventory time series')
    plt.xticks(rotation=45)
    output_figures['INVENTORY_NORM_TS'] = fig4

    # cumulative function
    cdf_inventory = _cumulativeFunction(D_global_inventory['WH_INVENTORY_NORMALISED'])
    fig5 = plt.figure()
    cdf_inventory.plot(drawstyle='steps', color='orange')
    plt.xlabel('Normalised inventory values')
    plt.ylabel('Probability')
    plt.title('Normalised inventory cumulative probability function')
    output_figures['INVENTORY_NORM_CUM'] = fig5

    cdf_inventory = _cumulativeFunction(D_global_inventory['WH_INVENTORY_VOLUME'])
    fig6 = plt.figure()
    cdf_inventory.plot(drawstyle='steps', color='orange')
    plt.xlabel('Inventory values')
    plt.ylabel('Probability')
    plt.title('Inventory cumulative probability function')
    output_figures['INVENTORY_CUM'] = fig6
    return output_figures


def defineStockoutCurve(inventorySeries: pd.DataFrame):
    """
    Define the stockout risk curve

    Args:
        inventorySeries (pd.DataFrame): Input Dataframe with inventory values.

    Returns:
        output_figure (TYPE): Output dict containing figures.

    """

    output_figure = {}

    # calculate the cumulative and the risk
    cumulative = _cumulativeFunction(inventorySeries)
    risk = 1 - cumulative

    # plot the curve
    fig1 = plt.figure()
    plt.plot(cumulative.index, cumulative.values, drawstyle='steps', color='skyblue')
    plt.plot(risk.index, risk.values, drawstyle='steps', color='orange')
    plt.legend(['Cumulative distribution function', 'Stockout risk'])
    plt.title("Stockout risk function")
    plt.xlabel("Inventory value")
    plt.ylabel("Risk or probability")

    output_figure['stockout curve'] = fig1

    return output_figure
