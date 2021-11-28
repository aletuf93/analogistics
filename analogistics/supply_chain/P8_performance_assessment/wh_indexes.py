# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analogistics.statistics import time_series as ts
from analogistics.supply_chain.information_framework import movementfunctionfromInventory
from analogistics.explore import paretoDataframe


def calculatePopularity(movements: pd.Series):
    """
    Define the popularity for a SKU

    Args:
        movements (pd.Series): series of the movement with one item per day.

    Returns:
        pop_in (float): relative popularity IN per day.
        pop_out (float): realtiva popularity OUT per day.
        pop_absolute_in (float): popularity IN per day.
        pop_absolute_out (float): popularity OUT per day.

    """

    pop_in = len(movements[movements > 0]) / len(movements)
    pop_out = len(movements[movements < 0]) / len(movements)
    pop_absolute_in = len(movements[movements > 0])
    pop_absolute_out = len(movements[movements < 0])
    return pop_in, pop_out, pop_absolute_in, pop_absolute_out


def calculateCOI(inventory: pd.Series):
    """
    Calculate the COI index of an SKU, given the inventory function

    Args:
        inventory (pd.Series): series of the inventory of an SKU.

    Returns:
        COI_in (float): COI index IN.
        COI_out (float): COI index OUT.

    """

    # define inventory from movements
    movements = movementfunctionfromInventory(inventory)
    movements = movements.dropna()
    pop_in, pop_out, _, _ = calculatePopularity(movements['QUANTITY'])

    # calculate daily COI
    I_t_avg = np.nanmean(inventory)
    if I_t_avg > 0:
        COI_in = pop_in / I_t_avg
        COI_out = pop_out / I_t_avg
    else:
        COI_in = COI_out = np.nan

    return COI_in, COI_out


def calculateTurn(inventory: pd.Series):
    """
    Calculate the TURN index of an SKU, given the inventory function

    Args:
        inventory (pd.series): series of the inventory of an SKU.

    Returns:
        turn (float): Output turn index.
    """

    # define inventory from movements
    movements = movementfunctionfromInventory(inventory)
    movements = movements.dropna()

    # calculate the average outbound quantity per day
    out_qty_day = -np.sum(movements[movements['QUANTITY'] < 0]['QUANTITY']) / len(movements)

    # calculate average inventory quantity
    I_t_avg = np.nanmean(inventory)
    if I_t_avg > 0:
        turn = out_qty_day / I_t_avg
    else:
        turn = np.nan

    return turn


def calculateOrderCompletion(D_mov: pd.DataFrame, itemcode: str,
                             itemfield: str = 'ITEMCODE', ordercodefield: str = 'ORDERCODE'):
    """
    calculate the Order Completion (OC) index

    Args:
        D_mov (pd.DataFrame): dataframe with movements reporting ordercode and itemcode columns.
        itemcode (str): itemcode to calculate the order competion (OC) index.
        itemfield (str, optional): string name of D_mov clumn with itemcode. Defaults to 'ITEMCODE'.
        ordercodefield (str, optional): string name of D_mov clumn with ordercode. Defaults to 'ORDERCODE'.

    Returns:
        OC (float): Output OC index.

    """

    # clean data
    D_mov = D_mov[[itemfield, ordercodefield]]
    D_mov = D_mov[D_mov[ordercodefield] != 'nan']
    D_mov = D_mov.dropna()
    D_mov = D_mov.reset_index()

    orders = list(set(D_mov[D_mov[itemfield] == itemcode][ordercodefield]))

    idx = [j in orders for j in D_mov[ordercodefield]]
    D_orders = D_mov.loc[idx]

    OC = 0
    for ordercode in orders:
        D_orders_filtered = D_orders[D_orders[ordercodefield] == ordercode]
        OC = OC + 1 / len(D_orders_filtered)
    return OC


def fourierAnalysisInventory(inventory: pd.Series):
    """
    fourier analysis of the inventory curve

    Args:
        inventory (pd.series): list of inventory values.

    Returns:
        first_carrier (TYPE): frequency (in 1/days) with the highest amplitude value.
        period (TYPE): period (in days) associated with the frequency with the highest amplitude value.

    """

    D = ts.fourierAnalysis(np.array(inventory))
    D = D.sort_values(by='Amplitude', ascending=False)
    first_carrier = D.iloc[0]['Frequency_domain_value']  # 1 / days
    period = 1 / first_carrier
    return first_carrier, period


def updatePopularity(D_SKUs: pd.DataFrame):
    """
    Update the popularity index

    Args:
        D_SKUs (pd.dataFrame): Input dataframe with SKUs.

    Returns:
        D_SKUs (pd.DataFrame): Output DataFrame with updated popularity.

    """

    # create results columns
    D_SKUs['POP_IN'] = np.nan
    D_SKUs['POP_OUT'] = np.nan
    D_SKUs['POP_IN_TOT'] = np.nan
    D_SKUs['POP_OUT_TOT'] = np.nan

    for index, row in D_SKUs.iterrows():
        # select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        # calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements = movements.dropna()
        if len(movements) > 0:
            POP_IN, POP_OUT, POP_IN_TOT, POP_OUT_TOT = calculatePopularity(movements['QUANTITY'])

            # update the dataframe
            D_SKUs.at[index, 'POP_IN'] = POP_IN
            D_SKUs.at[index, 'POP_OUT'] = POP_OUT
            D_SKUs.at[index, 'POP_IN_TOT'] = POP_IN_TOT
            D_SKUs.at[index, 'POP_OUT_TOT'] = POP_OUT_TOT
    return D_SKUs


def updateCOI(D_SKUs: pd.DataFrame):
    """
    Update the COI index

    Args:
        D_SKUs (pd.DataFrame): Input dataframe with SKUs.

    Returns:
        D_SKUs (pd.DataFrame): Output DataFrame with updated COI.

    """

    # create result columns
    D_SKUs['COI_IN'] = np.nan
    D_SKUs['COI_OUT'] = np.nan
    for index, row in D_SKUs.iterrows():
        # select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        # calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements = movements.dropna()
        if len(movements) > 0:
            COI_IN, COI_OUT = calculateCOI(I_t)

            # update the dataframe
            D_SKUs.at[index, 'COI_IN'] = COI_IN
            D_SKUs.at[index, 'COI_OUT'] = COI_OUT

    return D_SKUs


def updateTURN(D_SKUs: pd.DataFrame):
    """
    Update TURN index

    Args:
        D_SKUs (pd.DataFrame): Input dataframe with SKUs.

    Returns:
        D_SKUs (TYPE): Output DataFrame with updated TURN.

    """

    # create result columns
    D_SKUs['TURN'] = np.nan

    for index, row in D_SKUs.iterrows():
        # select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        # calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements = movements.dropna()
        if len(movements) > 0:
            TURN = calculateTurn(I_t)

            # update the dataframe
            D_SKUs.at[index, 'TURN'] = TURN

    return D_SKUs


def updateOrderCompletion(D_SKUs: pd.DataFrame, D_mov: pd.DataFrame):
    """
    Update OC index

    Args:
        D_SKUs (pd.DataFrame): Input dataframe with SKUs.
        D_mov (pd.DataFrame): Input dataframe with movements.

    Returns:
        D_SKUs (pd.dataFrame): Output DataFrame with updated OC.

    """

    # create result columns
    D_SKUs['OC'] = np.nan

    for index, row in D_SKUs.iterrows():

        part = row['ITEMCODE']

        # calculate the popularity
        OC = calculateOrderCompletion(D_mov, part, itemfield='ITEMCODE', ordercodefield='ORDERCODE')

        # update the dataframe
        D_SKUs.at[index, 'OC'] = OC

    return D_SKUs


def updateFourieranalysis(D_SKUs: pd.DataFrame):
    """
    Update the Fourier Analysis

    Args:
        D_SKUs (pd.DataFrame): Input dataframe with SKUs.

    Returns:
        D_SKUs (pd.DataFrame): Output DataFrame with updated fourier analysis.

    """

    # create result columns
    D_SKUs['FOURIER_CARRIER'] = np.nan
    D_SKUs['FOURIER_PERIOD'] = np.nan

    for index, row in D_SKUs.iterrows():
        # select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        # calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements = movements.dropna()
        if len(movements) > 0:
            carrier, period = fourierAnalysisInventory(I_t)

            # update the dataframe
            D_SKUs.at[index, 'FOURIER_CARRIER'] = carrier
            D_SKUs.at[index, 'FOURIER_PERIOD'] = period

    return D_SKUs


# %% PARETO AND HISTOGRAM PLOT

def whIndexParetoPlot(D_SKUs: pd.DataFrame, columnIndex: str):
    """
    Define the Pareto and histogram plot for a WH index

    Args:
        D_SKUs (pd.DataFrame): Input dataframe with SKUs.
        columnIndex (str): Name of the index to plot.

    Returns:
        output_figures (dict): Output dictionary with figures.

    """

    output_figures = {}

    # define the pareto values
    D_SKUs_pop = paretoDataframe(D_SKUs, columnIndex)

    # build the pareto figures
    fig1 = plt.figure()
    plt.plot(np.arange(0, len(D_SKUs_pop)), D_SKUs_pop[f"{columnIndex}_CUM"], color='orange')
    plt.title(f"{columnIndex} Pareto curve")
    plt.xlabel("N. of SKUs")
    plt.ylabel("Popularity percentage")

    # save the Pareto figure
    output_figures[f"{columnIndex}_pareto"] = fig1

    fig2 = plt.figure()
    plt.hist(D_SKUs_pop[columnIndex], color='orange')
    plt.title(f"{columnIndex} histogram")
    plt.xlabel(f"{columnIndex}")
    plt.ylabel("Frequency")

    # save the Pareto figure
    output_figures[f"{columnIndex}_hist"] = fig2

    return output_figures
