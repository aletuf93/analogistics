# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def returnInventoryRiskFromInventoryFunction(inventory_values: list, inventory: float) -> float:
    """
    return the risk probability value, given the inventory values, and the value of inventory to evaluate

    Args:
        inventory_values (list): list of float with inventory values.
        inventory (float): value of inventory.

    Returns:
        float: risk value associated with the value of inventory to evaluate.

    """

    inventory = float(inventory)
    # build empirical pdf and cdf
    D_inventory = pd.DataFrame(inventory_values, columns=['INVENTORY'])
    D_inventory = D_inventory.groupby('INVENTORY').size().to_frame().reset_index()
    D_inventory.columns = ['INVENTORY', 'FREQUENCY']
    D_inventory = D_inventory.sort_values(by=['INVENTORY'])
    D_inventory['PROB'] = D_inventory['FREQUENCY'] / sum(D_inventory['FREQUENCY'])
    D_inventory['CUMULATIVE'] = D_inventory['PROB'].cumsum()

    # calculate the inventory quantity
    if(inventory > max(inventory_values)):
        D_opt_x_max = D_inventory.iloc[-1]
    else:
        D_opt_x_max = D_inventory[D_inventory['INVENTORY'] >= inventory].iloc[0]

    if(inventory < min(inventory_values)):
        D_opt_x_min = D_inventory.iloc[0]
    else:
        D_opt_x_min = D_inventory[D_inventory['INVENTORY'] < inventory].iloc[-1]

    x_array = [D_opt_x_min['INVENTORY'], D_opt_x_max['INVENTORY']]
    y_array = [D_opt_x_min['CUMULATIVE'], D_opt_x_max['CUMULATIVE']]

    prob = np.interp(inventory, x_array, y_array)
    risk = 1 - prob
    return risk


def returnInventoryValueFromInventoryFunctionRisk(inventory_values: list, risk: float) -> float:
    """
    Calculates the inventory value associated with a given risk value.

    Args:
        inventory_values (list): list of float with inventory values.
        risk (float): value of risk to evaluate.

    Returns:
        float: Inventory level associated with the given risk.

    """

    # build empirical pdf and cdf
    D_inventory = pd.DataFrame(inventory_values, columns=['INVENTORY'])
    D_inventory = D_inventory.groupby('INVENTORY').size().to_frame().reset_index()
    D_inventory.columns = ['INVENTORY', 'FREQUENCY']
    D_inventory = D_inventory.sort_values(by=['INVENTORY'])
    D_inventory['PROB'] = D_inventory['FREQUENCY'] / sum(D_inventory['FREQUENCY'])
    D_inventory['CUMULATIVE'] = D_inventory['PROB'].cumsum()

    # calculate the inventory quantity
    prob = 1 - risk
    D_opt_x_max = D_inventory[D_inventory['CUMULATIVE'] >= prob].iloc[0]
    D_opt_x_min = D_inventory[D_inventory['CUMULATIVE'] < prob].iloc[-1]

    x_array = [D_opt_x_min['CUMULATIVE'], D_opt_x_max['CUMULATIVE']]
    y_array = [D_opt_x_min['INVENTORY'], D_opt_x_max['INVENTORY']]
    inventory = np.interp(prob, x_array, y_array)
    return inventory


def returnTriangularCDF(x: float, a: float, b: float, c: float) -> float:
    """
    Uses a triangular distribution and return  the CDF

    Args:
        x (float): independent variable of the CDF.
        a (float): min value of the triangular distribution.
        b (float): max value of the triangular sitribution.
        c (float): mode of the triangular distribution.

    Returns:
        float: value of risk associated with the inventory level x.

    """

    probability = np.nan
    if x <= a:
        probability = 0
    elif (a < x) & (x <= c):
        probability = ((x - a) ** 2) / ((b - a) * (c - a))
    elif (c < x) & (x < b):
        probability = 1 - ((b - x) ** 2) / ((b - a) * (b - c))
    elif (x >= b):
        probability = 1
    else:
        print("Error in the CDF")
    return 1 - probability


def returnTriangularValue(risk: float, a: float, b: float, c: float) -> float:
    """
    Returns the value of the probability distribution associated with a given risk (risk = 1 - probability)

    Args:
        risk (float): value of risk associated to the probability distribution (inverse of the probability).
        a (float): min value of the triangular distribution.
        b (float): max value of the triangular sitribution.
        c (float): mode of the triangular distribution.

    Returns:
        x (float): quantity value associated with the risk probability.

    """

    u = 1 - risk
    x = np.nan
    if (0 <= u) & (u < ((c - a) / (b - a))):
        x = a + np.sqrt((b - a) * (c - a) * u)
    elif (((c - a) / (b - a)) <= u) & (u <= 1):
        x = b - np.sqrt((b - a) * (b - c) * (1 - u))
    else:
        print("Error in the CDF")
    return x
