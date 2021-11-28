import numpy as np
import pandas as pd
# from scipy.stats import poisson
from analogistics.statistics import time_series as ts
# from analogistics.supply_chain.P1_familyProblem.part_classification import returnsparePartclassification


def returnResampleMovements(D_mov_loc: pd.DataFrame):
    """
    Resample movements daily and split into positive and negative

    Args:
        D_mov_loc (pd.DataFrame): Input movements DataFrame.

    Returns:
        MOVEMENT_POSITIVE_DAYS (list): Output positive days list.
        MOVEMENT_POSITIVE (list): Output positive days list.
        MOVEMENT_NEGATIVE_DAYS (list): Output negative days list.
        MOVEMENT_NEGATIVE (list): Output negative days list.

    """

    # Create series of positive movements
    D_mov_loc_positive = D_mov_loc[D_mov_loc['MOVEMENTS'] > 0]
    D_mov_loc_positive_series = D_mov_loc_positive
    D_mov_loc_positive_series = D_mov_loc_positive_series.set_index('TIMESTAMP_IN', drop=True)
    D_mov_loc_positive_series = D_mov_loc_positive_series['MOVEMENTS']
    D_mov_loc_positive_series = D_mov_loc_positive_series.resample('1D').sum()
    D_mov_loc_positive_series = D_mov_loc_positive_series.to_frame()
    D_mov_loc_positive_series['PERIOD'] = D_mov_loc_positive_series.index
    D_mov_loc_positive_series['PERIOD'] = ts.sampleTimeSeries(D_mov_loc_positive_series['PERIOD'], 'day')
    MOVEMENT_POSITIVE_DAYS = list(D_mov_loc_positive_series['PERIOD'])
    MOVEMENT_POSITIVE = list(D_mov_loc_positive_series['MOVEMENTS'])

    # Create series of negative movements
    D_mov_loc_negative = D_mov_loc[D_mov_loc['MOVEMENTS'] < 0]
    D_mov_loc_negative_series = D_mov_loc_negative
    D_mov_loc_negative_series = D_mov_loc_negative_series.set_index('TIMESTAMP_IN', drop=True)
    D_mov_loc_negative_series = D_mov_loc_negative_series['MOVEMENTS']
    D_mov_loc_negative_series = D_mov_loc_negative_series.resample('1D').sum()
    D_mov_loc_negative_series = D_mov_loc_negative_series.to_frame()
    D_mov_loc_negative_series['PERIOD'] = D_mov_loc_negative_series.index
    D_mov_loc_negative_series['PERIOD'] = ts.sampleTimeSeries(D_mov_loc_negative_series['PERIOD'], 'day')
    D_mov_loc_negative_series['MOVEMENTS'] = np.abs(D_mov_loc_negative_series['MOVEMENTS'])
    MOVEMENT_NEGATIVE_DAYS = list(D_mov_loc_negative_series['PERIOD'])
    MOVEMENT_NEGATIVE = list(D_mov_loc_negative_series['MOVEMENTS'])

    return MOVEMENT_POSITIVE_DAYS, MOVEMENT_POSITIVE, MOVEMENT_NEGATIVE_DAYS, MOVEMENT_NEGATIVE


def extractInventoryFromDataframe(D_loc: pd.DataFrame, dateAttribute: str = 'INVENTORY_DAYS',
                                  listField: str = 'INVENTORY_QUANTITY') -> pd.DataFrame:
    """
    Return Inventory Function

    Args:
        D_loc (pd.DataFrame): Input dataFrame.
        dateAttribute (str, optional): column name with inventory days. Defaults to 'INVENTORY_DAYS'.
        listField (str, optional): column name with inventory quantity. Defaults to 'INVENTORY_QUANTITY'.

    Returns:
        D_inventory (pd.dataFrame): Output Inventory DataFrame.

    """
    D_inventory = pd.DataFrame([], columns=['GLOBAL_TREND_QUANTITY'])

    for i in range(0, len(D_loc)):
        # i=33159
        list_days = D_loc.iloc[i][dateAttribute]
        # go on only if an inventory has been saved
        if isinstance(list_days, list):
            list_inventory = np.array(D_loc.iloc[i][listField])
            list_inventory = np.nan_to_num(list_inventory)  # convert nan to 0

            D_temp = pd.DataFrame(list_inventory, index=list_days, columns=['LOC_INVENTORY'])
            D_inventory = pd.concat([D_temp, D_inventory], axis=1, sort=False)
            D_inventory = D_inventory.fillna(0)
            D_inventory['GLOBAL_TREND_QUANTITY'] = D_inventory['GLOBAL_TREND_QUANTITY'] + D_inventory['LOC_INVENTORY']
            D_inventory = D_inventory.drop(columns=['LOC_INVENTORY'])
    return D_inventory


def returnInventoryPart(D_movements: pd.DataFrame, D_inventory: pd.DataFrame,
                        timeLineDays: pd.DataFrame, quantityColums: str = 'QUANTITY'):
    """
    Defines the inventory function (grouped by day) of a part

    Args:
        D_movements (pd.DataFrame): dataframe of movements of a single part (ITEMCODE).
        D_inventory (pd.DataFrame): dataframe of inventory of a single part (ITEMCODE) already grouped by TIMESTAMP and ITEMCODE.
        timeLineDays (pd.DataFrame): dataframe wih a column TIMELINE having an aggregation of all the days to generate the inventory array.
        quantityColums (str, optional): indicates the column with the movement or inventory quantity. Defaults to 'QUANTITY'.

    Returns:
        array_days (list): list of days where the inventory is reconstructed.
        array_inventory (list): list of inventory values where the inventory is reconstructed.

    """

    # Drop rows without the sign
    D_movements = D_movements[D_movements['INOUT'].isin(['+', '-'])]
    # if at least one movement is available
    if len(D_movements) > 0:
        # identify the sign
        D_movements['MOVEMENT'] = D_movements['INOUT'].astype(str) + D_movements[quantityColums].astype(str)
        D_movements['MOVEMENT'] = D_movements['MOVEMENT'].astype(float)

        # group on a daily basis
        D_movements['PERIOD'] = ts.sampleTimeSeries(D_movements['TIMESTAMP_IN'], 'day')
        D_movements_grouped = D_movements.groupby(['PERIOD']).sum()['MOVEMENT'].reset_index()
        D_movements_grouped = D_movements_grouped.sort_values(by='PERIOD')

        # define the inventory, given the movements
        D_movements_grouped['INVENTORY'] = np.nan
        D_movements_grouped.at[0, 'INVENTORY'] = D_movements_grouped.iloc[0]['MOVEMENT']
        for i in range(1, len(D_movements_grouped)):
            D_movements_grouped.at[i, 'INVENTORY'] = D_movements_grouped.iloc[i - 1]['INVENTORY'] + D_movements_grouped.iloc[i]['MOVEMENT']
        if min(D_movements_grouped['INVENTORY']) < 0:
            D_movements_grouped['INVENTORY'] = D_movements_grouped['INVENTORY'] - min(D_movements_grouped['INVENTORY'])
        # if no movements are available, set to zero
    else:
        D_movements_grouped = pd.DataFrame([[timeLineDays['TIMELINE'].iloc[0], 0]], columns=['PERIOD', 'INVENTORY'])

    # merge with time
    D_inventory_part = timeLineDays.merge(D_movements_grouped, how='left', left_on='TIMELINE', right_on='PERIOD')

    # use forward fill to clean null (if no inventory observations are available, use the last inventory calculated
    D_inventory_part['INVENTORY'] = D_inventory_part['INVENTORY'].fillna(method='ffill')

    # fill with zero
    D_inventory_part['INVENTORY'] = D_inventory_part['INVENTORY'].fillna(0)

    # if at least one observed inventory is availbale, correct the curve
    if len(D_inventory) > 0:
        D_inventory['PERIOD'] = ts.sampleTimeSeries(D_inventory['TIMESTAMP'], 'day')
        D_inventory = D_inventory.merge(D_inventory_part, how='left', left_on='PERIOD', right_on='TIMELINE')

        # for each inventory observed, try to fix the inventory curve
        for index, row in D_inventory.iterrows():

            # chek if the inventory has been undersetimated
            if row[quantityColums] > row.INVENTORY:
                gap = row[quantityColums] - row.INVENTORY
                D_inventory_part['INVENTORY'] = D_inventory_part['INVENTORY'] + gap

    array_days = list(D_inventory_part['TIMELINE'])
    array_inventory = list(D_inventory_part['INVENTORY'])

    return array_days, array_inventory


def movementfunctionfromInventory(I_t_cleaned: list) -> pd.DataFrame:
    """
    Define movement function rom Inventory function

    Args:
        I_t_cleaned (list): list of inventory values without nan values.

    Returns:
        list: pandas dataframe of movements.

    """

    M_t = []
    for j in range(1, len(I_t_cleaned)):
        M_t.append(I_t_cleaned[j] - I_t_cleaned[j - 1])
    M_t = pd.DataFrame(M_t, columns=['QUANTITY'])
    return M_t


def assessInterarrivalTime(I_t: list):
    """
    measure the interarrival time between inbound activities

    Args:
        I_t (list): Inventory function.

    Returns:
        mean_interarrival_time_in (float): Mean interarrival time.
        std_interarrival_time_in (float): standard deviation of the interarrival time.
        interarrival_time (list): list of the interarrival times.

    """

    # remove nan values
    I_t_cleaned = [x for x in I_t if str(x) != 'nan']  # remove nan inventories (e.g. at the beginning of the series if the part is not in the WH)

    # generate the movement function
    M_t = movementfunctionfromInventory(I_t_cleaned)

    M_t_in = M_t[M_t['QUANTITY'] > 0]
    interarrival_time = []
    for j in range(1, len(M_t_in)):
        interarrival_time.append(M_t_in.index[j] - M_t_in.index[j - 1])

    # if one or zero data point set the interarrival time equal to zero
    if len(M_t_in) <= 1:
        interarrival_time.append(0)

    mean_interarrival_time_in = np.mean(interarrival_time)
    std_interarrival_time_in = np.std(interarrival_time)
    return mean_interarrival_time_in, std_interarrival_time_in, interarrival_time


def updatePartInventory(D_SKUs: pd.DataFrame, D_movements: pd.DataFrame, D_inventory: pd.DataFrame,
                        timecolumn_mov: str, itemcodeColumns_sku: str, itemcodeColumns_mov: str,
                        itemcodeColumns_inv: str) -> pd.DataFrame:
    """
    Update the inventory of a part

    Args:
        D_SKUs (pd.DataFrame): Input SKUs DataFrame.
        D_movements (pd.DataFrame): Input Movements dataFrame.
        D_inventory (pd.DataFrame): Input Inventory DataFrame.
        timecolumn_mov (str): column name containing movements timestamp.
        itemcodeColumns_sku (str): column name containing SKUs item code.
        itemcodeColumns_mov (str): column name containing movements item code.
        itemcodeColumns_inv (str): column name containing inventory item code.
    Returns:
        D_SKUs (TYPE): DESCRIPTION.

    """

    D_SKUs['INVENTORY_QTY'] = [[] for i in range(0, len(D_SKUs))]
    D_SKUs['INVENTORY_DAYS'] = [[] for i in range(0, len(D_SKUs))]

    # define the inventory quantity
    firstDay = min(D_movements[timecolumn_mov]).date()
    lastDay = max(D_movements[timecolumn_mov]).date()
    timeLine = pd.date_range(start=firstDay, end=lastDay).to_frame()
    timeLineDays = ts.sampleTimeSeries(timeLine[0], 'day').to_frame()
    timeLineDays.columns = ['TIMELINE']

    D_SKUs = D_SKUs.reset_index(drop=True)
    # Build a daily inventory array for each part
    for index, row in D_SKUs.iterrows():
        # print(part)
        # part = list(set(D_mov['ITEMCODE']))[0]
        part = row[itemcodeColumns_sku]

        # filter movements by itemcode
        D_mov_part = D_movements[D_movements[itemcodeColumns_mov] == part]

        # filter inventory by itemcode
        D_inv_part = D_inventory[D_inventory[itemcodeColumns_inv] == part]

        array_days, array_inventory = returnInventoryPart(D_mov_part, D_inv_part, timeLineDays)
        # plt.plot(array_days,array_inventory)

        # update the dataframe
        D_SKUs.at[index, 'INVENTORY_QTY'] = array_inventory
        D_SKUs.at[index, 'INVENTORY_DAYS'] = array_days
    return D_SKUs


def updateInterarrivalTime(D_SKUs: pd.DataFrame) -> pd.DataFrame:
    """
    Update the interarrival time

    Args:
        D_SKUs (pd.DataFrame): Input SKUs DataFrame.

    Returns:
        D_SKUs (pd.DataFrame): Output SKUs DataFrame.

    """

    # create result columns
    D_SKUs['INTERARRIVAL_MEAN_IN'] = np.nan
    D_SKUs['INTERARRIVAL_STD_IN'] = np.nan

    for index, row in D_SKUs.iterrows():
        # select inventory curve
        I_t = D_SKUs.loc[index]['INVENTORY_QTY']
        # calculate the popularity
        movements = movementfunctionfromInventory(I_t)
        movements = movements.dropna()
        if len(movements) > 0:
            mean_interarrival_time_in, std_interarrival_time_in, _ = assessInterarrivalTime(I_t)

            # update the dataframe
            D_SKUs.at[index, 'INTERARRIVAL_MEAN_IN'] = mean_interarrival_time_in
            D_SKUs.at[index, 'INTERARRIVAL_STD_IN'] = std_interarrival_time_in

    return D_SKUs
