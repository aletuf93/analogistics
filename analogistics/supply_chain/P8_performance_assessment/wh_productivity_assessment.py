# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from analogistics.clean import cleanUsingIQR


def spaceProductivity(D_movements: pd.dataFrame, variableToPlot: str, inout_column: str,
                      x_col: str, y_col: str, z_col: str, graphType: str = '2D', cleanData: bool = False):
    """
    3D warehouse productivity plot

    Args:
        D_movements (pd.dataFrame): pandas dataframe with movements.
        variableToPlot (str): string with the column to plot. or "popularity" for movement count.
        inout_column (str): string of the column with inout.
        x_col (str): string of the column with x coordinates.
        y_col (str): string of the column with y coordinates.
        z_col (str): string of the column with z coordinates.
        graphType (str, optional): The default is '2D'. 2D or 3D depending on the graph type. Defaults to '2D'.
        cleanData (bool, optional): if True, IQR is used to clean popularity of each location. Defaults to False.

    Returns:
        dict: Output dictionary with figures.

    """

    def _scaleSize(series):
        if min(series) == max(series):
            return [1 for i in range(0, len(series))]
        else:
            return (series - min(series)) / (max(series) - min(series))

    figure_output = {}
    # group data
    if variableToPlot == 'popularity':
        if graphType == '3D':
            D_mov = D_movements.groupby(['PERIOD', inout_column, x_col, y_col, z_col]).size().reset_index()
            D_mov.columns = ['PERIOD', 'INOUT', 'LOCCODEX', 'LOCCODEY', 'LOCCODEZ', 'POPULARITY']
        elif graphType == '2D':
            D_mov = D_movements.groupby(['PERIOD', inout_column, x_col, y_col]).size().reset_index()
            D_mov.columns = ['PERIOD', 'INOUT', 'LOCCODEX', 'LOCCODEY', 'POPULARITY']
    else:
        if graphType == '3D':
            D_mov = D_movements.groupby(['PERIOD', inout_column, x_col, y_col, z_col]).sum()[variableToPlot].reset_index()
            D_mov.columns = ['PERIOD', 'INOUT', 'LOCCODEX', 'LOCCODEY', 'LOCCODEZ', 'POPULARITY']
        elif graphType == '2D':
            D_mov = D_movements.groupby(['PERIOD', inout_column, x_col, y_col]).sum()[variableToPlot].reset_index()
            D_mov.columns = ['PERIOD', 'INOUT', 'LOCCODEX', 'LOCCODEY', 'POPULARITY']

    # split data into inbound and outbound
    D_loc_positive = D_mov[D_mov[inout_column] == '+']
    D_loc_negative = D_mov[D_mov[inout_column] == '-']

    # render inbound figure
    if len(D_loc_positive) > 0:

        # clean data
        if cleanData:
            D_warehouse_grouped, _ = cleanUsingIQR(D_loc_positive, features=['POPULARITY'], capacityField=[])
        else:
            D_warehouse_grouped = D_loc_positive
        # create figures
        for period in set(D_warehouse_grouped['PERIOD']):
            # period = list(set(D_warehouse_grouped['PERIOD']))[0]
            D_warehouse_grouped_filtered = D_warehouse_grouped[D_warehouse_grouped['PERIOD'] == period]
            D_warehouse_grouped_filtered['SIZE'] = _scaleSize(D_warehouse_grouped_filtered['POPULARITY'])

            # scale size
            D_warehouse_grouped_filtered['SIZE'] = 100 * D_warehouse_grouped_filtered['SIZE']

            # graphType 2-Dimensional
            if graphType == '2D':
                fig1 = plt.figure()
                plt.scatter(D_warehouse_grouped_filtered['LOCCODEX'],
                            D_warehouse_grouped_filtered['LOCCODEY'],
                            D_warehouse_grouped_filtered['SIZE'],
                            c=D_warehouse_grouped_filtered['SIZE'])
                plt.colorbar()
                plt.title(f"Warehouse INBOUND productivity, period:{period}")
                plt.xlabel("Warehouse front (x)")
                plt.ylabel("Warehouse depth (y)")
                figure_output[f"IN_productivity_2D_{period}"] = fig1

            # graphtype 3-Dimensional
            elif graphType == '3D':
                fig1 = plt.figure()
                fig1.add_subplot(111, projection='3d')
                plt.scatter(x=D_warehouse_grouped_filtered['LOCCODEX'],
                            y=D_warehouse_grouped_filtered['LOCCODEY'],
                            zs=D_warehouse_grouped_filtered['LOCCODEZ'],
                            s=D_warehouse_grouped_filtered['SIZE'],
                            c=D_warehouse_grouped_filtered['SIZE']
                            )
                plt.colorbar()
                plt.xlabel("Warehouse front (x)")
                plt.ylabel("Warehouse depth (y)")
                plt.title(f"Warehouse INBOUND productivity, period:{period}")
                figure_output[f"IN_productivity_3D_{period}"] = fig1

    # render outbound figure
    if len(D_loc_negative) > 0:

        # clean data
        if cleanData:
            D_warehouse_grouped, _ = cleanUsingIQR(D_loc_negative, features=['POPULARITY'], capacityField=[])
        else:
            D_warehouse_grouped = D_loc_negative
        # create figures
        for period in set(D_warehouse_grouped['PERIOD']):
            # period = list(set(D_warehouse_grouped['PERIOD']))[0]
            D_warehouse_grouped_filtered = D_warehouse_grouped[D_warehouse_grouped['PERIOD'] == period]
            D_warehouse_grouped_filtered['SIZE'] = _scaleSize(D_warehouse_grouped_filtered['POPULARITY'])

            # scale size
            D_warehouse_grouped_filtered['SIZE'] = 100 * D_warehouse_grouped_filtered['SIZE']

            # graphType 2-Dimensional
            if graphType == '2D':
                fig1 = plt.figure()
                plt.scatter(D_warehouse_grouped_filtered['LOCCODEX'],
                            D_warehouse_grouped_filtered['LOCCODEY'],
                            D_warehouse_grouped_filtered['SIZE'],
                            c=D_warehouse_grouped_filtered['SIZE'])
                plt.colorbar()
                plt.title(f"Warehouse OUTBOUND productivity, period:{period}")
                plt.xlabel("Warehouse front (x)")
                plt.ylabel("Warehouse depth (y)")
                figure_output[f"OUT_productivity_2D_{period}"] = fig1

            # graphtype 3-Dimensional
            elif graphType == '3D':
                fig1 = plt.figure()
                fig1.add_subplot(111, projection='3d')
                plt.scatter(x=D_warehouse_grouped_filtered['LOCCODEX'],
                            y=D_warehouse_grouped_filtered['LOCCODEY'],
                            zs=D_warehouse_grouped_filtered['LOCCODEZ'],
                            s=D_warehouse_grouped_filtered['SIZE'],
                            c=D_warehouse_grouped_filtered['SIZE']
                            )
                plt.colorbar()
                plt.xlabel("Warehouse front (x)")
                plt.ylabel("Warehouse depth (y)")
                plt.title(f"Warehouse OUTBOUND productivity, period:{period}")
                figure_output[f"OUT_productivity_3D_{period}"] = fig1
    return figure_output


def timeProductivity(D_movements: pd.DataFrame, variableToPlot: str, inout_column: str):
    """
    time productivity plots

    Args:
        D_movements (pd.DataFrame): input movements dataframe.
        variableToPlot (str): string with the column to plot. or "popularity" for movement count.
        inout_column (str): string of the inout column.

    Returns:
        figure_output (TYPE): dictionary with output figures.

    """

    figure_output = {}

    if variableToPlot == 'popularity':
        D_mov = D_movements.groupby(['PERIOD', inout_column]).size().reset_index()

    else:
        D_mov = D_movements.groupby(['PERIOD', inout_column]).sum()[variableToPlot].reset_index()
    D_mov.columns = ['PERIOD', 'INOUT', 'MOVEMENTS']

    D_loc_positive = D_mov[D_mov[inout_column] == '+']
    D_loc_negative = D_mov[D_mov[inout_column] == '-']

    # render inbound figure
    if len(D_loc_positive) > 0:

        fig1 = plt.figure()
        plt.plot(D_loc_positive['PERIOD'],
                 D_loc_positive['MOVEMENTS'])
        plt.title("Warehouse INBOUND productivity")
        plt.xticks(rotation=45)
        plt.xlabel("Timeline")
        plt.ylabel("N. of lines")
        figure_output["IN_productivity_trend"] = fig1

    # render inbound figure
    if len(D_loc_negative) > 0:

        fig1 = plt.figure()
        plt.plot(D_loc_negative['PERIOD'],
                 D_loc_negative['MOVEMENTS'])
        plt.title("Warehouse OUTBOUND productivity")
        plt.xticks(rotation=45)
        plt.xlabel("Timeline")
        plt.ylabel("N. of lines")
        figure_output["OUT_productivity_trend"] = fig1
    return figure_output


def movementsStatistics(D_movements: pd.dataFrame):
    """
    Calculate statistics on the movements of a warehouse.

    Args:
        D_movements (pd.dataFrame): Input movements dataframe.

    Returns:
        D_wh_stat (pd.dataFrame): Output statistics DataFrame.

    """

    D_wh_stat = pd.DataFrame(columns=['Description', 'value'])

    # calculate the number of movements
    n_mov = len(D_movements)
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Num. Movements', n_mov]], columns=D_wh_stat.columns))

    # calculate the inbound movements
    D_in = D_movements[D_movements['INOUT'] == '+']
    n_mov_in = len(D_in)
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Num. Movements INBOUND', n_mov_in]], columns=D_wh_stat.columns))

    # calculate the inbound movements
    D_out = D_movements[D_movements['INOUT'] == '-']
    n_mov_out = len(D_out)
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Num. Movements OUTBOUND', n_mov_out]], columns=D_wh_stat.columns))

    # calculate the number of movement days
    n_days = (max(D_movements['TIMESTAMP_IN']) - min(D_movements['TIMESTAMP_IN'])).days
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Num. Days', n_days]], columns=D_wh_stat.columns))

    # check layout data
    D_lay = D_movements.dropna(subset=['RACK'])['RACK'].drop_duplicates()
    if len(D_lay) > 2:
        lay = 'OK'
    else:
        lay = 'NO'
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Layout data', lay]], columns=D_wh_stat.columns))

    # check coordinates data
    D_lay = D_movements.dropna(subset=['LOCCODEX'])['LOCCODEX'].drop_duplicates()
    if len(D_lay) > 2:
        lay = 'OK'
    else:
        lay = 'NO'
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Coordinates data', lay]], columns=D_wh_stat.columns))

    # calcualte the number of storage locations
    n_loc = len(D_movements['IDLOCATION'].drop_duplicates())
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Num. Locations', n_loc]], columns=D_wh_stat.columns))

    # calcualte the number of storage locations
    n_idwh = len(D_movements['IDWH'].drop_duplicates())
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Num. Wh systems', n_idwh]], columns=D_wh_stat.columns))

    # calculate the number of SKUs
    n_skus = len(D_movements['ITEMCODE'].drop_duplicates())
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Num. SKUs', n_skus]], columns=D_wh_stat.columns))

    # check volume data
    D_lay = D_movements.dropna(subset=['VOLUME'])['VOLUME'].drop_duplicates()
    if len(D_lay) > 2:
        vol = 'OK'
    else:
        vol = 'NO'
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Volume data', vol]], columns=D_wh_stat.columns))

    # check pickinglist data
    D_lay = D_movements.dropna(subset=['PICKINGLIST'])['PICKINGLIST'].drop_duplicates()
    if len(D_lay) > 2:
        vol = 'OK'
    else:
        vol = 'NO'
    D_wh_stat = D_wh_stat.append(pd.DataFrame([['Picking list data', vol]], columns=D_wh_stat.columns))

    return D_wh_stat
