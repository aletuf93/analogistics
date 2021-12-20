
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculateABCsaving(p: float, q: float, D_parts: pd.DataFrame):
    """
    Calculate the saving based on re-assignment with classes ABC, compared with a random assignment scenario

    Args:
        p (float): warehouse front length in meters.
        q (float): warehouse depth in meters.
        D_parts (pd.DataFrame): dataframe containing SKUs with columns POP_IN_TOT and POP_OUT_TOT.

    Returns:
        list: list of threshold of class A.
        list: list of threshold of class B.
        list: optimal saving inbound.
        list: optimal saving inbound.
        float: best threshold A class.
        float: best threshold B class.
        float: best total saving (IN + OUT).

    """

    # Check the input columns of the dataframe
    checkColumns = ['POP_IN_TOT', 'POP_OUT_TOT']
    for col in checkColumns:
        if col not in D_parts.columns:
            print(f"Column {col} not in dataframe D_parts")
            return [], [], [], [], [], [], []

    # ############################ SCENARIO RANDOM #################################
    if (~np.isnan(p) or ~np.isnan(q)):

        # count pick in and out
        pickIN = sum(D_parts['POP_IN_TOT'])
        pickOUT = sum(D_parts['POP_OUT_TOT'])

        D_parts['POP_TOT'] = D_parts['POP_IN_TOT'] + D_parts['POP_OUT_TOT']

        # Random scenario
        # I/O distributed on the front
        r_cicloSemplice = (q / 2 + p / 3) * 2

        KmIn_rand = pickIN * r_cicloSemplice
        KmOut_rand = pickOUT * r_cicloSemplice

        SAVING_IN = []
        SAVING_OUT = []
        soglieA = []
        soglieB = []

        # ############################ SCENARIO ABC ###################################
        for i in range(0, 100, 10):
            for j in range(i + 1, 100, 10):
                sogliaA = i / 100
                sogliaB = j / 100

                D_pop_totale = D_parts.groupby('ITEMCODE')['POP_TOT'].sum().reset_index()
                D_pop_totale = D_pop_totale.sort_values(by='POP_TOT', ascending=False)

                sogliaClasseA = int(np.round(sogliaA * len(D_pop_totale)))
                sogliaClasseB = int(np.round(sogliaB * len(D_pop_totale)))

                ITEM_A = D_pop_totale['ITEMCODE'].iloc[0: sogliaClasseA].reset_index()
                ITEM_B = D_pop_totale['ITEMCODE'].iloc[sogliaClasseA: sogliaClasseB].reset_index()
                ITEM_C = D_pop_totale['ITEMCODE'].iloc[sogliaClasseB: len(D_pop_totale)].reset_index()

                # Count pickIn
                num_pickin_A = sum(D_parts[D_parts['ITEMCODE'].isin(ITEM_A['ITEMCODE'])]['POP_IN_TOT'])
                num_pickin_B = sum(D_parts[D_parts['ITEMCODE'].isin(ITEM_B['ITEMCODE'])]['POP_IN_TOT'])
                num_pickin_C = sum(D_parts[D_parts['ITEMCODE'].isin(ITEM_C['ITEMCODE'])]['POP_IN_TOT'])

                # Count le pickOUT
                num_pickout_A = sum(D_parts[D_parts['ITEMCODE'].isin(ITEM_A['ITEMCODE'])]['POP_OUT_TOT'])
                num_pickout_B = sum(D_parts[D_parts['ITEMCODE'].isin(ITEM_B['ITEMCODE'])]['POP_OUT_TOT'])
                num_pickout_C = sum(D_parts[D_parts['ITEMCODE'].isin(ITEM_C['ITEMCODE'])]['POP_OUT_TOT'])

                len_q_A = len(ITEM_A) / (len(ITEM_A) + len(ITEM_B) + len(ITEM_C))
                len_q_B = len(ITEM_B) / (len(ITEM_A) + len(ITEM_B) + len(ITEM_C))
                len_q_C = len(ITEM_C) / (len(ITEM_A) + len(ITEM_B) + len(ITEM_C))

                # Calculate the km
                # check OK number picks
                if((num_pickin_A + num_pickin_B + num_pickin_C) == pickIN) & ((num_pickout_A + num_pickout_B + num_pickout_C) == pickOUT):

                    # I/O distributed on the front
                    dist_A = (q * len_q_A / 2 + p / 3) * 2
                    dist_B = (q * (len_q_A + len_q_B / 2) + p / 3) * 2
                    dist_C = (q * (len_q_A + len_q_B + len_q_C / 2) + p / 3) * 2

                    KmIn_ABC = (num_pickin_A * dist_A + num_pickin_B * dist_B + num_pickin_C * dist_C)
                    KmOut_ABC = (num_pickout_A * dist_A + num_pickout_B * dist_B + num_pickout_C * dist_C)

                    if (KmIn_rand == 0):  # avoid division by zero
                        sav_IN = 0
                    else:
                        sav_IN = 1 - float(KmIn_ABC / KmIn_rand)

                    if (KmOut_rand == 0):  # avoid division by zero
                        sav_OUT = 0
                    else:
                        sav_OUT = 1 - float(KmOut_ABC / KmOut_rand)

                    SAVING_IN.append(sav_IN)
                    SAVING_OUT.append(sav_OUT)
                    soglieA.append(sogliaA)
                    soglieB.append(sogliaB)

                    # calculate best saving scenario
                    SAV_TOT = np.asarray(SAVING_IN) + np.asarray(SAVING_OUT)
                    idx = np.nanargmax(SAV_TOT)
                    best_A = np.round(soglieA[idx], 1)
                    best_B = np.round(soglieB[idx], 1)

                else:
                    print("Error: num pick scenario ABC does not match num pick scenario random")

        return soglieA, soglieB, SAVING_IN, SAVING_OUT, best_A, best_B, SAV_TOT[idx]
    else:
        return [], [], [], [], [], [], []


def defineABCclassesOfStorageLocations(D_nodes: pd.DataFrame, AclassPerc: float = .2, BclassPerc: float = .5) -> pd.DataFrame:
    """
    Define the classes A, B, C for each storage location (nodes of the warehouse).

    Args:
        D_nodes (pd.DataFrame): pandas dataframe with storage locations and INPUT_DISTANCE and OUTPUT_DISTANCE .
        AclassPerc (float, optional): class A threshold. Defaults to .2.
        BclassPerc (float, optional): class B threshold. Defaults to .5.

    Returns:
        TYPE: input dataframe with the column CLASS (A,B,C) for each storage location.

    """

    # Check the input columns of the dataframe
    checkColumns = ['INPUT_DISTANCE', 'OUTPUT_DISTANCE']
    for col in checkColumns:
        if col not in D_nodes.columns:
            print(f"Column {col} not in dataframe D_parts")
            return []

    # calculate total distance
    D_nodes['WEIGHT'] = D_nodes['INPUT_DISTANCE'] + D_nodes['OUTPUT_DISTANCE']
    D_nodes = D_nodes.sort_values(by='WEIGHT', ascending=False)

    D_nodes['WEIGHT'] = D_nodes['WEIGHT'] / sum(D_nodes['WEIGHT'])
    D_nodes['WEIGHT_cum'] = D_nodes['WEIGHT'].cumsum()

    # assign classes
    D_nodes['CLASS'] = np.nan

    for i in range(0, len(D_nodes)):
        if D_nodes.iloc[i]['WEIGHT_cum'] < AclassPerc:
            D_nodes.iloc[i, D_nodes.columns.get_loc('CLASS')] = 'A'
        elif (D_nodes.iloc[i]['WEIGHT_cum'] >= AclassPerc) & (D_nodes.iloc[i]['WEIGHT_cum'] < BclassPerc):
            D_nodes.iloc[i, D_nodes.columns.get_loc('CLASS')] = 'B'
        else:
            D_nodes.iloc[i, D_nodes.columns.get_loc('CLASS')] = 'C'

    return D_nodes


def defineABCclassesOfParts(D_parts: pd.DataFrame, columnWeightList: list,
                            AclassPerc: float = .2, BclassPerc: float = .5) -> pd.DataFrame:
    """
    Assign classes A, B, C to the SKUs of the master files.

    Args:
        D_parts (pd.DataFrame): dataframe of parts.
        columnWeightList (list): list of column of D_parts with the weights to consider to define ABC classes.
        AclassPerc (float, optional): cut percentile of class A. Defaults to .2.
        BclassPerc (float, optional): cut percentile of class B. Defaults to .5.

    Returns:
        D_parts (pd.DataFrame): Output DataFrame with classes for each SKU.

    """

    D_parts['WEIGHT'] = 0
    # calculate total distance

    for col in columnWeightList:
        if col in D_parts.columns:
            D_parts['WEIGHT'] = D_parts['WEIGHT'] + D_parts[col]
        else:
            print(f"Column {col} not in index, column ignored")
    D_parts = D_parts.sort_values(by='WEIGHT', ascending=False)

    D_parts['WEIGHT'] = D_parts['WEIGHT'] / sum(D_parts['WEIGHT'])
    D_parts['WEIGHT_cum'] = D_parts['WEIGHT'].cumsum()

    # assign classes
    D_parts['CLASS'] = np.nan

    for i in range(0, len(D_parts)):
        if D_parts.iloc[i]['WEIGHT_cum'] < AclassPerc:
            D_parts.iloc[i, D_parts.columns.get_loc('CLASS')] = 'A'
        elif (D_parts.iloc[i]['WEIGHT_cum'] >= AclassPerc) & (D_parts.iloc[i]['WEIGHT_cum'] < BclassPerc):
            D_parts.iloc[i, D_parts.columns.get_loc('CLASS')] = 'B'
        else:
            D_parts.iloc[i, D_parts.columns.get_loc('CLASS')] = 'C'

    return D_parts


def plotSavingABCclass(p: float, q: float, D_SKUs: pd.DataFrame) -> dict:
    """
    Plot the 3d graph with saving given different values of a, b, c thresholds

    Args:
        p (float): warehouse front length in meters.
        q (float): warehouse depth in meters.
        D_SKUs (pd.DataFrame): DataFrame of the Sku master file.

    Returns:
        dict: output dictionary containing figures.

    """
    figure_output = {}

    # calculate saving
    soglieA, soglieB, SAVING_IN, SAVING_OUT, best_A, best_B, SAV_TOT = calculateABCsaving(p, q, D_SKUs)

    # inbound saving
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.scatter(soglieA,
               soglieB,
               SAVING_IN,
               color='orange'
               )

    plt.xlabel("A class threshold")
    plt.ylabel("B class threshold")
    plt.title("Inbound saving ")
    figure_output["IN_saving_ABC_inbound"] = fig1

    # outbound saving
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.scatter(soglieA,
               soglieB,
               SAVING_OUT,
               color='orange'
               )

    plt.xlabel("A class threshold")
    plt.ylabel("B class threshold")
    plt.title("Outbound saving ")
    figure_output["IN_saving_ABC_outbound"] = fig1

    return figure_output
