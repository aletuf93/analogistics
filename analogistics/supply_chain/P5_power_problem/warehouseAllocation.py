# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def AllocateSKUs(D_mov: pd.DataFrame, D_part: pd.DataFrame) -> pd.DataFrame:
    '''
    allocate EQS, EQT and OPT allocation for all the SKUs

    Parameters
    ----------
    D_mov : TYPE pandas dataframe
        DESCRIPTION. movements dataframe containing ITEMCODE and QUANTITY columns
    D_part : TYPE pandas dataframe
        DESCRIPTION. sku master file containing ITEMCODE and VOLUME (volume cannot be null or zero)

    Returns
    -------
    D_mov_qty : TYPE pandas dataframe
        DESCRIPTION. dataframe containing the SKU master file with EQS, EQT and OPT values

    '''

    D_mov_qty = D_mov.groupby(['ITEMCODE']).sum()['QUANTITY'].to_frame().reset_index()
    D_mov_qty.columns = ['ITEMCODE', 'QUANTITY']
    D_mov_qty = D_mov_qty.merge(D_part, on='ITEMCODE', how='left')
    D_mov_qty['fi'] = D_mov_qty['QUANTITY'] * D_mov_qty['VOLUME']
    D_mov_qty = D_mov_qty.dropna()
    D_mov_qty = D_mov_qty[D_mov_qty['fi'] > 0]

    if len(D_mov_qty) > 0:
        D_mov_qty['EQS'] = 1 / len(D_mov_qty)
        D_mov_qty['EQT'] = D_mov_qty['fi'] / sum(D_mov_qty['fi'])
        D_mov_qty['OPT'] = np.sqrt(D_mov_qty['fi']) / sum(np.sqrt(D_mov_qty['fi']))
    return D_mov_qty


def discreteAllocationParts(D_parts: pd.DataFrame, availableSpacedm3: float, method: str = 'OPT') -> pd.DataFrame:
    """
    Allocates the number of parts, given the available space and the results of the allocation methods (EQS, EQT, OPT)

    Args:
        D_parts (pd.DataFrame): dataframe of the SKUs master file containing VOLUME, EQS, EQT and OPT columns.
        availableSpacedm3 (float): available space in the same unit of measure of the parts VOLUME.
        method (str, optional): allocation method (EQS, EQT, OPT). Defaults to 'OPT'.

    Returns:
        pd.DataFrame: SKUs master file with the allocated number of SKUs.

    """

    # Check the input columns of the dataframe
    checkColumns = ['VOLUME', 'EQS', 'EQT', 'OPT']
    for col in checkColumns:
        if col not in D_parts.columns:
            print(f"Column {col} not in dataframe D_parts")
            return []

    # Check the input method
    checkMethods = ['EQS', 'EQT', 'OPT']
    if method not in checkMethods:
        print("Unknown allocation method, choose between EQS, EQT, OPT")
        return []

    # allocate the parts
    D_parts['ALLOCATED_VOLUME'] = D_parts[method] * availableSpacedm3
    D_parts[f"N_PARTS_{method}"] = np.round(D_parts['ALLOCATED_VOLUME'] / D_parts['VOLUME'], 0)
    return D_parts
