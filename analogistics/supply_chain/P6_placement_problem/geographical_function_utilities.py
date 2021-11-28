
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def surfaceFromPoints(D_input: pd.DataFrame, xCol: str, yCol: str, zCol: str):
    """
    represent a 3D function from points

    Args:
        D_input (pd.DataFrame): Input pandas dataframe.
        xCol (str): name of the column with x coordinates.
        yCol (str): name of the column with y coordinates.
        zCol (str): name of the column with z coordinates.

    Returns:
        X (TYPE): DESCRIPTION.
        Y (TYPE): DESCRIPTION.
        grid (TYPE): DESCRIPTION.

    """
    # identify the rectangular to represent
    min_x = min(D_input[xCol])
    max_x = max(D_input[xCol])

    min_y = min(D_input[yCol])
    max_y = max(D_input[yCol])

    # define the grid
    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    X, Y = np.meshgrid(x, y)
    xy_coord = list(zip(D_input[xCol], D_input[yCol]))

    # interpolate the function in the missing points
    grid = griddata(xy_coord, np.array(D_input[zCol]), (X, Y), method='linear')
    return X, Y, grid
