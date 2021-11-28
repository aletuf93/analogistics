# -*- coding: utf-8 -*-

# https://github.com/gboeing/osmnx-examples/tree/master/notebooks


from analogistics.clean import cleanUsingIQR

import osmnx as ox
import numpy as np
import pandas as pd


def import_graph_drive(D_node: pd.DataFrame, latCol: str, lonCol: str, D_plant: pd.DataFrame,
                       plantLatitude: float, plantLongitude: float, cleanOutliers: bool = False):
    """
    imports a road network using osmnx library

    Args:
        D_node (pd.DataFrame): table containing the nodes of the network.
        latCol (str): name attribute of the latitude of the node collection.
        lonCol (str): name attribute of the longitude of the node collection.
        D_plant (pd.DataFrame): table containing the plant of the network.
        plantLatitude (float): name attribute of the latitude of the plant collection.
        plantLongitude (float): name attribute of the longitude of the plant collection.
        cleanOutliers (bool, optional): if True to remove outliers of latitude and logitude by using IQR. Defaults to False.

    Returns:
        nx.Graph: output graph.
        float: percentage coverages

    """

    coverages = (1, np.nan)

    # remove latitude and longitude outliers
    if cleanOutliers:
        D_node, coverages, = cleanUsingIQR(D_node, [latCol, lonCol])

    allLatitudes = list(D_node[latCol]) + list(D_plant[plantLatitude])
    allLongitudes = list(D_node[lonCol]) + list(D_plant[plantLongitude])

    min_lat = min(allLatitudes)
    max_lat = max(allLatitudes)
    min_lon = min(allLongitudes)
    max_Lon = max(allLongitudes)

    G = ox.graph_from_bbox(max_lat, min_lat, max_Lon, min_lon, network_type='drive')

    output_coverages = pd.DataFrame(coverages)
    return G, output_coverages
