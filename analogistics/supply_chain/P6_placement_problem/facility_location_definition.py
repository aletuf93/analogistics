
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import osmnx as ox

import networkx as nx

from analogistics.chart.chart_3D_surface import createFigureWith3Dsurface


from analogistics.supply_chain.P8_performance_assessment.utilities_movements import getCoverageStats
from analogistics.clean import cleanUsingIQR

from sklearn.metrics import mean_squared_error


from sklearn import cluster
from sklearn.mixture import GaussianMixture


def mercatorProjection(latitude: float, longitude: float):
    """
    Return the Mercator projection coordinates of given latitude and longitude

    Args:
        latitude (float): Latitude of a point.
        longitude (float): Longitude of a point.

    Returns:
        x (TYPE): Mercator X coordinate.
        y (TYPE): Mercator Y coordinate.

    """

    R = 6378.14  # earth's ray
    e = 0.0167  # earth's eccentricity

    lon_rad = (np.pi / 180) * longitude
    lat_rad = (np.pi / 180) * latitude

    x = R * lon_rad
    y = R * np.log(((1 - e * np.sin(lat_rad)) / (1 + e * np.sin(lat_rad))) ** (e / 2) * np.tan(np.pi / 4 + lat_rad / 2))
    return x, y


def optimalLocationRectangularDistance(D_filtered: pd.DataFrame, latCol: str, lonCol: str, weightCol: str):
    """
    this function returns the optimal location based on rectangular distances

    Args:
        D_filtered (pd.DataFrame): Input dataframe.
        latCol (str): Column name containing latitude.
        lonCol (str): Column name containing longitude.
        weightCol (str): Column name containing weight (e.g. production quantity or flows).

    Returns:
        lat_optimal (TYPE): Optimal latitude.
        lon_optimal (TYPE): Optimal longitude.

    """

    # optimal location
    op_w = sum(D_filtered[weightCol]) / 2  # identify the median of the sum of weights

    # identify optimal latitude
    if len(D_filtered) > 1:  # when there are more than a single point
        D_filtered = D_filtered.sort_values(by=latCol, ascending=True)  # sort by latitude
        D_filtered['X_cumsum'] = D_filtered[weightCol].cumsum()  # calculate the cumulated sum

        # identify the LATITUDE closer to the optimal location
        D_opt_x_max = D_filtered[D_filtered['X_cumsum'] >= op_w].iloc[0]
        D_opt_x_min = D_filtered[D_filtered['X_cumsum'] < op_w].iloc[-1]

        x_array = [D_opt_x_min['X_cumsum'], D_opt_x_max['X_cumsum']]
        y_array = [D_opt_x_min[latCol], D_opt_x_max[latCol]]
        lat_optimal = np.interp(op_w, x_array, y_array)

        # identify the LONGITUDE closer to the optimal location
        D_filtered = D_filtered.sort_values(by=lonCol, ascending=True)  # sort by latitude
        D_filtered['Y_cumsum'] = D_filtered[weightCol].cumsum()  # calculate the cumulated sum

        D_opt_x_max = D_filtered[D_filtered['Y_cumsum'] >= op_w].iloc[0]
        D_opt_x_min = D_filtered[D_filtered['Y_cumsum'] < op_w].iloc[-1]

        x_array = [D_opt_x_min['Y_cumsum'], D_opt_x_max['Y_cumsum']]
        y_array = [D_opt_x_min[lonCol], D_opt_x_max[lonCol]]
        lon_optimal = np.interp(op_w, x_array, y_array)

    else:  # with a single point take the coordinates of the point
        lat_optimal = float(D_filtered.iloc[0][latCol])
        lon_optimal = float(D_filtered.iloc[0][lonCol])

    return lat_optimal, lon_optimal


def optimalLocationGravityProblem(D_filtered: pd.DataFrame, latCol: str, lonCol: str, weightCol: str):
    """
    this dunction calculate the optimal location with squared euclidean distances

    Args:
        D_filtered (pd.DataFrame): Input dataframe.
        latCol (str): Column name containing latitude.
        lonCol (str): Column name containing longitude.
        weightCol (str): Column name containing weight (e.g. production quantity or flows).

    Returns:
        lat_optimal (TYPE): Optimal latitude.
        lon_optimal (TYPE): Optimal longitude.

    """

    D_filtered_notnan = D_filtered.dropna(subset=[latCol, lonCol, weightCol])
    D_filtered_notnan = D_filtered_notnan[D_filtered_notnan[weightCol] > 0]
    if len(D_filtered_notnan) > 0:
        lat_optimal = sum(D_filtered_notnan[latCol] * D_filtered_notnan[weightCol]) / sum(D_filtered_notnan[weightCol])
        lon_optimal = sum(D_filtered_notnan[lonCol] * D_filtered_notnan[weightCol]) / sum(D_filtered_notnan[weightCol])
    else:
        lat_optimal = lon_optimal = 0
    return lat_optimal, lon_optimal


def optimalLocationEuclideanDistance(D_filtered: pd.DataFrame, latCol: str, lonCol: str, weightCol: str):
    """
    this function calculates the optimal location with euclidean distances using the kuhn procedure

    Args:
        D_filtered (pd.DataFrame): Input dataframe.
        latCol (str): Column name containing latitude.
        lonCol (str): Column name containing longitude.
        weightCol (str): Column name containing weight (e.g. production quantity or flows).

    Returns:
        lat_optimal (TYPE): Optimal latitude.
        lon_optimal (TYPE): Optimal longitude.

    """

    def _funcGKuhn(wi, xj_1, yj_1, ai, bi):
        # implements the function g in the kuhn procedure for euclidean distances
        return wi / ((xj_1 - ai) ** 2 + (yj_1 - bi) ** 2)

    # remove null values
    D_filtered_notnan = D_filtered.dropna(subset=[latCol, lonCol, weightCol])

    # identify the first solution of the gravity problem
    lat_optimal_0, lon_optimal_0 = optimalLocationGravityProblem(D_filtered_notnan, latCol, lonCol, weightCol)

    xj_1 = lon_optimal_0
    yj_1 = lat_optimal_0
    wi = D_filtered_notnan[weightCol]
    ai = D_filtered_notnan[lonCol]
    bi = D_filtered_notnan[latCol]

    # iterates Kuhn procedure to approximate the solution
    diff_x = 1  # a latitude degree is about 111 km
    while diff_x > 0.01:
        lon_optimal_j = sum(_funcGKuhn(wi, xj_1, yj_1, ai, bi) * ai) / sum(_funcGKuhn(wi, xj_1, yj_1, ai, bi))
        diff_x = np.abs(xj_1 - lon_optimal_j)
        # print(diff_x)
        xj_1 = lon_optimal_j

    # iterates Kuhn procedure to approximate the solution
    diff_x = 1
    while diff_x > 0.01:
        lat_optimal_j = sum(_funcGKuhn(wi, xj_1, yj_1, ai, bi) * bi) / sum(_funcGKuhn(wi, xj_1, yj_1, ai, bi))
        diff_x = np.abs(yj_1 - lat_optimal_j)
        # print(diff_x)
        yj_1 = lat_optimal_j

    return lat_optimal_j, lon_optimal_j


def func_rectangularDistanceCost(x: float, y: float, x_opt: float, y_opt: float, wi: float) -> float:
    """
    return cost values with rectangular distances

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        x_opt (float): X coordinate of the optimal location.
        y_opt (float): Y coordinate of the optimal location.
        wi (float): weight (e.g. flow).

    Returns:
        float: Cost value.

    """

    return (np.abs(x - x_opt) + np.abs(y - y_opt)) * wi


def func_gravityDistanceCost(x: float, y: float, x_opt: float, y_opt: float, wi: float) -> float:
    """
    return cost values with squared euclidean distances

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        x_opt (float): X coordinate of the optimal location.
        y_opt (float): Y coordinate of the optimal location.
        wi (float): weight (e.g. flow).

    Returns:
        float: Cost value.

    """

    return ((x - x_opt) ** 2 + (y - y_opt) ** 2) * wi


def func_euclideanDistanceCost(x: float, y: float, x_opt: float, y_opt: float, wi: float) -> float:
    """
    return cost values with euclidean distance

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        x_opt (float): X coordinate of the optimal location.
        y_opt (float): Y coordinate of the optimal location.
        wi (float): weight (e.g. flow).

    Returns:
        float: Cost value.

    """

    return np.sqrt((x - x_opt) ** 2 + (y - y_opt) ** 2) * wi


def defineDistanceTableEstimator(D_mov: pd.DataFrame, lonCol_From_mov: str, latCol_From_mov: str,
                                 lonCol_To_mov: str, latCol_To_mov: str, G: nx.graph,
                                 cleanOutliersCoordinates: bool = False, capacityField: str = 'QUANTITY'):
    """

    Args:
        D_mov (pd.DataFrame): Inpud dataframe.
        lonCol_From_mov (str): name of the D_mov dataframe with longitude of the loading node.
        latCol_From_mov (str): name of the D_mov dataframe with latitude of the loading node.
        lonCol_To_mov (str): name of the D_mov dataframe with longitude of the discharging node.
        latCol_To_mov (str): name of the D_mov dataframe with latitude of the loading node.
        G (nx.graph): road graph obtained with osmnx.
        cleanOutliersCoordinates (bool, optional): is true to remove outliers in latitude and longitude. Defaults to False.
        capacityField (str, optional): field of quantity to measure the coverage statistics on it. Defaults to 'QUANTITY'.

    Returns:
        D_dist (TYPE): DESCRIPTION.
        df_coverages (TYPE): DESCRIPTION.

    """

    # clean data and get coverages
    analysisFieldList = [lonCol_From_mov, latCol_From_mov, lonCol_To_mov, latCol_To_mov]
    coverages, _ = getCoverageStats(D_mov, analysisFieldList, capacityField=capacityField)
    D_dist = D_mov[[lonCol_From_mov, latCol_From_mov, lonCol_To_mov, latCol_To_mov]].drop_duplicates().dropna().reset_index()
    if cleanOutliersCoordinates:
        D_dist, coverages_outl = cleanUsingIQR(D_dist, [lonCol_From_mov, latCol_From_mov, lonCol_To_mov, latCol_To_mov])
        coverages = (coverages[0] * coverages_outl[0], coverages[1] * coverages_outl[1])

    df_coverages = pd.DataFrame(coverages)

    D_dist['REAL_DISTANCE'] = np.nan
    D_dist['MERCATOR_X_FROM'] = np.nan
    D_dist['MERCATOR_Y_FROM'] = np.nan
    D_dist['MERCATOR_X_TO'] = np.nan
    D_dist['MERCATOR_Y_TO'] = np.nan

    for index, row in D_dist.iterrows():

        # get the coordinates
        lonFrom = row[lonCol_From_mov]
        latFrom = row[latCol_From_mov]
        lonTo = row[lonCol_To_mov]
        latTo = row[latCol_To_mov]

        # get the closest node on the graph
        node_from = ox.get_nearest_node(G, (latFrom, lonFrom), method='euclidean')
        node_to = ox.get_nearest_node(G, (latTo, lonTo), method='euclidean')
        length = nx.shortest_path_length(G=G, source=node_from, target=node_to, weight='length')
        D_dist['REAL_DISTANCE'].loc[index] = length

        # convert into mercator coordinates
        x_merc_from, y_merc_from = mercatorProjection(latFrom, lonFrom)
        x_merc_to, y_merc_to = mercatorProjection(latTo, lonTo)

        D_dist['MERCATOR_X_FROM'].loc[index] = x_merc_from
        D_dist['MERCATOR_Y_FROM'].loc[index] = y_merc_from
        D_dist['MERCATOR_X_TO'].loc[index] = x_merc_to
        D_dist['MERCATOR_Y_TO'].loc[index] = y_merc_to

    D_dist['EUCLIDEAN_DISTANCE'] = 1000 * func_euclideanDistanceCost(D_dist['MERCATOR_X_FROM'],
                                                                     D_dist['MERCATOR_Y_FROM'],
                                                                     D_dist['MERCATOR_X_TO'],
                                                                     D_dist['MERCATOR_Y_TO'],
                                                                     1)
    D_dist['RECTANGULAR_DISTANCE'] = 1000 * func_rectangularDistanceCost(D_dist['MERCATOR_X_FROM'],
                                                                         D_dist['MERCATOR_Y_FROM'],
                                                                         D_dist['MERCATOR_X_TO'],
                                                                         D_dist['MERCATOR_Y_TO'],
                                                                         1)
    D_dist['GRAVITY_DISTANCE'] = 1000 * func_gravityDistanceCost(D_dist['MERCATOR_X_FROM'],
                                                                 D_dist['MERCATOR_Y_FROM'],
                                                                 D_dist['MERCATOR_X_TO'],
                                                                 D_dist['MERCATOR_Y_TO'],
                                                                 1)

    error_euclidean = mean_squared_error(D_dist['REAL_DISTANCE'], D_dist['EUCLIDEAN_DISTANCE'])
    error_rectangular = mean_squared_error(D_dist['REAL_DISTANCE'], D_dist['RECTANGULAR_DISTANCE'])
    error_gravity = mean_squared_error(D_dist['REAL_DISTANCE'], D_dist['GRAVITY_DISTANCE'])

    print(f"MSE EUCLIDEAN: {np.round(error_euclidean,2)}")
    print(f"MSE RECTANGULAR: {np.round(error_rectangular,2)}")
    print(f"MSE GRAVITY: {np.round(error_gravity,2)}")
    return D_dist, df_coverages


def calculateOptimalLocation(D_table: pd.DataFrame,
                             timeColumns: list,
                             distanceType: str,
                             latCol: str,
                             lonCol: str,
                             codeCol_node: str,
                             descrCol_node: str,
                             cleanOutliers: bool = False):
    """
    this function import a table D_table where each row is a node of the network

    Args:
        D_table (pd.DataFrame): DESCRIPTION.
        timeColumns (list): list of the column name with the time horizon containing quantity data.
        distanceType (str): type of distance to consider for optimization.
        latCol (str): column name of the latitude of the node.
        lonCol (str): column name of the longitude of the node.
        codeCol_node (str): column with description of the node (the same appearing in plantListName).
        descrCol_node (str): column with description of the node.
        cleanOutliers (bool, optional): if True use IQR to remove latitude and longitude outliers. Defaults to False.

    Returns:
        D_res (pd.DataFrame): it returns a dataframe D_res with the ID, LATITUDE, LONGITUDE AND YEAR for
        each flow adding the column COST AND FLOW representing the distance  travelled (COST) and the flow intensity (FLOW).
        The column COST_NORM is a the flows scaled between 0 and 100.
        D_res_optimal (pd.DataFram): it returns a dataframe D_res_optimal with the loptimal latitude and longitude for each
        time frame, and a column COST and FLOW with the total cost (distance) and flows.
        output_coverages (TYPE): DESCRIPTION.

    """

    # clean data and calculate coverages
    output_coverages = {}

    analysisFieldList = [latCol, lonCol]
    outputCoverages, _ = getCoverageStats(D_table, analysisFieldList, capacityField=timeColumns[0])
    D_table = D_table.dropna(subset=[latCol, lonCol])
    if cleanOutliers:
        D_table, coverages, = cleanUsingIQR(D_table, [latCol, lonCol])
        outputCoverages = (coverages[0] * outputCoverages[0], coverages[1] * outputCoverages[1])
    output_coverages['coverages'] = pd.DataFrame(outputCoverages)

    # fill nan values
    D_table = D_table.fillna(0)

    # identify years in the column
    yearsColumns = timeColumns

    # identify useful columns
    D_res = pd.DataFrame(columns=[codeCol_node, descrCol_node, latCol, lonCol, 'YEAR', 'COST'])
    D_res_optimal = pd.DataFrame(columns=['PERIOD', latCol, lonCol, 'YEAR', 'COST', 'FLOW'])

    for year in yearsColumns:
        # year = yearsColumns[0]
        D_filter_columns = [codeCol_node, descrCol_node, latCol, lonCol, year]
        D_filtered = D_table[D_filter_columns]
        D_filtered = D_filtered.rename(columns={year: 'FLOW'})
        D_filtered['YEAR'] = year

        # define optimal location
        if distanceType.lower() == 'rectangular':
            lat_optimal, lon_optimal = optimalLocationRectangularDistance(D_filtered, latCol, lonCol, 'FLOW')
            D_filtered['COST'] = func_rectangularDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
        elif distanceType.lower() == 'gravity':
            lat_optimal, lon_optimal = optimalLocationGravityProblem(D_filtered, latCol, lonCol, 'FLOW')
            D_filtered['COST'] = func_gravityDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
        elif distanceType.lower() == 'euclidean':
            lat_optimal, lon_optimal = optimalLocationEuclideanDistance(D_filtered, latCol, lonCol, 'FLOW')
            D_filtered['COST'] = func_euclideanDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
        D_res = D_res.append(D_filtered)

        D_res_optimal = D_res_optimal.append(pd.DataFrame([[f"OPTIMAL LOCATION YEAR: {year}",
                                                            lat_optimal,
                                                            lon_optimal,
                                                            year,
                                                            sum(D_res['COST']),
                                                            sum(D_res['FLOW']),
                                                            ]], columns=D_res_optimal.columns))

    # D_res['COST_norm']=(D_res['COST']-min(D_res['COST']))/(max(D_res['COST'])-min(D_res['COST']))*10
    D_res['FLOW_norm'] = (D_res['FLOW'] - min(D_res['FLOW'])) / (max(D_res['FLOW']) - min(D_res['FLOW'])) * 100

    D_res = D_res.rename(columns={'COST': 'COST_TOBE'})

    return D_res, D_res_optimal, output_coverages


def calculateMultipleOptimalLocation(D_table: pd.DataFrame,
                                     timeColumns: list,
                                     distanceType: str,
                                     latCol: str,
                                     lonCol: str,
                                     codeCol_node: str,
                                     descrCol_node: str,
                                     cleanOutliers: bool = False,
                                     k: int = 1,
                                     method: str = 'kmeans'):
    """
    this function defines k facility location using an aggregation method

    Args:
        D_table (pd.DataFrame): DESCRIPTION.
        timeColumns (list): list of the column name with the time horizon containing quantity data.
        distanceType (str): type of distance to consider for optimization.
        latCol (str): column name of the latitude of the node.
        lonCol (str): column name of the longitude of the node.
        codeCol_node (str): column with description of the node (the same appearing in plantListName).
        descrCol_node (str): column with description of the node.
        cleanOutliers (bool, optional): if True use IQR to remove latitude and longitude outliers. Defaults to False.
        k (int, optional): Number of clusters to define. Defaults to 1.
        method (str, optional): Clustering method to use (e.g. kmeans). Defaults to 'kmeans'.

    Returns:
        D_res (pd.DataFrame): it returns a dataframe D_res with the ID, LATITUDE, LONGITUDE AND YEAR for
        each flow adding the column COST AND FLOW representing the distance  travelled (COST) and the flow intensity (FLOW).
        The column COST_NORM is a the flows scaled between 0 and 100.
        D_res_optimal (pd.DataFram): it returns a dataframe D_res_optimal with the loptimal latitude and longitude for each
        time frame, and a column COST and FLOW with the total cost (distance) and flows.
        output_coverages (TYPE): DESCRIPTION.

    """

    # clean data and calculate coverages
    output_coverages = {}

    analysisFieldList = [latCol, lonCol]
    outputCoverages, _ = getCoverageStats(D_table, analysisFieldList, capacityField=timeColumns[0])
    D_table = D_table.dropna(subset=[latCol, lonCol])
    if cleanOutliers:
        D_table, coverages, = cleanUsingIQR(D_table, [latCol, lonCol])
        outputCoverages = (coverages[0] * outputCoverages[0], coverages[1] * outputCoverages[1])
    output_coverages['coverages'] = pd.DataFrame(outputCoverages)

    # fill nan values
    D_table = D_table.fillna(0)

    # identify years in the columns
    yearsColumns = timeColumns

    # cluster points
    if method == 'kmeans':
        km = cluster.KMeans(n_clusters=k).fit(D_table[[latCol, lonCol]])
        D_table['CLUSTER'] = pd.DataFrame(km.labels_)

    elif method == 'gmm':
        gmm = GaussianMixture(n_components=k, covariance_type='full').fit(D_table[[latCol, lonCol]])
        D_table['CLUSTER'] = pd.DataFrame(gmm.predict(D_table[[latCol, lonCol]]))
    else:
        print("No valid clustering method")
        return [], [], []

    # identify useful columns
    D_res = pd.DataFrame(columns=[codeCol_node, descrCol_node, latCol, lonCol, 'YEAR', 'COST', 'CLUSTER'])
    D_res_optimal = pd.DataFrame(columns=['PERIOD', latCol, lonCol, 'YEAR', 'COST', 'FLOW', 'CLUSTER'])

    # analyse each cluster separately
    for cluster_id in set(D_table['CLUSTER']):
        # cluster_id=0
        D_table_filtered = D_table[D_table['CLUSTER'] == cluster_id]
        for year in yearsColumns:
            # year = yearsColumns[0]
            D_filter_columns = [codeCol_node, descrCol_node, latCol, lonCol, year, 'CLUSTER']
            D_filtered = D_table_filtered[D_filter_columns]
            D_filtered = D_filtered.rename(columns={year: 'FLOW'})
            D_filtered['YEAR'] = year

            # define optimal location
            if distanceType.lower() == 'rectangular':
                lat_optimal, lon_optimal = optimalLocationRectangularDistance(D_filtered, latCol, lonCol, 'FLOW')
                D_filtered['COST'] = func_rectangularDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
            elif distanceType.lower() == 'gravity':
                lat_optimal, lon_optimal = optimalLocationGravityProblem(D_filtered, latCol, lonCol, 'FLOW')
                D_filtered['COST'] = func_gravityDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
            elif distanceType.lower() == 'euclidean':
                lat_optimal, lon_optimal = optimalLocationEuclideanDistance(D_filtered, latCol, lonCol, 'FLOW')
                D_filtered['COST'] = func_euclideanDistanceCost(D_filtered[lonCol], D_filtered[latCol], lon_optimal, lat_optimal, D_filtered['FLOW'])
            D_res = D_res.append(D_filtered)

            D_res_optimal = D_res_optimal.append(pd.DataFrame([[f"OPTIMAL LOCATION YEAR: {year}",
                                                                lat_optimal,
                                                                lon_optimal,
                                                                year,
                                                                sum(D_res['COST']),
                                                                sum(D_res['FLOW']),
                                                                cluster_id
                                                                ]], columns=D_res_optimal.columns))

    D_res['FLOW_norm'] = (D_res['FLOW'] - min(D_res['FLOW'])) / (max(D_res['FLOW']) - min(D_res['FLOW'])) * 100

    D_res = D_res.rename(columns={'COST': 'COST_TOBE'})

    return D_res, D_res_optimal, output_coverages


def calculateCostASIS(D_plant: pd.DataFrame,
                      latCol_plant: str,
                      lonCol_plant: str,
                      plantListName: list,
                      D_node: pd.DataFrame,
                      nodeCol_node: str,
                      latCol_node: str,
                      lonCol_node: str,
                      distanceType: str):
    """
    define the cost as-is of a network, given a cost estimator of the distance function distanceType
    a dataframe D_plant with one row for each plant of the network to evaluate.

    Args:
        D_plant (pd.DataFrame): DESCRIPTION.
        latCol_plant (str): string with the column of D_plant with latitudes.
        lonCol_plant (str): string with the column of D_plant with longitudes.
        plantListName (list): string of the column of D_plant containing a list of all the ids of the nodes served by the plant.
        D_node (pd.DataFrame): input table with one row for each node of the network. Can be a D_res output of the function calculateOptimalLocation.
        nodeCol_node (str): DESCRIPTION.
        latCol_node (str): string with the column of D_node with latitudes.
        lonCol_node (str): string with the column of D_node with longitudes.
        distanceType (str): chosen distance function distanceType='euclidean','gravity','rectangular'.

    Returns:
        D_node (pd.DataFrame): returns node dataframe with additional column for the as-is cost.

    """

    D_node['COST_ASIS'] = np.nan
    D_node = D_node.reset_index(drop=True)

    # assign each pod to the serving facility
    for plant in D_plant['_id']:
        # plant = 652

        plant_client_list = D_plant[D_plant['_id'] == plant][plantListName].iloc[0]
        plant_client_list = [str(i) for i in plant_client_list]

        # consider latitude and longitude of the plant
        lat_plant = D_plant[D_plant['_id'] == plant][latCol_plant].iloc[0]
        lon_plant = D_plant[D_plant['_id'] == plant][lonCol_plant].iloc[0]

        # define a column of the dataframe for the serving plant
        D_node[plant] = [str(id_nodo) in plant_client_list for id_nodo in D_node[nodeCol_node]]

        # D_res['all']=D_res[652].astype(int) + D_res[2615].astype(int) +D_res[603].astype(int) + D_res[610].astype(int)
        D_filtered = D_node[D_node[plant].isin([True])]
        idx_to_upload = D_filtered.index.values

        # identify a distance function
        if distanceType.lower() == 'rectangular':
            func = func_rectangularDistanceCost
        elif distanceType.lower() == 'gravity':
            func = func_gravityDistanceCost
        elif distanceType.lower() == 'euclidean':
            func = func_euclideanDistanceCost

        distancecost = list(func(D_filtered[lonCol_node],
                                 D_filtered[latCol_node],
                                 lon_plant,
                                 lat_plant,
                                 D_filtered['FLOW'])
                            )
        for i in range(0, len(distancecost)):
            D_node['COST_ASIS'].loc[idx_to_upload[i]] = distancecost[i]
    return D_node


def tracciaCurveIsocosto(D_res: pd.DataFrame, D_res_optimal: pd.DataFrame, latCol: str, lonCol: str, distanceType: str,
                         D_plant: pd.DataFrame = [], plantLongitude: str = [], plantLatitude: str = [],
                         roadGraph: nx.Graph = []):
    """
    Produce plots with iso-cost curves

    Args:
        D_res (pd.DataFrame): Input dataframe.
        D_res_optimal (pd.DataFrame): Input dataframe with optimal points.
        latCol (str): column name with latitude.
        lonCol (str): column name with longitude.
        distanceType (str): type of distance cost function.
        D_plant (pd.DataFrame, optional): dataframe containing plant coordinates. Defaults to [].
        plantLongitude (str, optional): column name with latitude. Defaults to [].
        plantLatitude (str, optional): column name with longitude. Defaults to [].
        roadGraph (nx.Graph, optional): Road graph. Defaults to [].

    Returns:
        outputFigure (dict): dictionary of output figures.
        fig_curve_cost3D (plotly.figure): Interactive 3d plot with .

    """

    outputFigure = {}
    X_list = []
    Y_list = []
    grid_list = []
    time_list = []

    year_list = list(set(D_res['YEAR']))
    year_list.sort()
    for year in year_list:
        # year = list(set(D_res['YEAR']))[0]
        D_res_test = D_res[(D_res['FLOW'] > 0) & (D_res['YEAR'] == year)]
        if len(D_res_test) > 2:
            D_res_optimal_filtered = D_res_optimal[D_res_optimal['YEAR'] == year]

            # identify the rectangular to represent
            min_lon = min(D_res_test[lonCol])
            max_lon = max(D_res_test[lonCol])

            min_lat = min(D_res_test[latCol])
            max_lat = max(D_res_test[latCol])

            # define the grid
            lon = np.linspace(min_lon, max_lon, 100)
            lat = np.linspace(min_lat, max_lat, 100)
            X, Y = np.meshgrid(lon, lat)
            xy_coord = list(zip(D_res_test[lonCol], D_res_test[latCol]))

            # interpolate missing points
            grid = griddata(xy_coord, np.array(D_res_test['COST_TOBE']), (X, Y), method='linear')

            # save values for the representation
            X_list.append(X)
            Y_list.append(Y)
            grid_list.append(grid)
            time_list.append(year)

            # if a road graph is given, plot it
            if roadGraph == []:
                fig1 = plt.figure()
                ax = fig1.gca()
            else:
                fig1, ax = ox.plot_graph(roadGraph, bgcolor='k',
                                         node_size=1, node_color='#999999', node_edgecolor='none', node_zorder=2,
                                         edge_color='#555555', edge_linewidth=0.5, edge_alpha=1)
                plt.legend(['Node', 'Edges'])

            im = ax.contour(X, Y, grid, cmap='Reds')

            ax.set_xlabel('LONGITUDE')
            ax.set_ylabel('LATITUDE')
            fig1.colorbar(im, ax=ax)

            ax.set_title(f"Isocost line {distanceType}, period: {year}")

            # represent optimal points
            ax.scatter(D_res_optimal_filtered[lonCol], D_res_optimal_filtered[latCol], 100, marker='^', color='green')
            plt.legend(['Optimal points'])

            # represent as-is points
            if len(D_plant) > 0:
                ax.scatter(D_plant[plantLongitude], D_plant[plantLatitude], 100, marker='s', color='black')
                plt.legend(['Optimal points', 'Actual points'])

            fig1 = ax.figure
            outputFigure[f"isocost_{distanceType}_{year}"] = fig1

            plt.close('all')
    # costruisco il grafico 3d
    fig_curve_cost3D = createFigureWith3Dsurface(X_list, Y_list, grid_list, time_list)
    return outputFigure, fig_curve_cost3D


def calculateSaving(D_res_asis: pd.DataFrame,
                    D_res_tobe: pd.DataFrame,
                    periodCol_asis: str = 'YEAR',
                    periodCol_tobe: str = 'YEAR',
                    costCol_asis: str = 'COST_ASIS',
                    costCol_tobe: str = 'COST_TOBE',
                    title: str = ''
                    ):
    """
    Calculate the saving of a given new configuration

    Args:
        D_res_asis (pd.DataFrame): Input dataframe AS IS.
        D_res_tobe (pd.DataFrame): Input dataframe TO BE.
        periodCol_asis (str, optional): column name with time (AS IS). Defaults to 'YEAR'.
        periodCol_tobe (str, optional): column name with time (TO BE). Defaults to 'YEAR'.
        costCol_asis (str, optional): column name with cost (AS IS). Defaults to 'COST_ASIS'.
        costCol_tobe (str, optional): column name with cost (TO BE). Defaults to 'COST_TOBE'.
        title (str, optional): Figure title. Defaults to ''.

    Returns:
        output_figure (dict): Output dictionary containing figures.
        df_saving (pd.DataFrame): Output dataFrame with saving.

    """

    output_figure = {}
    df_saving = pd.DataFrame()

    D_saving_asis = D_res_asis.groupby(periodCol_asis)[costCol_asis].sum().to_frame()
    D_saving_tobe = D_res_asis.groupby(periodCol_tobe)[costCol_tobe].sum().to_frame()
    D_saving = D_saving_asis.merge(D_saving_tobe, how='left', left_on=periodCol_asis, right_on=periodCol_tobe)

    D_saving['SAVING'] = D_saving[costCol_tobe] / D_saving[costCol_asis]
    fig1 = plt.figure()
    plt.plot(D_saving.index, D_saving['SAVING'])
    plt.title(title)
    plt.xticks(rotation=45)

    output_figure['savingPercentage'] = fig1
    df_saving = pd.DataFrame([np.mean(D_saving['SAVING'])])

    return output_figure, df_saving