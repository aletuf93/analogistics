import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import cluster
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import MinMaxScaler


def groupVariableKMean(D_table: pd.DataFrame, inputColumns: list, k: int) -> pd.DataFrame:
    """
    Perform k-means and return the coordinates of the points in a DataFrame

    Args:
        D_table (pd.DataFrame): input dataframe.
        inputColumns (list): list of columns to consider in the k-means.
        k (int): number of groups to create.

    Returns:
        D_table (pd.DataFrame): output DataFrame.

    """

    X = D_table[inputColumns]
    km = cluster.KMeans(n_clusters=k).fit(X)
    D_table[f"CLUSTER_KMEANS_{str(k)}"] = [i for i in km.labels_]
    return D_table


def GroupingVariableGMM(D_table: pd.DataFrame, inputColumns: list, k: int) -> pd.DataFrame:
    """
    Perform Gaussian Mixture Model and return the coordinates of the points in a DataFrame

    Args:
        D_table (pd.DataFrame): input dataframe.
        inputColumns (list): list of columns to consider in the clustering.
        k (int): number of groups to create.

    Returns:
        D_table (pd.DataFrame): output DataFrame.

    """
    X = D_table[inputColumns]
    gmm = GaussianMixture(n_components=k, covariance_type='full').fit(X)
    D_table[f"CLUSTER_GMM_{str(k)}"] = [i for i in gmm.predict(X)]
    return D_table


def GroupingVariableHierarchical(D_table: pd.DataFrame, inputColumns: list,
                                 k: int, grouping_method: str) -> pd.DataFrame:
    """
    Perform Hierarchical Clustering and return the coordinates of the points in a DataFrame

    Args:
        D_table (pd.DataFrame): input dataframe.
        inputColumns (list): list of columns to consider in the clustering.
        k (int): number of groups to create.
        grouping_method (str): distance metric to consider.

    Returns:
        D_table (TYPE): output DataFrame.

    """
    X = D_table[inputColumns]
    hierCl = cluster.AgglomerativeClustering(n_clusters=k, linkage=grouping_method).fit(X)
    D_table[f"CLUSTER_HIER_{str(k)}"] = pd.DataFrame(hierCl.labels_)
    return D_table


def HierarchicalClusterJaccard(D_table: pd.DataFrame, targetColumn: str,
                               k: int, groupingMethod: str):
    """
    Performs hierarchical clustering from an incidence matrix

    Args:
        D_table (pd.DataFrame): table with n rows, one rows for each item to cluster.
        targetColumn (str): column containing all the observed values for each item.
        k (int): number of clusters to generate.
        groupingMethod (str): specifies the type of linkage to use in hierarchical clustering in ('single','complete','average').

    Returns:
        D_table (TYPE): initial dataframe with an additional column containing the clusters.

    """

    D_Sim = D_table[targetColumn].str.get_dummies(sep=';')
    for j in D_Sim.columns:
        D_Sim[j] = D_Sim[j].astype(bool)

    Y = pdist(D_Sim, 'jaccard')
    Y[np.isnan(Y)] = 0
    z = linkage(Y, method=groupingMethod)

    cutree = cut_tree(z, n_clusters=k)
    D_table[f"CLUSTER_HIER_JAC_{str(k)}"] = pd.DataFrame(cutree)

    return D_table


def HierarchicalClusteringDendrogram(X: pd.DataFrame, grouping_method: str, distance_method: str) -> dict:
    """
    Plot the dendrogram of the hierarchical clustering

    Args:
        X (pd.DataFrame): input DataFrame.
        grouping_method (str): linkage method ('single','complete','average').
        distance_method (str): distance metric.

    Returns:
        dict: DESCRIPTION.

    """
    output_figure = {}

    fig1 = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    res = pdist(X, distance_method)

    # define linkage
    Z = linkage(res, method=grouping_method, metric=distance_method)
    plt.title(f"Hierarchical Clustering Dendrogram, {grouping_method} linkage, {distance_method} distance")
    plt.xlabel('item')
    plt.ylabel('similarity')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    output_figure['dendrogram'] = fig1
    plt.close('all')
    return output_figure


def capacitatedClustering(D: pd.DataFrame, simMin: float,
                          dem: np.array, capacity: float) -> pd.DataFrame:
    """
    Greedy capacitated clustering algorithm, based on gierarchical clustering.

    Args:
        D (pd.DataFrame): array of coordinates (nxm), n= number of points or observations.
        simMin (float): minimum similarity vlaue to group two points together (0->1).
        dem (np.array): array of the demand of dimension n.
        capacity (float): maximum capacity for each cluster.

    Returns:
        capCluster (TYPE): array of dimension n with the code of the cluster for each observation.

    """

    method = 'single'
    select = len(D)

    # Consider the distance matrix
    M = squareform(pdist(D))

    # scale to a proximity matrix
    scaler = MinMaxScaler()
    scaler.fit(M)
    M = scaler.transform(M)
    M = 1 - M

    # Set the diagonal to zero to avoid self-clustering
    np.fill_diagonal(M, 0)

    # start clustering loop
    progressivoCluster = 0
    capCluster = np.zeros(select)
    capSatura = False

    while not(capSatura):
        progressivoCluster = progressivoCluster + 1

        # rank all the points
        simOrdered = np.unique(np.reshape(M, [M.size, 1]))
        simOrdered = np.sort(simOrdered)
        simOrdered = simOrdered[simOrdered >= simMin]
        simOrdered = simOrdered[::-1]  # rank descending

        if(len(simOrdered) == 0):  # if no candidate left, then finish
            capSatura = True
        trovato = False

        while ((not(trovato)) & (not(capSatura))):  # go on looping while the capacity is not saturated and the following node to group has been found

            for gg in range(0, len(simOrdered)):  # consider all the points, icluded the first (equal to 1)
                if((not(trovato)) & (not(capSatura))):
                    simValue = simOrdered[gg]  # scan all the similarity values
                    inc = np.where(M == simValue)  # find all the rows and columns with the same similarity value

                    # scan all the nodes with the same values
                    for jj in range(0, len(inc[0])):
                        if((not(trovato)) & (not(capSatura))):
                            max_id_row = inc[0][jj]  # row with node candidate to be aggregated (1)
                            max_id_column = inc[1][jj]  # column with node candidate to be aggregated (2)

                            # check the capacity constraint

                            # Identify belnging to the 1 candidate
                            # find in capCluster the assignment of node 1 (use 0 if never assigned to a cluster)
                            currentId1 = capCluster[max_id_row]
                            if(not(currentId1 == 0)):  # if already assigned to a cluster, inherit all the nodes of that cluster
                                currentId1 = capCluster == currentId1
                            else:  # otherwise, assign a zero vector wit a one corresponding to node 1
                                currentId1 = np.zeros(len(capCluster))
                                currentId1[max_id_row] = 1

                            # Identify belnging to the 2 candidate
                            # find in capCluster the assignment of node 2 (use 0 if never assigned to a cluster)
                            currentId2 = capCluster[max_id_column]
                            if(not(currentId2 == 0)):  # if already assigned to a cluster, inherit all the nodes of that cluster
                                currentId2 = capCluster == currentId2
                            else:  # otherwise, assign a zero vector wit a one corresponding to node 1
                                currentId2 = np.zeros(len(capCluster))
                                currentId2[max_id_column] = 1

                        totalCapacity = currentId1 + currentId2
                        totalCapacity = sum(dem * totalCapacity)  # aggregate the capacity of all the nodes candidate to aggregation
                        if(totalCapacity < capacity):  # if capacity is respected, the aggregate
                            trovato = True

                    if((gg == len(simOrdered) - 1) & (not(trovato))):  # if all the loops have been finished and there is no aggregation, then exit with the current assignment
                        capSatura = True

        # if a cluster has been foud, update the similarity values
        if (not(capSatura)):

            # identify cluster 1
            currentiId1 = capCluster[max_id_row]
            if(currentiId1 == 0):  # if never aggregated, update only that node
                capCluster[max_id_row] = progressivoCluster
            else:
                capCluster[capCluster == currentiId1] = progressivoCluster

            # identify cluster 2
            currentiId2 = capCluster[max_id_column]
            if(currentiId2 == 0):  # if never aggregated, update only that node
                capCluster[max_id_column] = progressivoCluster
            else:
                capCluster[capCluster == currentiId2] = progressivoCluster

            # update the similarity values in the matrix

            for h in range(0, len(M)):  # scan all the rows of the matrix

                # update all the indixes of the column
                if((h != max_id_column) & (h != max_id_row)):  # do not update the diagonal
                    if method == 'single':
                        M[h, max_id_column] = min(M[h, max_id_column], M[h, max_id_row])
                    # elif method=='complete':
                    #    M[h,max_id_column]=max(M[h,max_id_column],M[h,max_id_row])
                    # elif method=='average':
                    #    M[h,max_id_column]=np.mean(M[h,max_id_column],M[h,max_id_row])
                    M[max_id_column, h] = M[h, max_id_column]  # make the maric symmetric

                # update all the indixes of the rows
                if((h != max_id_row) & (h != max_id_column)):  # do not update the diagonal
                    if method == 'single':
                        M[h, max_id_row] = min(M[h, max_id_row], M[h, max_id_column])
                    # elif method=='complete':
                    #    M[h,max_id_row]=max(M[h,max_id_row],M[h,max_id_column])
                    # elif method=='average':
                    #    M[h,max_id_row]=np.mean(M[h,max_id_row],M[h,max_id_column])
                    M[max_id_row, h] = M[h, max_id_row]  # make the maric symmetric

            # Make zero the similarity values of the nodes aggregated to avoit to aggregate them again
            M[max_id_row, max_id_column] = 0
            M[max_id_column, max_id_row] = 0

    # if when finished some nodes are still zero (unassigned), the assign to different clusters
    for jj in range(0, len(capCluster)):
        if(capCluster[jj] == 0):
            capCluster[jj] = progressivoCluster
            progressivoCluster = progressivoCluster + 1
    return capCluster
