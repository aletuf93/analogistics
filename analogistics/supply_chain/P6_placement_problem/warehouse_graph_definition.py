
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math


import analogistics.graph.graph as dg
from analogistics.clean import cleanUsingIQR


def defineCoordinatesFromRackBayLevel(D_layout: pd.DataFrame, aisleX: float = 5.0, bayY: float = 0.9):
    """
    Define the cartesian coordinates of a warehouse location, based on the number of bay, rack (aisle), and level.

    Args:
        D_layout (pd.DataFrame): Input layout dataframe.
        aisleX (float, optional): Lenght of an aisle (in meters, or coherent uom with D_layout). Defaults to 5.0.
        bayY (float, optional): Length of a bay (in meters, or coherent uom with D_layout). Defaults to 0.9.

    Returns:
        D_layout (pd.DataFrame): Output dataFrame with coordinates.

    """

    print(f"Assuming aisle width of {aisleX} meters and bay width (pallet) of {bayY} meters")

    # identify aisles
    D_layout['loccodex'] = -1
    D_layout['loccodey'] = -1
    allAisles = list(set(D_layout.rack))
    allAisles.sort()
    j = 0
    # scan all aisles
    for x in allAisles:
        # assign x coordinate based on the distance between aisles
        idx_x = D_layout.rack == x
        D_layout['loccodex'].loc[idx_x] = aisleX * j
        j = j + 1

        # identify all the bays of an aisle
        allBays = list(set(D_layout['bay'].loc[idx_x]))
        i = 0
        for y in allBays:
            # assign y coordinate based on the distance between bays
            # hypothesis: all bays are based on the warehouse front
            idx_y = (D_layout.rack == x) & (D_layout.bay == y)
            D_layout['loccodey'].loc[idx_y] = bayY * i
            i = i + 1
    return D_layout


def estimateMissingAislecoordX(D_layout: pd.DataFrame) -> pd.DataFrame:
    """
    estimate the values of the aisle coordinate, when not mapped ("aislecodex" column of the dataframe D_layout)

    Args:
        D_layout (pd.DataFrame): Input dataFrame.

    Returns:
        D_layout (pd.DataFrame): Output dataframe with coordinates.

    """

    # ####################################################
    # ### fix nan in loccodex e loccodey #################
    # ####################################################
    D_layout = D_layout.reset_index()

    # if rack information are given
    if 'rack' in D_layout.columns:
        D_layout = D_layout.sort_values(['rack', 'bay'], ascending=[True, True])
        allRacks = list(set(D_layout.rack.dropna()))
        for rack in allRacks:
            D_rack = D_layout[D_layout.rack == rack]

            # try to calculate the average value for the rack
            avgXCoord = np.mean(D_rack.loccodex)
            if not(math.isnan(avgXCoord)):  # if a value is found
                D_rack['loccodex'].fillna(avgXCoord, inplace=True)

            else:  # otherwise search within the neighborhood, and interpolate
                D_rack_null = D_layout[['rack', 'loccodex']].drop_duplicates()
                D_rack_null = D_rack_null.sort_values('rack')
                D_rack_null['loccodex'].fillna(method='backfill', inplace=True)
                fillValue = float(D_rack_null[D_rack_null.rack == rack].loccodex)
                # Then, substitute values
                D_rack['loccodex'].fillna(fillValue, inplace=True)

            # set aisles coordinates based on nearest neighbor
            D_rack['loccodey'].interpolate(method='linear', limit_direction='forward', inplace=True)

            # update D_layout
            D_layout.loc[D_rack.index] = D_rack

        # delete the remaining nan
        D_layout = D_layout.sort_values(by=['rack', 'bay'])
        print(f"====={len(D_layout[D_layout.loccodex.isnull()])} x coordinates have been randomly interpolated")
        D_layout['loccodex'].fillna(method='ffill', inplace=True)  # fill scanning forward
        D_layout['loccodex'].fillna(method='bfill', inplace=True)  # fill scanning backward

    else:
        print("No rack information")

    # ####################################################
    # ##### estimate coordinates of the missing racks ####
    # ####################################################

    # identify mapped aisle coordinates (aislecodex)
    D_givAisl = D_layout[D_layout['aislecodex'].notna()]
    D_givAisl = D_givAisl[['loccodex', 'aislecodex']]
    D_givAisl = D_givAisl.drop_duplicates()

    # identify coordinates to map
    D_estAisl = D_layout[D_layout['loccodex'].notna()].loccodex
    allXcoords = list(set(D_estAisl))
    allXcoords.sort()

    # join the coordinates, and put the farthest in the same aisle
    dist = 0
    for j in range(1, len(allXcoords)):
        dist = dist + np.abs(allXcoords[j] - allXcoords[j - 1])
    if len(allXcoords) > 1:
        avg_dist = dist / (len(allXcoords) - 1)
    else:
        avg_dist = 0

    # if the distance is above the average, join in the same aisle
    D_estAisl = pd.DataFrame(columns=D_givAisl.columns)
    j = 0
    while j < len(allXcoords):
        if j < len(allXcoords) - 1:  # for each aisle, except the last
            dist = np.abs(allXcoords[j + 1] - allXcoords[j])
            if dist >= avg_dist:  # if they are greater or equal than the average, theiy are on the same aisle
                aisle = min(allXcoords[j + 1], allXcoords[j]) + dist / 2
                D_estAisl = D_estAisl.append(pd.DataFrame([[allXcoords[j], aisle]], columns=D_estAisl.columns))
                D_estAisl = D_estAisl.append(pd.DataFrame([[allXcoords[j + 1], aisle]], columns=D_estAisl.columns))
                j = j + 2  # joined two, jumo two
            else:  # otherwise, it is a single aisle
                D_estAisl = D_estAisl.append(pd.DataFrame([[allXcoords[j], allXcoords[j]]], columns=D_estAisl.columns))
                j = j + 1  # joined one, jumo one
        elif j == len(allXcoords) - 1:  # if it is the last aisle
            D_estAisl = D_estAisl.append(pd.DataFrame([[allXcoords[j], allXcoords[j]]], columns=D_estAisl.columns))
            j = j + 1  # joined one, jump one

    #  data cleaning
    # replace None with nan
    D_layout.replace(to_replace=[None], value=np.nan, inplace=True)
    # check null aisle values
    index = D_layout['aislecodex'].index[D_layout['aislecodex'].apply(np.isnan)]

    for rows in index:
        loccodex = D_layout.loc[rows].loccodex

        # if the value is known
        if loccodex in D_givAisl.loccodex:
            D_layout['aislecodex'].loc[rows] = float(D_givAisl[D_givAisl['loccodex'] == loccodex].aislecodex)
        else:
            D_layout['aislecodex'].loc[rows] = float(D_estAisl[D_estAisl['loccodex'] == loccodex].aislecodex)

    # check if coordinates exist otherwise replace with rack/bay/level

    # remove rack/bay/level
    if 'rack' in D_layout.columns:
        D_layout = D_layout.sort_values(by=['rack', 'bay'])
    else:
        D_layout = D_layout.sort_values(by=['aislecodex'])
    D_layout = D_layout[['idlocation', 'aislecodex', 'loccodex', 'loccodey']]

    # interpolate missing y-coordinate

    print(f"====={len(D_layout[D_layout.loccodey.isnull()])} y coordinates have been randomly interpolated")
    D_layout['loccodey'].interpolate(method='linear', limit_direction='forward', inplace=True)
    D_layout['loccodey'].fillna(method='ffill', inplace=True)  # fill scanning forward
    D_layout['loccodey'].fillna(method='bfill', inplace=True)  # fill scanning backward

    # round everithing avoiding decimal values
    D_layout['aislecodex'] = np.round(D_layout['aislecodex'], 0)
    D_layout['loccodey'] = np.round(D_layout['loccodey'], 0)

    return D_layout


def defineGraphNodes(D_layout: pd.DataFrame, D_IO: pd.DataFrame):
    """
    Define correspondence between idlocation and node id (graph)

    Args:
        D_layout (pd.DataFrame): Input layout dataframe.
        D_IO (pd.DataFrame): Input I/O locations dataframe.

    Returns:
        D_nodes (pd.DataFrame): Output nodes dataframe.
        D_res_dict (dict): dictionaty with correspondence between idlocation and nodeid.
        D_IO (pd.DataFrame): Output I/O dataframe.

    """

    #  define all the nodes of the graph
    D_nodes = D_layout[['aislecodex', 'loccodey']].drop_duplicates().reset_index(drop=True)

    # add correspondence between D_layout and D_nodes
    D_layout['idNode'] = None
    for index, node in D_nodes.iterrows():
        idx_node = (D_layout.aislecodex == node.aislecodex) & (D_layout.loccodey == node.loccodey)
        D_layout.idNode.loc[idx_node] = index

    # add Input-Output nodes
    # redefine index of D_IO to avoid overlaps with D_nodes
    D_IO.index = np.arange(max(D_nodes.index.values) + 1, max(D_nodes.index.values) + 1 + len(D_IO))

    for index, node in D_IO.iterrows():
        idx_node = node.idlocation  # use id location of the fake locations
        temp = pd.DataFrame([[idx_node, node.loccodex, node.loccodex, node.loccodey, index]],
                            columns=D_layout.columns)
        D_layout = D_layout.append(temp)

    D_res = D_layout[['idlocation', 'idNode']]
    D_res = D_res.drop_duplicates()

    D_res_dict = dict(zip(D_res.idlocation, D_res.idNode))

    return D_nodes, D_res_dict, D_IO


def addtraversaledges(D_nodes: pd.DataFrame, list_aisles: list, edgeTable: pd.DataFrame,
                      columns_edgeTable: list, index_source: list, index_target: list) -> pd.DataFrame:
    """
    Add the traversal edges to connect aisles horizontally

    Args:
        D_nodes (pd.DataFrame): Input nodes DataGrame.
        list_aisles (list): list of aisles.
        edgeTable (pd.DataFrame): Input dataframe with edgetable.
        columns_edgeTable (list): DESCRIPTION.
        index_source (list): DESCRIPTION.
        index_target (list): DESCRIPTION.

    Returns:
        edgeTable (pd.DataFrame): Output Dataframe wth edgetable containing horizontal arcs.

    """
    D_Aisle1 = D_nodes[D_nodes.aislecodex == list_aisles[index_source]]  # identify coordinates of the first aisle
    D_Aisle2 = D_nodes[D_nodes.aislecodex == list_aisles[index_target]]  # identify coordinates of the first aisle

    # if connecting two aisles with more than a single bay
    if (len(D_Aisle1) > 1) & (len(D_Aisle2) > 1):
        # identify the two bays on the back
        node1_front_index = D_Aisle1['loccodey'].idxmax()
        node2_front_index = D_Aisle2['loccodey'].idxmax()

        # add the arc
        length = np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index] - D_Aisle2.aislecodex.loc[node2_front_index]), 1)
        temp = pd.DataFrame([[node1_front_index, node2_front_index, length]],
                            columns=columns_edgeTable)
        edgeTable = edgeTable.append(temp)

        # identify the two bays on the front
        node1_front_index = D_Aisle1['loccodey'].idxmin()
        node2_front_index = D_Aisle2['loccodey'].idxmin()

        # add the arc
        length = np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index] - D_Aisle2.aislecodex.loc[node2_front_index]), 1)
        temp = pd.DataFrame([[node1_front_index, node2_front_index, length]],
                            columns=columns_edgeTable)
        edgeTable = edgeTable.append(temp)

    else:  # otherwise connect single bays

        if len(D_Aisle1) > 1:  # if the first aisle has more than a single bay

            # identify the coordinates of the first aisle
            node1_back_index = D_Aisle1['loccodey'].idxmax()
            node1_front_index = D_Aisle1['loccodey'].idxmin()

            node2_front_index = D_Aisle2['loccodey'].idxmax()  # return the index of the sigle bay

            # make a single connection to the closer
            length_back = np.round(np.abs(D_Aisle1.aislecodex.loc[node1_back_index] - D_Aisle2.aislecodex.loc[node2_front_index]) + np.abs(D_Aisle1.loccodey.loc[node1_back_index] - D_Aisle2.loccodey.loc[node2_front_index]), 1)
            length_front = np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index] - D_Aisle2.aislecodex.loc[node2_front_index]) + np.abs(D_Aisle1.loccodey.loc[node1_front_index] - D_Aisle2.loccodey.loc[node2_front_index]), 1)

            # if it is shorter on the front, add a single arc
            if length_front <= length_back:
                temp = pd.DataFrame([[node1_front_index, node2_front_index, length_front]],
                                    columns=columns_edgeTable)
                edgeTable = edgeTable.append(temp)
            else:  # otherwise connect backwards
                temp = pd.DataFrame([[node1_back_index, node2_front_index, length_back]],
                                    columns=columns_edgeTable)
                edgeTable = edgeTable.append(temp)

        else:  # all the other cases (bay->bay or bay->aisle)

            # identify the coordinates of the first aisle
            node1_front_index = D_Aisle1['loccodey'].idxmax()

            # identify the coordinates of the second
            node2_back_index = D_Aisle2['loccodey'].idxmax()
            node2_front_index = D_Aisle2['loccodey'].idxmin()

            # make a single connection to the closer
            length_back = np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index] - D_Aisle2.aislecodex.loc[node2_back_index]) + np.abs(D_Aisle1.loccodey.loc[node1_front_index] - D_Aisle2.loccodey.loc[node2_back_index]), 1)
            length_front = np.round(np.abs(D_Aisle1.aislecodex.loc[node1_front_index] - D_Aisle2.aislecodex.loc[node2_front_index]) + np.abs(D_Aisle1.loccodey.loc[node1_front_index] - D_Aisle2.loccodey.loc[node2_front_index]), 1)

            # if it is shorter on the front, add a single arc
            if length_front <= length_back:
                temp = pd.DataFrame([[node1_front_index, node2_front_index, length_front]],
                                    columns=columns_edgeTable)
                edgeTable = edgeTable.append(temp)
            else:  # otherwise connect backwards
                temp = pd.DataFrame([[node1_front_index, node2_back_index, length_back]],
                                    columns=columns_edgeTable)
                edgeTable = edgeTable.append(temp)

    return edgeTable


def defineEdgeTable(D_nodes: pd.DataFrame, D_IO: pd.DataFrame) -> pd.DataFrame:
    """
    Define a dataframe containing the arcs of the warehouse graph

    Args:
        D_nodes (pd.DataFrame): Input DataFrame with nodes.
        D_IO (pd.DataFrame): Input DataFrame with i/o points.

    Returns:
        edgeTable (pd.DataFrame): Output dataframe with arcs.

    """

    # avoid considering - temporarily - i/O and fake locations
    D_fakes = pd.DataFrame(columns=D_nodes.columns)
    for index, row in D_IO.iterrows():
        loccodex = row.loccodex
        loccodey = row.loccodey
        D_fakes = D_fakes.append(D_nodes[((D_nodes.aislecodex == loccodex) & (D_nodes.loccodey == loccodey))])
        D_nodes = D_nodes[~((D_nodes.aislecodex == loccodex) & (D_nodes.loccodey == loccodey))]

    columns_edgeTable = ['nodeFrom', 'nodeTo', 'length']
    edgeTable = pd.DataFrame(columns=columns_edgeTable)

    # #####################################################
    # ##### add vertical arcs(aisles) #####################
    # #####################################################
    set_aisles = set(D_nodes.aislecodex)  # identify all aisles
    for aisle in set_aisles:
        # aisle=list(set_aisles)[0]
        D_currentAisle = D_nodes[D_nodes.aislecodex == aisle]  # filter by aisle
        D_currentAisle = D_currentAisle.sort_values(by='loccodey')  # sort by bay

        # simplify the graph identifying all the bays with the y coordinate on the aisle
        for i in range(1, len(D_currentAisle)):  # identify the arcs

            # identify the parameters of the arcs, and their attributes
            nodeFrom = D_currentAisle.index[i - 1]
            nodeTo = D_currentAisle.index[i]
            length = np.round(np.abs(D_currentAisle.loccodey.iloc[i - 1] - D_currentAisle.loccodey.iloc[i]), 1)

            temp = pd.DataFrame([[nodeFrom, nodeTo, length]],
                                columns=columns_edgeTable)
            edgeTable = edgeTable.append(temp)

    # #####################################################
    # ##### add traversal arcs ############################
    # #####################################################

    list_aisles = list(set_aisles)  # identify the coordinates of each aisle
    list_aisles.sort()  # sort by coordinate
    for i in range(1, len(list_aisles)):
        #  consiser the current index to create an arc with the near aisle
        if i == 1:
            edgeTable = addtraversaledges(D_nodes, list_aisles, edgeTable, columns_edgeTable, i - 1, i)
            if len(list_aisles) > 2:
                edgeTable = addtraversaledges(D_nodes, list_aisles, edgeTable, columns_edgeTable, i - 1, i + 1)

        elif i == len(list_aisles) - 1:
            if len(list_aisles) > 2:
                edgeTable = addtraversaledges(D_nodes, list_aisles, edgeTable, columns_edgeTable, i - 1, i)
                edgeTable = addtraversaledges(D_nodes, list_aisles, edgeTable, columns_edgeTable, i - 1, i - 2)
        else:
            edgeTable = addtraversaledges(D_nodes, list_aisles, edgeTable, columns_edgeTable, i - 1, i)
            if len(list_aisles) > 3:
                edgeTable = addtraversaledges(D_nodes, list_aisles, edgeTable, columns_edgeTable, i - 1, i - 2)
                edgeTable = addtraversaledges(D_nodes, list_aisles, edgeTable, columns_edgeTable, i - 1, i + 1)

    # #####################################################
    # ##### add arcs to the I/O and fake locations ########
    # #####################################################

    # find input
    D_in = D_IO[D_IO.inputloc == 1]
    for idx in D_in.index:
        # identify the coordinates
        loccodex = D_in.loccodex[idx]
        loccodey = D_in.loccodey[idx]

        # identify the closest node
        distanceArray = np.abs(D_nodes.aislecodex - loccodex) + np.abs(D_nodes.loccodey - loccodey)
        idx_min = distanceArray.idxmin()
        length = min(distanceArray)

        # create the arc
        nodeFrom = idx
        nodeTo = idx_min
        temp = pd.DataFrame([[nodeFrom, nodeTo, length]],
                            columns=columns_edgeTable)
        edgeTable = edgeTable.append(temp)

        # identify fake locations mapped on the same coordinates
        for idx_fake, row_fake in D_fakes.iterrows():
            #  if a fake is in the same location of an I/O coordinate
            if ((row_fake.aislecodex == loccodex) & (row_fake.loccodey == loccodey)):
                # add the arc
                nodeFrom = idx_fake
                temp = pd.DataFrame([[nodeFrom, nodeTo, 0]],
                                    columns=columns_edgeTable)
                edgeTable = edgeTable.append(temp)

    # find output
    D_out = D_IO[D_IO.outputloc == 1]
    for idx in D_out.index:
        # identify the coordinates
        loccodex = D_out.loccodex[idx]
        loccodey = D_out.loccodey[idx]

        # identify the closest node
        distanceArray = np.abs(D_nodes.aislecodex - loccodex) + np.abs(D_nodes.loccodey - loccodey)
        idx_min = distanceArray.idxmin()
        length = min(distanceArray)

        # create the arc
        nodeFrom = idx
        nodeTo = idx_min
        temp = pd.DataFrame([[nodeFrom, nodeTo, length]],
                            columns=columns_edgeTable)
        edgeTable = edgeTable.append(temp)

        # identify fake locations mapped on the same coordinates
        for idx_fake, row_fake in D_fakes.iterrows():
            # if a fake is mapped on a I/O location
            if ((row_fake.aislecodex == loccodex) & (row_fake.loccodey == loccodey)):

                # add the arc
                nodeFrom = idx_fake
                temp = pd.DataFrame([[nodeFrom, nodeTo, 0]],
                                    columns=columns_edgeTable)
                edgeTable = edgeTable.append(temp)
    edgeTable = edgeTable.drop_duplicates()
    return edgeTable


def analyseWhTraffic(D_mov_input: pd.DataFrame, D_res: pd.DataFrame, G, numPicks: int = -1,
                     edgePredecessors: bool = True, D_layout: pd.DataFrame = []):
    """
    analyse the traffic of a warehouse with a simulation on a sample of picking lists

    Args:
        D_mov_input (pd.DataFrame): dataframe containing the columns IDLOCATION and PICKINGLIST.
        D_res (pd.DataFrame): dictionary matching idlocation with node of the graph.
        G (TYPE): graph of the storage system.
        numPicks (int, optional): number of picks to simulate. Defaults to -1.
        edgePredecessors (bool, optional): if true save the path of arcs for each picking lists (to define traffic chart). Defaults to True.
        D_layout (pd.dataFrame, optional): considered when edgepredecessor is true to define the traffic chart. Defaults to [].

    Returns:
        D_stat_arcs (pd.DataFrame): Output DataFrame with traffic.
        D_stat_picks (pd.DataFrame): Output DataFrame with statistics .

    """

    # rename column with capital letters
    D_mov = D_mov_input
    D_mov = D_mov.rename(columns={'IDLOCATION': 'idlocation',
                                  'PICKINGLIST': 'pickinglist'})
    # get IO nodes
    inputloc = nx.get_node_attributes(G, 'input')
    outputloc = nx.get_node_attributes(G, 'output')

    inputloc = list(inputloc.keys())[0]
    outputloc = list(outputloc.keys())[0]

    # check all locations in D_res
    print(f"There are {len(D_mov.loc[~(D_mov.idlocation.isin(D_res.keys()))])} unmapped locations")

    # chek if pickinglist are available
    picklists = list(set(D_mov.pickinglist))
    if len(picklists) < 2:  # set pickinglist on ordercode
        D_mov['pickinglist'] = D_mov.ordercode
        picklists = list(set(D_mov.pickinglist))
        print("====WARNING===== No pickinglists recorded. Using ordercode =========")

    if numPicks == -1:
        numPicks = len(picklists)

    cols_res = ['pickinglist', 'distance']
    D_stat_order = pd.DataFrame(columns=cols_res)  # dataframe to save statistics on the distances
    D_arcs = pd.DataFrame(columns=['nodeFrom', 'nodeTo'])  # dataframe to save statistics on the traffic

    # bootstrap pickinglist
    np.random.seed(42)
    pickups = np.random.randint(0, len(picklists), size=numPicks)
    count = 0
    for k in range(0, len(pickups)):
        pick = picklists[pickups[k]]

        count = count + 1
        print(f"{count/len(pickups)}")
        D_list = D_mov[D_mov.pickinglist == pick]

        #  check all idlocations in the pickinglist have been mapped
        if all(D_list.idlocation.isin(list(D_res.keys()))) & len(D_list) > 0:
            # scan all the orderlist and define the shortest path
            for i in range(0, len(D_list) + 1):

                # if it is the first row of a picking list
                if i == 0:
                    nodeFrom = inputloc
                    nodeTo = D_res[D_list.idlocation.iloc[i]]
                    if edgePredecessors:
                        path = nx.shortest_path(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                    dist = nx.shortest_path_length(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')

                # if it is the last row of a picking list
                elif i == len(D_list):
                    nodeFrom = D_res[D_list.idlocation.iloc[i - 1]]
                    nodeTo = outputloc
                    if edgePredecessors:
                        path = nx.shortest_path(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                    dist = nx.shortest_path_length(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                else:
                    nodeFrom = D_res[D_list.idlocation.iloc[i - 1]]
                    nodeTo = D_res[D_list.idlocation.iloc[i]]
                    if edgePredecessors:
                        path = nx.shortest_path(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')
                    dist = nx.shortest_path_length(G, source=nodeFrom, target=nodeTo, weight='length', method='dijkstra')

                # save dataframe with results
                temp = pd.DataFrame([[pick, dist]], columns=cols_res)
                D_stat_order = D_stat_order.append(temp)

                if edgePredecessors:
                    for j in range(1, len(path)):
                        temp = pd.DataFrame([[path[j - 1], path[j]]], columns=D_arcs.columns)
                        D_arcs = D_arcs.append(temp)

        else:
            print("Idlocations not found:")
            print(pick)
            print(D_list.idlocation.loc[~D_list.idlocation.isin(list(D_res.values()))])

    # group results
    D_stat_picks = D_stat_order.groupby(['pickinglist']).sum()

    # draw histogram
    plt.figure()
    plt.hist(D_stat_picks.distance)
    plt.ylabel('N. of picking lists')
    plt.xlabel('Distance')
    plt.title(f"Distance per picking list. {np.round(len(pickups)/len(picklists)*100,1)}% of the dataset ")

    # draw traffic chart
    D_stat_arcs = D_arcs.groupby(['nodeFrom', 'nodeTo']).size().reset_index()
    D_stat_arcs = D_stat_arcs.rename(columns={0: 'traffic'})

    # set edge attributes
    edge_attributes_all = {(nodeFrom, nodeTo): {'traffic': 0.0001} for (nodeFrom, nodeTo) in G.edges}
    nx.set_edge_attributes(G, edge_attributes_all)

    edge_attributes_traffic = {(nodeFrom, nodeTo): {'traffic': traff} for (nodeFrom, nodeTo, traff) in zip(D_stat_arcs.nodeFrom, D_stat_arcs.nodeTo, D_stat_arcs.traffic)}
    nx.set_edge_attributes(G, edge_attributes_traffic)

    distance = ''
    weight = 'traffic'
    title = 'Traffic chart'
    arcLabel = False
    nodeLabel = False
    trafficGraph = True
    printNodecoords = True

    if edgePredecessors:
        dg.printGraph(G, distance, weight, title, arcLabel, nodeLabel, trafficGraph, printNodecoords, D_layout)

    return D_stat_arcs, D_stat_picks


def defineWHgraph(D_layout: pd.DataFrame, D_IO: pd.DataFrame, D_fake: pd.DataFrame, allLocs: int,
                  draw: bool = False, arcLabel: bool = False, nodeLabel: bool = False, trafficGraph: bool = False):
    """
    the function returns a graph G, and a table with the mapping from idlocations to graph nodes

    Args:
        D_layout (pd.DataFrame): pandas dataframe containing the coordinates for each locations.
        D_IO (pd.DataFrame): dataframe containing the coordinates of the Input and output locations.
        D_fake (pd.DataFrame): dataframe containing the coordinates of the fake locations.
        allLocs (int): number of the initial locations (returned by the function preprocessing the coordinates).
        draw (bool, optional): when true plot the graph. Defaults to False.
        arcLabel (bool, optional): when true plot the arc labels. Defaults to False.
        nodeLabel (bool, optional): when true plot the node labels. Defaults to False.
        trafficGraph (bool, optional): when true compute and plot the traffic graph. Defaults to False.

    Returns:
        nx.Graph: Output networkx graph.
        pd.DataFrame: Output DataFrame.
        pd.DataFrame: DESOutput DataFrameCRIPTION.

    """

    D_layout.columns = [i.lower() for i in D_layout.columns]

    fakecoordx = D_IO.loccodex.iloc[0]
    fakecoordy = D_IO.loccodey.iloc[0]

    # map the coordinates of all the fake locations with the I/O
    D_layout.loccodex.loc[D_layout.idlocation.isin(D_fake.idlocation)] = fakecoordx
    D_layout.loccodey.loc[D_layout.idlocation.isin(D_fake.idlocation)] = fakecoordy

    # estimathe coordinates of the missing aisles
    D_layout = estimateMissingAislecoordX(D_layout)

    #  plot coordinates after removing nan

    if len(D_layout) == allLocs:
        # find coordinates between graph nodes and id locations
        D_nodes, D_res, D_IO = defineGraphNodes(D_layout, D_IO)

        # define arcs
        edgeTable = defineEdgeTable(D_nodes, D_IO)

        # define the graph
        G = dg.defineGraph(edgeTable)

        # set graph attribute coordinates x and y
        pos = {idlocation: (coordx, coordy) for (idlocation, coordx, coordy) in zip(D_nodes.index.values, D_nodes.aislecodex, D_nodes.loccodey)}
        pos_io = {idlocation: (coordx, coordy) for (idlocation, coordx, coordy) in zip(D_IO.index.values, D_IO.loccodex, D_IO.loccodey)}
        pos.update(pos_io)
        nx.set_node_attributes(G, pos, 'coordinates')

        # set boolean input
        attr_io = {idlocation: inputloc for (idlocation, inputloc) in zip(D_IO.index.values, D_IO.inputloc)}
        nx.set_node_attributes(G, attr_io, 'input')

        # set boolean input
        attr_io = {idlocation: outputloc for (idlocation, outputloc) in zip(D_IO.index.values, D_IO.outputloc)}
        nx.set_node_attributes(G, attr_io, 'output')

        # set distance between the nodes and the IO point
        # consider a single input point
        idlocation_IN = D_IO[D_IO.inputloc == 1].index.values[0]
        idlocation_OUT = D_IO[D_IO.outputloc == 1].index.values[0]

        # prepare dataframe with results
        D_allNodes = list(G.nodes)
        D_distanceIO = pd.DataFrame(index=D_allNodes)
        D_distanceIO['IN_dist'] = None
        D_distanceIO['OUT_dist'] = None

        # calculate IO distance for each node of the graph
        for index, row in D_distanceIO.iterrows():
            # distance IN
            dist_IN = nx.shortest_path_length(G, source=idlocation_IN, target=index, weight='length', method='dijkstra')
            dist_OUT = nx.shortest_path_length(G, source=idlocation_OUT, target=index, weight='length', method='dijkstra')
            D_distanceIO['IN_dist'].loc[index] = dist_IN
            D_distanceIO['OUT_dist'].loc[index] = dist_OUT

        # set input distance
        attr_dist_in = {idlocation: in_dist for (idlocation, in_dist) in zip(D_distanceIO.index.values, D_distanceIO.IN_dist)}
        nx.set_node_attributes(G, attr_dist_in, 'input_distance')

        # set output distance
        attr_dist_out = {idlocation: out_dist for (idlocation, out_dist) in zip(D_distanceIO.index.values, D_distanceIO.OUT_dist)}
        nx.set_node_attributes(G, attr_dist_out, 'output_distance')

        # draw graph
        if draw:
            # print the graph
            distance = weight = 'length'
            title = 'Warehouse graph'
            printNodecoords = False
            dg.printGraph(G, distance, weight, title, arcLabel, nodeLabel, trafficGraph, printNodecoords, D_layout)

        return G, D_res, D_layout
    else:
        print("=======EXIT=======Internal error. Some locations were not mapped")
        return [], [], []


def calculateExchangeSaving(D_mov_input: pd.DataFrame, D_res: pd.DataFrame,
                            G: nx.Graph, useSameLevel: bool = False) -> pd.DataFrame:
    """
    calculates the distance saving while exchanging two physical locations (popularity-distance graph)

    Args:
        D_mov_input (pd.DataFrame): dataframe with the set of the movements.
        D_res (pd.DataFrame): dictionary with correnspondence between IDLOCATION and NODE ID.
        G (nx.Graph): graph of the warehouse.
        useSameLevel (bool, optional): if True no exchanges are allowed between locations on different wh levels. Defaults to False.

    Returns:
        D_results (pd.DataFrame): Output dataFrame with saving and exchange.

    """

    D_mov = D_mov_input
    D_mov.columns = D_mov_input.columns.str.lower()

    if useSameLevel:
        # Calculate the popularity for each location and level
        D_bubbles = D_mov.groupby(['idlocation', 'level']).size().reset_index()

    else:
        D_bubbles = D_mov.groupby(['idlocation']).size().reset_index()
        D_bubbles = pd.DataFrame(D_bubbles)

    D_bubbles = D_bubbles.set_index('idlocation')
    D_bubbles = D_bubbles.rename(columns={0: 'popularity'})
    D_bubbles['idNode'] = None
    D_bubbles['distance'] = None

    # calculate distance for each location
    inputDistance = nx.get_node_attributes(G, 'input_distance')
    outputDistance = nx.get_node_attributes(G, 'output_distance')

    for index, row in D_bubbles.iterrows():
        if index not in D_res.keys():
            pass
        else:
            idNode = D_res[index]
            D_bubbles['idNode'].loc[index] = idNode
            D_bubbles['distance'].loc[index] = inputDistance[idNode] + outputDistance[idNode]

    # plot the distance of each point to the I/O
    nodecoords = nx.get_node_attributes(G, 'coordinates')

    # remove nan
    D_bubbles = D_bubbles.dropna()

    # save coordinates
    D_bubbles['loccodex'] = [nodecoords[idNode][0] for idNode in D_bubbles['idNode']]
    D_bubbles['loccodey'] = [nodecoords[idNode][1] for idNode in D_bubbles['idNode']]

    plt.figure()
    plt.scatter(D_bubbles.loccodex, D_bubbles.loccodey, c=D_bubbles.distance)
    plt.colorbar()
    plt.title("Distance of each node from the I/O")

    if useSameLevel:
        # make optimising exchanges on the dame level (iaisles has the same number of levels)

        res_cols = ['level', 'popularity', 'idNode', 'distance', 'new_idNode', 'new_distance', 'costASIS',
                    'costTOBE', 'idlocationASIS', 'idlocationTOBE']
        D_results = pd.DataFrame(columns=res_cols)

        for level in set(D_bubbles.level):
            D_bubbles_level = D_bubbles[D_bubbles.level == level]

            # sort the dataframe by popularity
            D_bubbles_pop = D_bubbles_level.sort_values(by='popularity', ascending=False)

            # sort the dataframe by distance
            D_bubbles_dist = D_bubbles_level.sort_values(by='distance', ascending=True)

            # work on the popularity dataframe and identify which location to pick from that popularityy
            D_bubbles_pop['new_idNode'] = D_bubbles_dist['idNode'].reset_index(drop=True).values
            D_bubbles_pop['new_distance'] = D_bubbles_dist['distance'].reset_index(drop=True).values

            # drop zeros from popularity and distance
            D_bubbles_pop['new_distance'] = D_bubbles_pop['new_distance'].replace(0, 0.0001)
            D_bubbles_pop['distance'] = D_bubbles_pop['distance'].replace(0, 0.0001)

            # estimate travelling and saving
            D_bubbles_pop['costASIS'] = D_bubbles_pop['popularity'] * D_bubbles_pop['distance']
            D_bubbles_pop['costTOBE'] = D_bubbles_pop['popularity'] * D_bubbles_pop['new_distance']

            # save exchange locations
            D_bubbles_pop['idlocationASIS'] = D_bubbles_pop.index.values
            D_bubbles_pop['idlocationTOBE'] = D_bubbles_dist.index.values

            D_results = D_results.append(D_bubbles_pop.reset_index(drop=True))

    else:
        res_cols = ['popularity', 'idNode', 'distance', 'new_idNode',
                    'new_distance', 'costASIS', 'costTOBE']
        D_results = pd.DataFrame(columns=res_cols)

        # sort the dataframe by popularity
        D_bubbles_pop = D_bubbles.sort_values(by='popularity', ascending=False)

        # sort the dataframe by distance
        D_bubbles_dist = D_bubbles.sort_values(by='distance', ascending=True)

        # identify the location to wiche it is better to pick that popularity
        D_bubbles_pop['new_idNode'] = D_bubbles_dist['idNode'].reset_index(drop=True).values
        D_bubbles_pop['new_distance'] = D_bubbles_dist['distance'].reset_index(drop=True).values

        # drop zeros from popularity and distances
        D_bubbles_pop['new_distance'] = D_bubbles_pop['new_distance'].replace(0, 0.0001)
        D_bubbles_pop['distance'] = D_bubbles_pop['distance'].replace(0, 0.0001)

        # estimate travelling and saving
        D_bubbles_pop['costASIS'] = D_bubbles_pop['popularity'] * D_bubbles_pop['distance']
        D_bubbles_pop['costTOBE'] = D_bubbles_pop['popularity'] * D_bubbles_pop['new_distance']

        # save locations exchange
        D_bubbles_pop['idlocationASIS'] = D_bubbles_pop.index.values
        D_bubbles_pop['idlocationTOBE'] = D_bubbles_dist.index.values

        D_results = D_results.append(D_bubbles_pop.reset_index(drop=True))

    D_results = D_results.reset_index(drop=True)

    D_results['saving_rank'] = 1 - D_results['costTOBE'] / D_results['costASIS']

    savingTotale = 1 - sum(D_results['costTOBE']) / sum(D_results['costASIS'])

    D_results['saving_exchange'] = D_results['saving_rank'] / (sum(D_results['saving_rank'])) * savingTotale

    print("=======================================================")
    print(f"The expected saving is: {np.round(savingTotale,3)*100}%")

    D_results['loccodexTOBE'] = [nodecoords[idNode][0] for idNode in D_results['new_idNode']]
    D_results['loccodeyTOBE'] = [nodecoords[idNode][1] for idNode in D_results['new_idNode']]

    D_results.popularity = D_results.popularity.astype(float)  # cast popularity

    return D_results


def returnPopularitydistanceGraphLocations(D_results: pd.DataFrame) -> dict:
    """
    Produce the graph popularity-distance

    Args:
        D_results (pd.DataFrame): Input Pandas DataFrame.

    Returns:
        dict: Output dictionary containing figures.

    """

    figure_out = {}
    D_results['distance'] = D_results['distance'].astype(float)

    D_graph = D_results.groupby(['idNode']).agg({'popularity': ['sum'], 'distance': ['mean']}).reset_index()
    D_graph.columns = ['idNode', 'popularity', 'distance']

    # clean popularity using IQR
    D_graph, _ = cleanUsingIQR(D_graph, features=['popularity'])

    # plot asis graph
    fig1 = plt.figure()
    plt.scatter(D_graph['popularity'], D_graph['distance'])
    plt.xlabel('Popularity')
    plt.ylabel('Distance')
    plt.title("AS-IS Scenario")
    figure_out['asis'] = fig1

    # graph pop-dist optimal
    D_results['new_distance'] = D_results['new_distance'].astype(float)
    D_graph = D_results.groupby(['new_idNode']).agg({'popularity': ['sum'], 'new_distance': ['mean']}).reset_index()
    D_graph.columns = ['idNode', 'popularity', 'distance']

    # clean popularity using IQR
    D_graph, _ = cleanUsingIQR(D_graph, features=['popularity'])

    # plot tobe graph
    fig2 = plt.figure()
    plt.scatter(D_graph['popularity'], D_graph['distance'])
    plt.xlabel('Popularity')
    plt.ylabel('Distance')
    plt.title("TO-BE Scenario")
    figure_out['tobe'] = fig2

    return figure_out


def returnbubbleGraphAsIsToBe(D_results: pd.DataFrame, cleanData: bool = False) -> dict:
    """
    Return the graph with storage plant layout and picking bubbles

    Args:
        D_results (pd.DataFrame): Inputn pndas DataFrame.
        cleanData (bool, optional): If true, data are cleaned using IQR. Defaults to False.

    Returns:
        dict: DESCRIPTION.

    """

    def _normaliseVector(x):
        return(x - min(x)) / (max(x) - min(x))

    figure_out = {}

    if cleanData:
        D_results, _ = cleanUsingIQR(D_results, ['popularity'])

    # graph as/is
    D_graph = D_results.groupby(['loccodex', 'loccodey'])['popularity'].agg(['sum']).reset_index()
    D_graph['size'] = _normaliseVector(D_graph['sum']) * 100

    fig1 = plt.figure()
    plt.scatter(D_graph.loccodex, D_graph.loccodey, D_graph['size'])
    plt.title("Warehouse as-is")
    figure_out['pick_layout_asis'] = fig1

    # graph to/be
    D_graph = D_results.groupby(['loccodexTOBE', 'loccodeyTOBE'])['popularity'].agg(['sum']).reset_index()
    D_graph['size'] = _normaliseVector(D_graph['sum']) * 100

    fig2 = plt.figure()
    plt.scatter(D_graph.loccodexTOBE, D_graph.loccodeyTOBE, D_graph['size'])
    plt.title("Warehouse to-be")
    figure_out['pick_layout_tobe'] = fig2

    return figure_out


def plotLocations(D_locations: pd.DataFrame) -> dict:
    """
    Produce a plot with the poistion of the storage locations
    Args:
        D_locations (pd.DataFrame): pandas dataframe with the coordinates of the locations.

    Returns:
        dict: dictionary of figures.

    """

    output_figures = {}

    # organise locations by type
    D_input_locations = D_locations[D_locations['INPUTLOC'].isin([True])]
    D_output_locations = D_locations[D_locations['OUTPUTLOC'].isin([True])]
    D_physical_locations = D_locations[D_locations['FAKELOC'].isin([False])]

    # plot locations
    fig1 = plt.figure()
    plt.scatter(D_physical_locations['LOCCODEX'], D_physical_locations['LOCCODEY'])
    plt.scatter(D_input_locations['LOCCODEX'], D_input_locations['LOCCODEY'])
    plt.scatter(D_output_locations['LOCCODEX'], D_output_locations['LOCCODEY'])
    plt.legend(["Locations", "Input", "Output"])

    output_figures['layout'] = fig1
    return output_figures


def extractIoPoints(D_loc: pd.DataFrame):
    """
    Find the input and output coordinates from the locations dataframe

    Args:
        D_loc (pd.DataFrame): Input dataFrame.

    Returns:
        input_loccodex (float): Input X coordinate.
        input_loccodey (float): Input Y coordinate.
        output_loccodex (float): Output X coordinate.
        output_loccodey (float): Output Y coordinate.

    """

    D_loc_IN = D_loc[D_loc['INPUTLOC'].isin([True])]
    D_loc_OUT = D_loc[D_loc['OUTPUTLOC'].isin([True])]
    D_loc = D_loc[(D_loc['INPUTLOC'].isin([False])) & (D_loc['OUTPUTLOC'].isin([False]))]

    # set Input point
    if len(D_loc_IN) == 0:
        input_loccodey = np.nanmin(D_loc['LOCCODEY']) - 1
        input_loccodex = np.nanmean(list(set(D_loc['LOCCODEX'])))
    else:
        input_loccodey = D_loc_IN.iloc[0]['LOCCODEY']
        input_loccodex = D_loc_IN.iloc[0]['LOCCODEX']

    # set output point
    if len(D_loc_OUT) == 0:
        output_loccodey = np.nanmin(D_loc['LOCCODEY']) - 1
        output_loccodex = np.nanmean(list(set(D_loc['LOCCODEX'])))
    else:
        output_loccodey = D_loc_OUT.iloc[0]['LOCCODEY']
        output_loccodex = D_loc_OUT.iloc[0]['LOCCODEX']
    return input_loccodex, input_loccodey, output_loccodex, output_loccodey


def calculateStorageLocationsDistance(D_loc: pd.DataFrame, input_loccodex: float,
                                      input_loccodey: float, output_loccodex: float,
                                      output_loccodey: float) -> pd.DataFrame:
    """
    calculate the sum of the rectangular distances from
    Input point -> physical location -> Output point

    Args:
        D_loc (pd.DataFrame): Input location DataFrame.
        input_loccodex (float): Input X coordinate.
        input_loccodey (float): Input Y coordinate.
        output_loccodex (float): Output X coordinate.
        output_loccodey (float): Output Y coordinate.

    Returns:
        D_loc (TYPE): DESCRIPTION.

    """

    D_loc = D_loc.dropna(subset=['LOCCODEX', 'LOCCODEY'])
    D_loc['INPUT_DISTANCE'] = np.abs(input_loccodex - D_loc['LOCCODEX']) + np.abs(input_loccodey - D_loc['LOCCODEY'])
    D_loc['OUTPUT_DISTANCE'] = np.abs(output_loccodex - D_loc['LOCCODEX']) + np.abs(output_loccodey - D_loc['LOCCODEY'])
    return D_loc


def prepareCoordinates(D_layout: pd.DataFrame, D_IO: pd.DataFrame = [], D_fake: pd.DataFrame = []):
    """

    Args:
        D_layout (pd.DataFrame): Input layout dataframe.
        D_IO (pd.DataFrame, optional): Input I/O DataFrame. Defaults to [].
        D_fake (pd.DataFrame, optional): Input fake locations DataFrame. Defaults to [].

    Returns:
        pd.DataFrame: D_layout with update coordinates.
        pd.DataFrame: D_IO with update coordinates.
        pd.DataFrame: D_fake with update coordinates.
        TYPE: DESCRIPTION.

    """

    D_layout.columns = D_layout.columns.str.lower()
    D_check = D_layout[['loccodex', 'loccodey']]

    allLocs = len(D_layout)

    # if at least one coordinate is given
    if len(D_check.drop_duplicates()) > 2:

        # import I/O points
        if len(D_IO) == 0:
            D_IO = pd.DataFrame(columns=['idlocation', 'inputloc', 'outputloc',
                                         'loccodex', 'loccodey', 'loccodez'])
        D_IO.columns = D_IO.columns.str.lower()
        D_IO = D_IO.dropna()

        # if a input point is not mapped it is placed in the middle of the front
        if len(D_IO[D_IO.inputloc == 1]) == 0:
            idlocation = -1
            loccodey = np.nanmin(D_layout.loccodey) - 1
            loccodex = np.nanmean(list(set(D_layout.loccodex)))
            loccodez = 0
            inputloc = 1
            outputloc = 0
            D_IO = D_IO.append(pd.DataFrame([[idlocation, inputloc, outputloc, loccodex,
                                              loccodey, loccodez]], columns=D_IO.columns))
            print(f"=======Input point unmapped. I is set to x:{loccodex},y:{loccodey}")

        # if the Output point is not mapped, it is placed in the middle of the front
        if len(D_IO[D_IO.outputloc == 1]) == 0:
            idlocation = -2
            loccodey = np.nanmin(D_layout.loccodey) - 1
            loccodex = np.nanmean(list(set(D_layout.loccodex)))
            loccodez = 0
            inputloc = 0
            outputloc = 1
            D_IO = D_IO.append(pd.DataFrame([[idlocation, inputloc, outputloc, loccodex,
                                              loccodey, loccodez]], columns=D_IO.columns))
            print(f"=======Output point unmapped. O is set to x:{loccodex},y:{loccodey}")

        # identify fake locations
        if len(D_fake) == 0:
            D_fake = pd.DataFrame(columns=['idlocation', 'inputloc', 'outputloc',
                                           'loccodex', 'loccodey', 'loccodez'])
        D_fake.columns = D_fake.columns.str.lower()

        return D_layout, D_IO, D_fake, allLocs

    else:
        print("======EXIT===== No coordinates mapped to define a graph")
        return [], [], [], []


def asisTobeBubblePopDist(D_results: pd.DataFrame, cleanData: bool = False) -> dict:
    """
    Plot ASIS - TOBE graph

    Args:
        D_results (pd.DataFrame): Input pandas DataFrame.
        cleanData (bool, optional): If true use IQR to clean data. Defaults to False.

    Returns:
        dict: Output dictionary with figures.

    """

    output_figures = {}
    if cleanData:
        D_results, _ = cleanUsingIQR(D_results, ['popularity'])

    D_results['distance'] = D_results['distance'].astype(float)

    # ASIS GRAPH
    D_graph = D_results.groupby(['idNode']).agg({'popularity': ['sum'],
                                                 'distance': ['mean']}).reset_index()
    D_graph.columns = ['idNode', 'popularity', 'distance']

    fig1 = plt.figure()
    plt.scatter(D_graph['distance'], D_graph['popularity'])
    plt.xlabel('Distance (m)')
    plt.ylabel('Popularity')
    plt.title("AS-IS configuration")
    output_figures['pop_dist_asis'] = fig1

    # TOBE GRAPH
    D_results['new_distance'] = D_results['new_distance'].astype(float)
    D_graph = D_results.groupby(['new_idNode']).agg({'popularity': ['sum'],
                                                     'new_distance': ['mean']}).reset_index()
    D_graph.columns = ['idNode', 'popularity', 'distance']

    fig2 = plt.figure()
    plt.scatter(D_graph['distance'], D_graph['popularity'])
    plt.xlabel('Distance (m)')
    plt.ylabel('Popularity')
    plt.title("TO-BE configuration")
    output_figures['pop_dist_tobe'] = fig2
    return output_figures
