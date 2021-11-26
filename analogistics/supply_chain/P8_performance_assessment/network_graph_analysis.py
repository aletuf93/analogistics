# -*- coding: utf-8 -*-
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import ast

from analogistics.statistics import time_series as ts
from analogistics.clean import cleanUsingIQR
from analogistics.supply_chain.P8_performanceAssessment.utilities_movements import getCoverageStats


def _returnDummyColumnsFromList(D: pd.DataFrame, columnName: str):
    '''
    extract the value from a pandas cell where a list is saved

    Parameters
    ----------
    D : TYPE pandas dataframe
        DESCRIPTION.
    columnName : TYPE
        DESCRIPTION.

    Returns
    -------
    s : TYPE column string name
        DESCRIPTION.

    '''
    x = D[columnName].values
    # print(x)
    # print(type(x))
    newList = []
    for j in x:
        for i in j:
            i = str(i)
            yyy = i.replace('nan,', '')
            yyy = yyy.replace('nan', '')
            newList.append(ast.literal_eval(yyy))

    s = pd.Series(newList)
    # X_add=pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    # print(s)
    return s


def networkRaysPlot(D_mov: pd.DataFrame, timecol: str, lonCol_to: str, latCol_to: str,
                    lonCol_from: str, latCol_from: str,
                    G: nx.Graph, capacityField: str = 'QUANTITY', sampleInterval: str = 'week',):
    '''
    the function plot rays on a graph G to represent the flow of a supply chain network

    D_mov is the input dataframe
    timecol is the column string with timestamp information
    lonCol_to is the column with the longitude of the origin points
    latCol_to is the column with the latitude of the origin points
    lonCol_from is the column with the longitude of the origin points
    latCol_from is the column with the latitude of the origin points
    G is a network graph to plot in background
    capacityField is a field of capacity to calculate coverages
    sampleInterval is the sampling interval od D_mov: day, week, month, year
    '''

    output_figure = {}
    output_coverages = {}

    D_mov['TIME_PERIOD'] = ts.sampleTimeSeries(D_mov[timecol], sampleInterval)

    # get coverages
    analysisFieldList = [lonCol_to, latCol_to, lonCol_from, latCol_from]
    outputCoverages, _ = getCoverageStats(D_mov, analysisFieldList, capacityField=capacityField)
    outputCoverages = pd.DataFrame(outputCoverages)
    output_coverages[f"Network_rays_{sampleInterval}"] = outputCoverages

    # group movements
    D_groupFlows = D_mov.groupby(['TIME_PERIOD', lonCol_to, latCol_to, lonCol_from, latCol_from]).size().reset_index()
    D_groupFlows['FLOW_NORM'] = (0.0001 + D_groupFlows[0] - min(D_groupFlows[0])) / (max(D_groupFlows[0]) - min(D_groupFlows[0])) * 3

    # plot figures
    D_groupFlows = D_groupFlows.sort_values(by='TIME_PERIOD')
    for period in D_groupFlows['TIME_PERIOD'].drop_duplicates():
        D_groupFlows_filtered = D_groupFlows[D_groupFlows['TIME_PERIOD'] == period]

        # if a road network graph is given, then plot it
        if G == []:
            fig1 = plt.figure()
            ax = fig1.gca()
        else:
            fig1, ax = ox.plot_graph(G, bgcolor='k',
                                     node_size=1, node_color='#999999',
                                     node_edgecolor='none', node_zorder=2,
                                     edge_color='#555555', edge_linewidth=0.5,
                                     edge_alpha=1, show=False)
            plt.legend(['Node', 'Edges'])
    
        # represent the rays
        for i in range(0,len(D_groupFlows_filtered)):
            x_from=D_groupFlows_filtered[lonCol_from].iloc[i]
            y_from=D_groupFlows_filtered[latCol_from].iloc[i]
            x_to=D_groupFlows_filtered[lonCol_to].iloc[i]
            y_to=D_groupFlows_filtered[latCol_to].iloc[i]
            c=D_groupFlows_filtered['FLOW_NORM'].iloc[i]
            
            
            ax.plot([x_from,x_to],[y_from,y_to], color='orange', linewidth=c)
            ax.set_title(f"Network rays {period}")
            
            fig1 = ax.figure
            fig1.show()
            output_figure[f"Network_rays_{sampleInterval}_{period}"]=fig1
            plt.close('all')
    return output_figure, output_coverages





# %% GRAPH HIGLIGHTING THE TRAVELLED ARCS

def networkRoadsRoutePlot(D_arcs,lonCol_from, latCol_from, latCol_to,lonCol_to, G):
    #D_arcs is the input dataframe containing one row for each arc travelled, in the format
    #of a dataframe returned by vessel statistics
    
    # lonCol_to is the column with the longitude of the origin points
    # latCol_to is the column with the latitude of the origin points
    # lonCol_from is the column with the longitude of the origin points
    # latCol_from is the column with the latitude of the origin points
    # G is a network graph to plot in 
    
    output_figures = {}
    
    D_arcs_filtered=D_arcs[[lonCol_from,latCol_from,latCol_to,lonCol_to]]
    
    
    D_arcs_filtered[lonCol_from] = [k for k in returnDummyColumnsFromList(D_arcs_filtered, lonCol_from)]
    D_arcs_filtered[latCol_from] = [k for k in returnDummyColumnsFromList(D_arcs_filtered, latCol_from)]
    D_arcs_filtered[latCol_to] = [ k for k in returnDummyColumnsFromList(D_arcs_filtered, latCol_to)]
    D_arcs_filtered[lonCol_to] = [ k for k in returnDummyColumnsFromList(D_arcs_filtered, lonCol_to)]
    
    
    all_variables = [lonCol_from, latCol_from, latCol_to, lonCol_to]
    for i in range(0,len(D_arcs_filtered)):
        for variable in all_variables:
            if isinstance(D_arcs_filtered[variable].iloc[i],float):
                D_arcs_filtered[variable].iloc[i] =  D_arcs_filtered[variable].iloc[i]
                
            elif isinstance(D_arcs_filtered[variable].iloc[i],list) & len(D_arcs_filtered[variable].iloc[i])>0:
                D_arcs_filtered[variable].iloc[i] =  D_arcs_filtered[variable].iloc[i][0]
            else:
                D_arcs_filtered[variable].iloc[i]=np.nan
    
    D_arcs_filtered=D_arcs_filtered.dropna()
    D_arcs_filtered_grouped = D_arcs_filtered.groupby([lonCol_from, latCol_from, latCol_to, lonCol_to]).size().reset_index()
    
    routes=[]
    for i in range(0,len(D_arcs_filtered_grouped)):
        lat_from = D_arcs_filtered_grouped[latCol_from].iloc[i]
        lon_from = D_arcs_filtered_grouped[lonCol_from].iloc[i]
        lat_to = D_arcs_filtered_grouped[latCol_to].iloc[i]
        lon_to = D_arcs_filtered_grouped[lonCol_to].iloc[i]
    
        #find closest node
        node_from = ox.get_nearest_node(G, (lat_from,lon_from), method='euclidean')
        node_to = ox.get_nearest_node(G, (lat_to,lon_to), method='euclidean')
        
        # calculate shortest paths for the 2 routes
        route = nx.shortest_path(G, node_from, node_to, weight='length')
        routes.append(route)
    fig, ax = ox.plot_graph_routes(G, routes, route_color='orange', route_linewidth=3,
                                   node_size=0,
                                   orig_dest_node_size =0)
    
    output_figures['Network arcs']=fig
    return output_figures
    

# %% REPRESENT THE NODES OF A SUPPLY CHAIN ON THE MAP
def supplyChainMap(D_table,latCol,lonCol,descrCol,color,size,cleanOutliers=False):
    '''
    D_table is an input dataframe
    latCol is the string of the column with latitudes
    lonCol is the string of the column with longitudes
    descrCol is the string of the column with the description of a node
    size is the string of the column with the size of a node
    color is the string of the column with different values for different colors
    cleanOutliers is True to clean outliers using IQR
    '''
    output_coverages={}
    
    analysisFieldList=[latCol, lonCol]
    outputCoverages, _ = getCoverageStats(D_table,analysisFieldList,capacityField=size)
    if cleanOutliers:
        D_table, coverages, =cleanUsingIQR(D_table, [latCol,lonCol])
        outputCoverages = (coverages[0]*outputCoverages[0],coverages[1]*outputCoverages[1])
    output_coverages['coverages'] = pd.DataFrame(outputCoverages)
    fig = px.scatter_mapbox(D_table, lat=latCol, lon=lonCol,
                     hover_name=descrCol, size=size,color=color)
    fig.update_layout(mapbox_style="open-street-map")
    
    return fig, output_coverages

# %% REPRESENT THE CENTER OF MASS OF THE SUPPLY CHAIN

def supplyChainCentersOfMass(D_plant,D_mov,latCol_plant,lonCol_plant,node_from,node_to,lonCol_to,latCol_to,capacityField='QUANTITY',cleanOutliers=False):
    '''
    
    represent the plant of the network and the center of mass considering
    latitude and longitude of the other nodes of the network. The assignment
    of nodes and plants is based on the movement dataframe D_mov
    
    D_plant is a dataframe with plant information
    D_move is a dataframe with movements information
    node_from is the string of the columns with the node code from in D_mov
    node_to is the string of the columns with the node code from in D_mov
    lonCol_to is the string of the columns with the discharging node longitude in D_mov
    latCol_to is the string of the columns with the discharging node latitude in D_mov
    '''
    output_coverages={}
    
    #get coverages and clean outliers
    analysisFieldList=[node_from, node_to,lonCol_to, latCol_to]
    outputCoverages, _ = getCoverageStats(D_mov,analysisFieldList,capacityField=capacityField)
    D_mov_filtered=D_mov.dropna(subset=analysisFieldList)
    
    if cleanOutliers:
        D_mov_filtered, coverages, =cleanUsingIQR(D_mov_filtered, [lonCol_to,latCol_to])
        outputCoverages = (coverages[0]*outputCoverages[0],coverages[1]*outputCoverages[1])
    output_coverages['coverages']=pd.DataFrame(outputCoverages)
    
    #calculate the center of mass
    D_plant['lon_ave'] = D_plant['lat_ave']=np.nan
    for index, row in D_plant.iterrows():
        
        D_table_plant = D_mov_filtered[D_mov_filtered[node_from]==row['_id']]
        lon_ave = np.nanmean(D_table_plant[lonCol_to])
        lat_ave = np.nanmean(D_table_plant[latCol_to])
        D_plant['lon_ave'].loc[index] = lon_ave
        D_plant['lat_ave'].loc[index] = lat_ave
        
   
    fig = go.Figure(go.Scattermapbox(
            lat=D_plant[latCol_plant],
            lon=D_plant[lonCol_plant],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=14,
                color = 'blue'
            ),
            text='ASIS',
            name='Actual Plants'
        ))
    
    fig.add_trace(
            go.Scattermapbox(
            lat=D_plant['lat_ave'],
            lon=D_plant['lon_ave'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=14,
                color = 'red',
                #symbol = 'cross'
            ),
            text="CENTER OF MASS",
            name='Centers of mass'
            )
        )
    #represent the covering for each plant
    D_covering = D_mov_filtered.groupby([node_from, node_to,latCol_to,lonCol_to]).size().reset_index() 
    D_covering[node_from]= D_covering[node_from].astype(str)
    fig_nodes=supplyChainMap(D_table=D_covering, 
                         latCol=latCol_to, 
                         lonCol=lonCol_to,
                         descrCol=node_to, 
                         size=0,
                         color=node_from,
                         cleanOutliers=False)[0]
    for i in range(0,len(fig_nodes.data)):
        fig.add_trace(fig_nodes.data[i])
    fig.update_layout(mapbox_style="open-street-map")
    return fig, output_coverages


# %% REPRESENT THE COVERING OF A SUPPLY CHAIN

def supplyChainCovering(D_plant,D_mov,latCol_plant,lonCol_plant,node_from,node_to,lonCol_to,latCol_to,capacityField='QUANTITY',cleanOutliers=False):
    '''
    
    represent the assignment of nodes an plant with different colors
    
    D_plant is a dataframe with plant information
    D_move is a dataframe with movements information
    node_from is the string of the columns with the node code from in D_mov
    node_to is the string of the columns with the node code from in D_mov
    lonCol_to is the string of the columns with the discharging node longitude in D_mov
    latCol_to is the string of the columns with the discharging node latitude in D_mov
    '''
    output_coverages={}
    
    #get coverages and clean outliers
    analysisFieldList=[node_from, node_to,lonCol_to, latCol_to]
    outputCoverages, _ = getCoverageStats(D_mov,analysisFieldList,capacityField=capacityField)
    D_mov_filtered=D_mov.dropna(subset=analysisFieldList)
    
    if cleanOutliers:
        D_mov_filtered, coverages, =cleanUsingIQR(D_mov_filtered, [lonCol_to,latCol_to])
        outputCoverages = (coverages[0]*outputCoverages[0],coverages[1]*outputCoverages[1])
    output_coverages['coverages']=pd.DataFrame(outputCoverages)
    
   
    #represent the covering for each plant
    D_covering = D_mov_filtered.groupby([node_from, node_to,latCol_to,lonCol_to]).size().reset_index() 
    D_covering[node_from]= D_covering[node_from].astype(str)
    fig_covering=supplyChainMap(D_table=D_covering, 
                         latCol=latCol_to, 
                         lonCol=lonCol_to,
                         descrCol=node_to, 
                         size=0,
                         color=node_from,
                         cleanOutliers=False)[0]
    
    fig_covering.update_layout(mapbox_style="open-street-map")
    return fig_covering, output_coverages


# %% calculate route distance
def networkRouteDistance(D_arcs,lonCol, latCol, G):
    '''
    

    Parameters
    ----------
    D_arcs : TYPE pandas dataframe
        DESCRIPTION. pandas dataframe listing all the nodes visited by a single tour
    lonCol : TYPE string
        DESCRIPTION. column name of the longityde
    latCol : TYPE string
        DESCRIPTION. column name of the latitude
    G : TYPE networkx graph
        DESCRIPTION. road graph

    Returns
    -------
    output_figures : TYPE matplotlib figure
        DESCRIPTION. figure indicating the travelled arcs on the graph
    distance_array : TYPE list
        DESCRIPTION. array of the partial distances travelled between the nodes

    '''
    
    output_figures=[]
    routes=[]
    distance_array=[0]
    
    
    for i in range(1,len(D_arcs)):
        lat_from = D_arcs[latCol].iloc[i-1]
        lon_from = D_arcs[lonCol].iloc[i-1]
        lat_to = D_arcs[latCol].iloc[i]
        lon_to = D_arcs[lonCol].iloc[i]
    
        #find closest node
        #print(lat_from)
        #print(lon_from)
        #print(lat_to)
        #print(lon_to)
        
        node_from = ox.get_nearest_node(G, (lat_from,lon_from), method='euclidean')
        node_to = ox.get_nearest_node(G, (lat_to,lon_to), method='euclidean')
        
        # calculate shortest paths for the 2 routes
        try:
            route = nx.shortest_path(G, node_from, node_to, weight='length')
            routes.append(route)
        except:
            pass
            
        try:
            distance = nx.shortest_path_length(G, node_from, node_to, weight='length')
            distance_array.append(distance)
        except:
            distance_array.append(np.nan)
        
    fig, ax = ox.plot_graph_routes(G, routes, route_color='orange', route_linewidth=3,
                                   node_size=0,
                                   orig_dest_node_size =0)
    
    output_figures=fig
    return output_figures, distance_array