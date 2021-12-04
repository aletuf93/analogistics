# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def createFigureBubbleFlows(D_res: pd.DataFrame, latCol: str, lonCol: str,
                            weightCol: str, descrCol: str) -> go.Figure:
    """

    Args:
        D_res (pd.DataFrame): Input dataframe with flows defined by the function calculateOptimalLocation.
        latCol (str): string with column name for latitude.
        lonCol (str): string with column name for longitude.
        weightCol (str): string with column name for flow intensity.
        descrCol (str): string with column name for node description.

    Returns:
        fig_geo (go.Figure): Ouptut Figure.

    """
    fig_geo = px.scatter_mapbox(D_res,
                                lat=latCol,
                                lon=lonCol,
                                color="COST",
                                hover_name=descrCol,
                                size=weightCol,
                                animation_frame="YEAR",
                                #  projection="natural earth")
                                )
    fig_geo.update_layout(mapbox_style="open-street-map")
    return fig_geo


def createFigureOptimalPoints(D_res_optimal: pd.DataFrame, latCol: str, lonCol: str, descrCol: str) -> go.Figure:
    """

    Args:
        D_res_optimal (pd.DataFrame): dataframe with flows defined by the function calculateOptimalLocation.
        latCol (str): string with column name for latitude.
        lonCol (str): string with column name for longitude.
        descrCol (str): string with column name for node description.

    Returns:
        fig_optimal (go.Figure): Output interactive Figure.

    """

    # define the optimal location
    fig_optimal = px.line_mapbox(D_res_optimal,
                                 lat=latCol,
                                 lon=lonCol,
                                 # animation_frame="YEAR",
                                 hover_name=descrCol,
                                 # mode='lines'
                                 # color='COLOR',
                                 # color_continuous_scale='Inferno',
                                 )
    fig_optimal.update_layout(mapbox_style="open-street-map")
    return fig_optimal


def createFigureWithOptimalPointsAndBubbleFlows(D_res: pd.DataFrame, D_res_optimal: pd.DataFrame,
                                                latCol: str, lonCol: str, descrCol: str = 'PERIOD') -> go.Figure:
    """
    generates the map with the optimal location of a network.
    Optimal locations for each period are in D_res_optimal. Period are prepresented
    as animation frames of the figure

    Args:
        D_res (pd.DataFrame): dataframe with flows defined by the function calculateOptimalLocation.
        D_res_optimal (pd.DataFrame): dataframe with optimal locations defined by the function calculateOptimalLocation.
        latCol (str): string with column name for latitude.
        lonCol (str): string with column name for longitude.
        descrCol (str, optional): string with column name for node description. Defaults to 'PERIOD'.

    Returns:
        fig (go.Figure): Output interactive figure.

    """

    # assign color for the optimal points
    D_res_optimal['COLOR'] = pd.factorize(D_res_optimal['YEAR'])[0]

    # #######################################
    # ######## DEFINE RAW FIGURE ############
    # #######################################
    # define raw figure
    fig_dict = {"data": [],
                "layout": {},
                "frames": []
                }

    # #######################################
    # ######## DEFINE BASE LAYOUT ###########
    # #######################################

    # Identify all the years
    years = list(set(D_res_optimal["YEAR"]))
    years.sort()

    # define layout hovermode
    fig_dict["layout"]["hovermode"] = "closest"

    # define sliders
    fig_dict["layout"]["sliders"] = {"args": ["transition", {"duration": 400,
                                                             "easing": "cubic-in-out"
                                                             }
                                              ],
                                     "initialValue": "1952",
                                     "plotlycommand": "animate",
                                     "values": years,
                                     "visible": True
                                     }

    # define menus and buttons
    fig_dict["layout"]["updatemenus"] = [{"buttons": [{"args": [None, {"frame": {"duration": 500,
                                                                                 "redraw": True},
                                                                       "fromcurrent": True,
                                                                       "transition": {"duration": 300,
                                                                                      "easing": "quadratic-in-out"}
                                                                       }],
                                                       "label": "Play",
                                                       "method": "animate"
                                                       },
                                                      {"args": [[None], {"frame": {"duration": 0,
                                                                                   "redraw": True},
                                                                         "mode": "immediate",
                                                                         "transition": {"duration": 0}}],
                                                       "label": "Pause",
                                                       "method": "animate"
                                                       }
                                                      ],
                                          "direction": "left",
                                          "pad": {"r": 10, "t": 87},
                                          "showactive": False,
                                          "type": "buttons",
                                          "x": 0.1,
                                          "xanchor": "right",
                                          "y": 0,
                                          "yanchor": "top"
                                          }
                                         ]

    # define slider dictionary
    sliders_dict = {"active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {"font": {"size": 20},
                                     "prefix": "Year:",
                                     "visible": True,
                                     "xanchor": "right"
                                     },
                    "transition": {"duration": 300,
                                   "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": []
                    }

    # #######################################
    # ######## DEFINE DATA ##################
    # #######################################

    # start from the first frame and define the figure
    year = years[0]

    # #######################################
    # ######## DEFINE FIGURE ################
    # #######################################

    # define the trace with optimal point
    currentColor = 0

    data_dict = go.Scattermapbox(lat=D_res_optimal[D_res_optimal['YEAR'] == year][latCol],
                                 lon=D_res_optimal[D_res_optimal['YEAR'] == year][lonCol],
                                 mode='markers',
                                 marker=go.scattermapbox.Marker(size=14,
                                                                # color='red',
                                                                # color=currentColor,
                                                                # color=cm.Reds(colore),
                                                                color=[i for i in range(0, len(D_res_optimal['YEAR'] == year))],
                                                                opacity=1,
                                                                colorscale="Reds"
                                                                ),
                                 text=D_res_optimal[D_res_optimal['YEAR'] == year]['PERIOD'],
                                 name='optimal'
                                 )

    fig_dict["data"].append(data_dict)

    # define the trace with bubbles of the other flows
    data_dict = go.Scattermapbox(lat=D_res[D_res['YEAR'] == year][latCol],
                                 lon=D_res[D_res['YEAR'] == year][lonCol],
                                 # color=,
                                 text=D_res[D_res['YEAR'] == year][descrCol],
                                 marker=go.scattermapbox.Marker(size=D_res[D_res['YEAR'] == year]['FLOW_norm'],
                                                                color=D_res[D_res['YEAR'] == year]['COST_TOBE'],
                                                                opacity=0.5,
                                                                showscale=True,
                                                                colorscale='Viridis',
                                                                ),
                                 name='flow intensity'
                                 )
    fig_dict["data"].append(data_dict)

    # #######################################
    # ######## DEFINE FRAMES ################
    # #######################################

    for year in years:
        frame = {"data": [], "name": year}

        # count the current color to have a gradient in the optimal point
        currentColor = currentColor + 1

        # define the trace with optimal point
        data_dict = go.Scattermapbox(lat=D_res_optimal[D_res_optimal['YEAR'] <= year][latCol],
                                     lon=D_res_optimal[D_res_optimal['YEAR'] <= year][lonCol],
                                     mode='markers',
                                     marker=go.scattermapbox.Marker(size=14,
                                                                    # color='red',
                                                                    # color=currentColor,
                                                                    # color=cm.Reds(colore),
                                                                    color=D_res_optimal[D_res_optimal['YEAR'] <= year]['COLOR'],
                                                                    # color=[i for i in range(0,len(D_res_optimal['YEAR']==year))],
                                                                    opacity=1,
                                                                    colorscale="Reds"
                                                                    ),
                                     text=D_res_optimal[D_res_optimal['YEAR'] <= year]['PERIOD'],
                                     name='optimal'
                                     )
        frame["data"].append(data_dict)

        # define the trace with bubbles of the other flows
        data_dict = go.Scattermapbox(lat=D_res[D_res['YEAR'] == year][latCol],
                                     lon=D_res[D_res['YEAR'] == year][lonCol],
                                     # color=,
                                     text=D_res[D_res['YEAR'] == year][descrCol],
                                     marker=go.scattermapbox.Marker(size=D_res[D_res['YEAR'] == year]['FLOW_norm'],
                                                                    color=D_res[D_res['YEAR'] == year]['COST_TOBE'],
                                                                    opacity=0.5,
                                                                    showscale=True,
                                                                    colorscale='Viridis',
                                                                    ),
                                     name='flow intensity'
                                     )
        frame["data"].append(data_dict)
        fig_dict["frames"].append(frame)

        # update the slider
        slider_step = {"args": [[year],
                                {"frame": {"duration": 300,
                                           "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 300}}
                                ],
                       "label": year,
                       "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    # update the layout
    fig_dict["layout"]["sliders"] = [sliders_dict]
    # create the figure
    fig = go.Figure(fig_dict)

    # update with openStreetMap style
    fig.update_layout(mapbox_style="open-street-map")
    return fig
