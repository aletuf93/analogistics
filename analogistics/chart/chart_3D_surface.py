import plotly.graph_objects as go
import pandas as pd
import numpy as np
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


def createsurfaceFromGrid(X: np.array, Y: np.array, grid: np.array) -> go.Figure:
    """
    Create a plotly 3d surface

    Args:
        X (np.array): array with the X coordinates (n*n).
        Y (np.array): array with the Y coordinate (n*n).
        grid (np.array): array with the Z values of the function (n*n).

    Returns:
        fig (go.Figure): static surface output.

    """

    data = []
    # plot the surface
    data.append(go.Surface(z=grid,
                           x=X,
                           y=Y)
                )

    # create the figure
    fig = go.Figure(data=data)
    return fig


def createFigureWith3Dsurface(X_list: list, Y_list: list, grid_list: list, time_list: list) -> go.Figure:
    """
    return an animated 3d surface with a frame for each period of the time list

    Args:
        X_list (list): array of X coordinates.
        Y_list (list): array of Y coordinates.
        grid_list (list): list of grids(with Z unction values).
        time_list (list): array of time elements.

    Returns:
        go.Figure: Output Interactive figure.

    """

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

    # define layout hovermode
    fig_dict["layout"]["hovermode"] = "closest"

    # define sliders
    fig_dict["layout"]["sliders"] = {"args": ["transition", {"duration": 400,
                                                             "easing": "cubic-in-out"
                                                             }
                                              ],
                                     "initialValue": "1952",
                                     "plotlycommand": "animate",
                                     "values": time_list,
                                     "visible": True
                                     }

    # define menus and buttons
    fig_dict["layout"]["updatemenus"] = [{"buttons": [{"args": [None, {"frame": {"duration": 500, "redraw": True},
                                                                       "fromcurrent": True, "transition": {"duration": 300,
                                                                                                           "easing": "quadratic-in-out"}}],
                                                       "label": "Play",
                                                       "method": "animate"
                                                       },
                                                      {"args": [[None], {"frame": {"duration": 0, "redraw": True},
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
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": []
                    }

    # #######################################
    # ######## DEFINE FIGURE ################
    # #######################################

    # plot the surface
    data_dict = go.Surface(z=grid_list[0],
                           x=X_list[0],
                           y=Y_list[0])

    fig_dict["data"].append(data_dict)

    # #######################################
    # ######## DEFINE FRAMES ################
    # #######################################

    for j in range(0, len((time_list))):
        frame = {"data": [], "name": time_list[j]}

        # define the trace with bubbles of the other flows
        data_dict = go.Surface(z=grid_list[j],
                               x=X_list[j],
                               y=Y_list[j])
        frame["data"].append(data_dict)
        fig_dict["frames"].append(frame)

        # update the slider
        slider_step = {"args": [[time_list[j]],
                                {"frame": {"duration": 300, "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 300}}
                                ],
                       "label": time_list[j],
                       "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    # update the layout
    fig_dict["layout"]["sliders"] = [sliders_dict]
    # create the figure
    fig = go.Figure(fig_dict)

    return fig
