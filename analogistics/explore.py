

import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns


def paretoDataframe(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Prepare a Dataframe with the cumulative and the percentual incidence of a column
    of a DataFrame to build the Pareto Chart

    Args:
        df (pd.DataFrame): pandas dataframe with unsorted values.
        field (str): column name to build the pareto.

    Returns:
        df (pd.DataFrame): pandas dataframe with cumulative and percentage columns.

    """

    df = df.dropna(subset=[field])
    df = df.sort_values(by=[field], ascending=False)
    df[f"{field}_PERC"] = df[field] / sum(df[field])
    df[f"{field}_CUM"] = df[f"{field}_PERC"].cumsum()
    return df


def paretoChart(df: pd.DataFrame, barVariable: str,
                paretoVariable: str, titolo: str) -> plt.Figure:
    """
    Plot the Pareto chart of a given variable of a DataFrame

    Args:
        df (pd.DataFrame): input pandas dataframe.
        barVariable (str): count variable of the dataframe.
        paretoVariable (str): numerical variable of the dataframe.
        titolo (str): title of the Figure.

    Returns:
        fig (plt.Figure): output figure.

    """

    df = df.sort_values(by=paretoVariable, ascending=False)
    df["cumpercentage"] = df[paretoVariable].cumsum() / df[paretoVariable].sum() * 100

    fig, ax = plt.subplots(figsize=(20, 10))

    # plot on principal axis
    ax.bar(np.linspace(0, 100, num=len(df)), df[paretoVariable], color="C0", width=0.5)
    # ax.bar(df[barVariable], df[paretoVariable], color="C0")
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis="y", colors="C0")

    # plot on secondary axis
    ax2 = ax.twinx()
    ax2.plot(np.linspace(0, 100, num=len(df)), df["cumpercentage"], color="C1", marker="D", ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.tick_params(axis="y", colors="C1")
    plt.title(titolo)
    plt.xlabel(str(barVariable))
    plt.ylabel('Percentage ' + str(paretoVariable))
    plt.ylim([0, 110])
    return fig


def scatterplotMatrix(X: pd.DataFrame, dirResults: str) -> bool:
    """
    Build a scatterplot matrix

    Args:
        X (pd.DataFrame): input dataframe.
        dirResults (str): output filename.

    Returns:
        bool: true if the execution ended correctly.

    """
    pal = sns.light_palette("orange", reverse=False)
    sns.set(style="ticks", color_codes=True)
    fig = sns.pairplot(X, diag_kind="kde", kind="reg", markers="+", palette=pal)
    fig.savefig(dirResults + '\\00_ScatterplotMatrix.png')
    return True


def correlationMatrix(X: pd.DataFrame, annotationCell: bool = True) -> dict:
    """
    Produces the correlation matrix of an input dataframe

    Args:
        X (pd.DataFrame): input dataframe.
        annotationCell (bool, optional): when true, the value of correlation is reported into the Figure. Defaults to True.

    Returns:
        dict: output dicsionary containing the figures.

    """

    output_figures = {}
    d = X

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap = sns.light_palette("orange", reverse=False)

    # Draw the heatmap with the mask and correct aspect ratio
    figCorr = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, annot=annotationCell,
                          square=True, linewidths=.5, cbar_kws={"shrink": .5},
                          xticklabels=True, yticklabels=True)
    figure = figCorr.get_figure()
    output_figures['CorrelationMatrix'] = figure
    plt.close('all')
    return output_figures


def histogramVariables(K: pd.DataFrame, dirResults: str):
    """
    Produces an histogram for each variable of a DataFrame

    Args:
        K (pd.DataFrame): input DataFrame.
        dirResults (str): output filename.

    Returns:
        bool: DESCRIPTION.

    """
    for i in range(0, len(K.columns)):
        columnName = K.columns[i]
        plt.figure(figsize=(20, 10))
        if(np.issubdtype(K.iloc[:, i].dtype, np.number)):
            plt.hist(K.iloc[:, i], color='orange')
            plt.title('Histogram var: ' + str(columnName))
            plt.savefig(dirResults + '\\00_Hist_' + str(columnName) + '.png')
        else:
            sns.countplot(x=columnName, data=K, color='orange')
            plt.xticks(rotation=30)
            plt.title('Histogram var: ' + str(columnName))
            plt.savefig(dirResults + '\\00_Hist_' + str(columnName) + '.png')
        plt.close('all')
    return True


def subsample(dataset: pd.DataFrame, ratio: float = 1.0) -> pd.DataFrame:
    """
    Create a random subsample from the dataset with replacement

    Args:
        dataset (pd.DataFrame): pandas dataframe.
        ratio (float, optional): sampling ratio. Defaults to 1.0.

    Returns:
        sample (TYPE): output dataframe.

    """
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = rn.randrange(len(dataset))
        sample.append(dataset[index])
    return sample
