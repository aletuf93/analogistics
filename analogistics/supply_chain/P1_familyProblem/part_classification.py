
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculateADICV2(D_mov: pd.DataFrame, itemfield: str, qtyfield: str, dateVar: str) -> pd.DataFrame:
    """
    Calculate the ADI and CV2 of an Item

    Args:
        D_mov (pd.DataFrame): Input movements DataFrame.
        itemfield (str): Column name of the item code.
        qtyfield (str): Column name of the quantity.
        dateVar (str): Column name of the timestamp.

    Returns:
        D_demandPatterns (TYPE): DESCRIPTION.

    """

    # identify the number of days of the input dataset
    N_Days = max(D_mov[dateVar]) - min(D_mov[dateVar])
    N_Days = N_Days.days

    D_demandPatterns = pd.DataFrame(columns=['ITEMCODE', 'ADI', 'CV2'])
    for item in set(D_mov[itemfield]):
        # item='17092774'
        df_filtered = D_mov[D_mov[itemfield] == item]
        CV2 = (np.std(df_filtered[qtyfield]) / np.mean(df_filtered[qtyfield])) ** 2

        # ADI in days
        df_filtered = df_filtered.sort_values(by=dateVar)
        ADI = len(df_filtered) / N_Days
        D_demandPatterns = D_demandPatterns.append(pd.DataFrame([[item, ADI, CV2]], columns=D_demandPatterns.columns))
        return D_demandPatterns


def returnsparePartclassification(ADI: float, CV2: float) -> str:
    """
    return the demand pattern of the spare part

    Args:
        ADI (float): identify the ADI value of a spare part.
        CV2 (float): identify the CV2 value of a spare part.

    Returns:
        str: String indicating the demand pattern.

    """

    if (ADI > 1.32) & (CV2 > 0.49):
        return "LUMPY"
    elif (ADI <= 1.32) & (CV2 > 0.49):
        return "ERRATIC"
    elif (ADI > 1.32) & (CV2 <= 0.49):
        return "INTERMITTENT"
    elif (ADI <= 1.32) & (CV2 <= 0.49):
        return "STABLE"


def demandPatternADICV2(df_results: pd.DataFrame, setTitle: str, draw: bool = False):
    """
    Plot the demand patterns

    Args:
        df_results (pd.DataFrame): Input DataFrame with columns: ADI (with the ADI value); CV2 (with the CV2 value);
        frequency (with the number of lines for each itemcode).
        setTitle (str): title of the figure.
        draw (bool, optional): If true plot the graph, otherwise only output the numbers. Defaults to False.

    Returns:
        fig (plt.figure): Output Figure.
        fig1 (plt.figure): Output Figure.
        numLumpy (float): Number of Lumpy items.
        numIntermittent (float): Number of Intermittent items.
        numErratic (float): Number of Erratic items.
        numStable (float): Number of stable items.

    """

    fig = fig1 = []
    df_results = df_results.dropna()
    if len(df_results) > 0:
        # calculate numerical results
        numLumpy = len(df_results[(df_results.ADI <= 1.32) & (df_results.CV2 > 0.49)])
        numErratic = len(df_results[(df_results.ADI > 1.32) & (df_results.CV2 > 0.49)])
        numIntermittent = len(df_results[(df_results.ADI <= 1.32) & (df_results.CV2 <= 0.49)])
        numStable = len(df_results[(df_results.ADI > 1.32) & (df_results.CV2 <= 0.49)])
        totParts = numLumpy + numErratic + numIntermittent + numStable

        if draw:
            # if totParts==len(df_results):
            A = np.array([[numLumpy, numErratic], [numIntermittent, numStable]])
            A_text = np.array([[f"Lumpy \n {numLumpy} parts \n Perc: {np.round(numLumpy*100/totParts, 2)} %", f"Erratic \n {numErratic} parts \n Perc: {np.round(numErratic*100/totParts, 2)} %"],
                               [f"Intermittent \n {numIntermittent} parts \n Perc: {np.round(numIntermittent*100/totParts, 2)} %", f"Stable \n {numStable} parts \n Perc: {np.round(numStable*100/totParts, 2)} %"]])
            fig, ax = plt.subplots()
            im = ax.imshow(A, cmap="YlOrRd")

            plt.title(f"Parts set: {setTitle}")

            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)

            for i in range(0, 2):
                for j in range(0, 2):
                    ax.text(j, i, A_text[i, j],
                            ha="center", va="center", color="k")

            # plot ADI and CV2

            fig1 = plt.figure()
            plt.scatter(df_results['ADI'], df_results['CV2'], df_results['frequency'],
                        color='skyblue', marker='o')
            plt.axvline(x=1.32, c='orange', linestyle='--')
            plt.axhline(y=0.49, c='orange', linestyle='--')
            plt.xlabel('ADI')
            plt.ylabel('CV2')
            plt.title(f"Demand pattern: {setTitle}")

    return fig, fig1, numLumpy, numIntermittent, numErratic, numStable
