
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

from analogistics.transform import dummyColumns
from analogistics.explore import correlationMatrix
from analogistics.learning.linear_grids import fit_linear_reg


def diagnoseForFeatureSelection(X: pd.DataFrame,
                                y: pd.Series,
                                dirResultsMachineLearning: str,
                                diagnoseModels: list) -> bool:
    """
    The function uses several strategies for feature selection

    Args:
        X (pd.DataFrame): Input pandas dataframe with the observation.
        y (pd.Series): Input series containing the target variable.
        dirResultsMachineLearning (str): output directory to save reports and figures.
        diagnoseModels (list): list of the models to test.

    Returns:
        bool: DESCRIPTION.

    """

    Q = dummyColumns(X)
    Y = pd.concat([Q, y], axis=1)
    plt.close('all')

    # Analyse correlation between variables
    correlationMatrix(Y, dirResultsMachineLearning, annotationCell=False)

    # Analysis of correlation with the target variable
    if diagnoseModels.count('correlation') > 0:
        selectByCorrelation(X, y, 1, dirResultsMachineLearning, True)

    # Study of the variance of the variables
    if diagnoseModels.count('variance') > 0:
        selectByVariance(X, 1, dirResultsMachineLearning, True)

    # Analysis of the selection of the variables by lasso
    if diagnoseModels.count('lasso') > 0:
        selectByLassoL1(X, y, 1, dirResultsMachineLearning, True)

    # Check if a PCA can express the variance of the dataset
    if diagnoseModels.count('PCA') > 0:
        PCAplot(len(Q.columns), Q, dirResultsMachineLearning, diagnose=True)

    # Plot the curve of the forward stepwise
    if diagnoseModels.count('forward stepwise') > 0:
        selectByForwardStepwiseSelection(X, y, dirResultsMachineLearning, len(X.columns), saveFig=True)

    return True


def selectPreprocessFeature(X: pd.DataFrame,
                            y: pd.Series,
                            value: float,
                            model: str) -> tuple:
    """
    Args:
        X (pd.DataFrame): feature dataframe.
        y (pd.Series): target variable.
        value (float): indicates the parameter of the model (if any). For correlation it is the minimum treshold.
         For forward stepwise it is the number of features to select.For PCA it is the number of principal component.
        model (str): type of the model chosen for feature selection.

    Returns:
        tuple: DESCRIPTION.

    """

    if model == 'correlation':
        return selectByCorrelation(X, y, value, False)
    elif model == 'variance':
        return selectByVariance(X, value, False)
    elif model == 'lasso':
        return selectByLassoL1(X, y, value, False)
    elif model == 'tree':
        return selectByTree(X, y)
    elif model == 'forward stepwise':
        return selectByForwardStepwiseSelection(X, y, value, False)
    elif model == 'PCA':
        return PCAplot(value, X, False)
    else:  # if no feature selection, the dataset is only returned as numerical (dummy) to code strings
        return dummyColumns(X)


def selectByCorrelation(X: pd.DataFrame,
                        y: pd.DataFrame,
                        corrThreshold: float,
                        diagnose: bool = False) -> tuple:
    """
    Select the features of the input dataframe X
    based on the correlation with y

    Args:
        X (pd.DataFrame): dataframe of the attributes.
        y (pd.DataFrame): dataframe with the target variable.
        corrThreshold (float): correlation treshold 0->1.
        diagnose (bool, optional): defaults to False. if true generates the correlation of each variable to the target var. Defaults to False.

    Returns:
        tuple: tuple composed of: dataframe with only the features above the correlation threshold; dictionary containing the figure of the correlation for each variable (if diagnose is true).

    """

    output_figure = {}
    targetVariable = y.columns[0]
    Q = dummyColumns(X)
    Q = pd.concat([Q, y], axis=1)
    cor = Q.corr()
    cor_target = abs(cor[targetVariable])

    if diagnose:
        numFeatSelected = []
        # count the number of selected features for each value of correlation
        for i in range(1, 101):
            val = i / 100
            nn = len(cor_target[cor_target > val])
            # print(nn)
            numFeatSelected.append(nn)
        fig1 = plt.figure()
        plt.plot(range(1, 101), numFeatSelected)
        plt.title('Correlation graph')
        plt.xlabel('Corr threshold %')
        plt.ylabel('Num. selected features')
        output_figure['CorrelationChart'] = fig1
        plt.close('all')

    # Selecting highly correlated features
    relevant_features = cor_target[cor_target > corrThreshold]
    relevant_features = list(relevant_features.index.values)
    try:  # check if the target feature is contained within the relevant features and remove
        relevant_features.remove(targetVariable)
    except Exception as e:
        print(e)
    Z = Q.loc[:, relevant_features]
    return Z, output_figure


def selectByVariance(X: pd.DataFrame,
                     perc: float,
                     diagnose: bool = False):
    """
    Select the features of the input dataframe X
    based on their variance

    Args:
        X (pd.DataFrame): dataframe of the attributes.
        perc (float): variance treshold.
        diagnose (bool, optional): defaults to False if true render the plot of the selected features depending o the variance. Defaults to False.

    Returns:
        res (TYPE): dataframe output dataframe with selected variables.
        output_figure (TYPE): dictionary containing the figures with the plot of the selected features depending o the variance.

    """

    output_figure = {}
    # It select the features having the same value in more than perc of the samples
    Q = dummyColumns(X)
    if diagnose:
        numFeatSelected = []
        for i in range(1, 101):
            val = i / 100
            sel = VarianceThreshold(threshold=(val))
            sel.fit_transform(Q)
            nn = len(Q.columns[sel.get_support()])
            # print(nn)
            numFeatSelected.append(nn)
        fig1 = plt.figure()
        plt.plot(range(1, 101), numFeatSelected)
        plt.title('Variance graph')
        plt.xlabel('Variance threshold %')
        plt.ylabel('Num. selected features')
        output_figure['VarianceChart'] = fig1
        plt.close('all')

    sel = VarianceThreshold(threshold=(perc))
    Z = sel.fit_transform(Q)
    feature_idx = sel.get_support()
    feature_name = Q.columns[feature_idx]
    res = pd.DataFrame(Z, columns=feature_name)
    return res, output_figure


def selectByLassoL1(X: pd.DataFrame,
                    y: pd.DataFrame,
                    value: float,
                    diagnose: bool = False):
    """
    Select the features of the input dataframe X
    based on the coefficients of a Lasso regression

    Args:
        X (pd.DataFrame): dataframe of the attributes.
        y (pd.DataFrame): series of the target variable.
        value (float): threshold.
        diagnose (bool, optional): DESCRIPTION. Defaults to False.

    Returns:
        res (TYPE): DESCRIPTION.
        output_figure (TYPE): DESCRIPTION.

    """

    output_figure = {}
    # lasso feature selection works for regression models
    Q = dummyColumns(X)
    clf = linear_model.LassoCV(cv=5)

    if diagnose:
        numFeatSelected = []
        for i in range(1, 101):
            val = i / 100
            sfm = SelectFromModel(clf, threshold=val)
            sfm.fit(Q, y)
            nn = len(Q.columns[sfm.get_support()])
            # print(nn)
            numFeatSelected.append(nn)
        fig1 = plt.figure()
        plt.plot(range(1, 101), numFeatSelected)
        plt.title('Lasso graph')
        plt.xlabel('Coeff threshold %')
        plt.ylabel('Num. selected features')
        output_figure['LassoChart'] = fig1
        plt.close('all')

    # _, _, _, alphaValue, _= LassoRegressionCV(Q,y,'',nFolds=5, saveFig=False) #previous version with lasso from ZO_RegressionLinearModel
    # las=LassoRegressionComplete(Q,y,alphaValue)

    sfm = SelectFromModel(clf, threshold=value)
    sfm.fit(Q, y)
    # n_features = sfm.transform(X).shape[1]
    # model = SelectFromModel(las, prefit=True,threshold=0.25)

    Z = sfm.transform(Q)
    feature_idx = sfm.get_support()
    feature_name = Q.columns[feature_idx]
    res = pd.DataFrame(Z, columns=feature_name)
    return res, output_figure


def selectByTree(X: pd.DataFrame, y: pd.Series):
    """
    select feature using a minimum threshold on the number of times a feature is chosen as
    a decision variables for a decision tree

    Args:
        X (TYPE): dataframe of the attributes .
        y (TYPE): dataframe with the target variable.

    Returns:
        res (TYPE): output dataframe with selected variables.

    """

    # lasso feature selection works for regression models
    Q = dummyColumns(X)
    tree = ExtraTreesClassifier(n_estimators=50).fit(Q, y)
    model = SelectFromModel(tree, prefit=True)
    Z = model.transform(Q)
    feature_idx = model.get_support()
    feature_name = Q.columns[feature_idx]
    res = pd.DataFrame(Z, columns=feature_name)
    return res


def ridgePath(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    Generate the path of the ridge regression using different alphas

    Args:
        df (pd.DataFrame): dataframe of the attributes.
        y (pd.Series): dataframe with the target variable.

    Returns:
        output_figure (dict): dictionary with the figure.

    """

    output_figure = {}

    X = dummyColumns(df)
    columnNames = list(pd.concat([X, y], axis=1))
    columnNames[-1] = []
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X, y)
        coefs.append(ridge.coef_[0])

    # Display results
    fig1 = plt.figure(figsize=(25, 10))
    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()
    plt.legend(columnNames)
    output_figure['RidgePath'] = fig1
    plt.close('all')
    return output_figure


def lassoPath(df: pd.DataFrame, y: pd.Series) -> dict:
    """
    Generate the path of the lasso regression using different alphas

    Args:
        df (pd.DataFrame): dataframe of the attributes .
        y (pd.Series): dataframe with the target variable.

    Returns:
        dict: dictionary with the figure.

    """

    output_figure = {}
    X = dummyColumns(df)

    columnNames = list(pd.concat([X, y], axis=1))
    columnNames[-1] = []

    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    coefs = []
    for a in alphas:
        lasso = linear_model.Lasso(alpha=a, fit_intercept=False)
        lasso.fit(X, y)
        coefs.append(lasso.coef_)

    # Display results
    fig1 = plt.figure(figsize=(25, 10))
    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Lasso coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()
    plt.legend(columnNames)
    output_figure['LassoPath'] = fig1
    plt.close('all')
    return output_figure


def selectByForwardStepwiseSelection(df: pd.DataFrame, y: pd.Series, n_feat: int):
    """
    Select features using forward stepwise selection

    Args:
        df (pd.DataFrame): dataframe of the attributes .
        y (pd.Series): series with the target variable.
        n_feat (int): number of features to consider.

    Returns:
        X_res (TYPE): dataframe with selected features.
        output_figure (TYPE): dictionary of figures.
        output_df (TYPE): dictionary of dataframes.

    """

    output_figure = {}
    output_df = {}

    # Converto eventuali variabili categoriche
    X = dummyColumns(df)
    k = len(X.columns)

    columnNames = ['Features', 'RSS', 'R_squared']
    resultFSs = pd.DataFrame(columns=columnNames)

    remaining_features = list(X.columns.values)
    features = []
    numb_features = [np.inf]
    RSS_list, R_squared_list = [np.inf], [np.inf]  # Due to 1 indexing of the loop...
    features_list = dict()

    for i in range(1, k + 1):
        best_RSS = np.inf

        for combo in itertools.combinations(remaining_features, 1):
            RSS = fit_linear_reg(X[list(combo) + features], y)  # Store temp result

            if RSS[0] < best_RSS:
                best_RSS = RSS[0]
                best_R_squared = RSS[1]
                best_feature = combo[0]

        # Updating variables for next loop
        features.append(best_feature)
        if len(remaining_features) > 0:
            remaining_features.remove(best_feature)

        # Saving values for plotting
        RSS_list.append(best_RSS)
        R_squared_list.append(best_R_squared)
        features_list[i] = features.copy()

        listResult = []
        listResult.append(features_list[i])
        listResult.append([round(best_RSS, 3)])
        listResult.append([round(best_R_squared, 3)])
        listResult = [listResult]
        row = pd.DataFrame(listResult, columns=columnNames)
        resultFSs = resultFSs.append(row)

        numb_features.append(len(features))
    # resultFSs.to_excel(dirResults+'\\00-ForwardStepwiseSelection.xlsx')

    # Salvo le variabili
    s = resultFSs['Features']
    mlb = preprocessing.MultiLabelBinarizer()
    aaa = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=resultFSs.index)
    bbb = pd.concat([resultFSs, aaa], axis=1)

    output_df['ForwardStepwiseSelection'] = bbb

    # Store in DataFrame
    df = pd.DataFrame({'numb_features': numb_features, 'RSS': RSS_list, 'R_squared': R_squared_list})
    df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
    # df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]

    output_df['ForwardStepwiseSelection_min'] = df_min

    df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
    df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(1, 2, 1)

    ax.scatter(df.numb_features, df.RSS, alpha=.2, color='darkblue')
    ax.set_xlabel('# Features')
    ax.set_ylabel('RSS')
    ax.set_title('RSS - Forward Stepwise selection')
    ax.plot(df.numb_features, df.min_RSS, color='r', label='Best subset')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(df.numb_features, df.R_squared, alpha=.2, color='darkblue')
    ax.plot(df.numb_features, df.max_R_squared, color='r', label='Best subset')
    ax.set_xlabel('# Features')
    ax.set_ylabel('R squared')
    ax.set_title('R_squared - Forward Stepwise selection')
    ax.legend()
    plt.show()
    output_figure['ForwardStepwiseSelection'] = fig
    plt.close('all')

    result = resultFSs.reset_index()
    selected_features = result.Features[n_feat - 1]

    X_res = X.loc[:, selected_features]
    return X_res, output_figure, output_df


def PCAplot(n_comp: int, df: pd.DataFrame, diagnose=False):
    """

    Args:
        n_comp (int): number of components of the PCA.
        df (pd.DataFrame): dataframe of the attributes .
        diagnose (TYPE, optional): if True produces the Pareto curve of the number of components. Defaults to False.

    Returns:
        pd.DataFrame: output compressed DataFrame.
        output_figure (TYPE): dictionary of figures.
        output_df (TYPE): dictionary of dataframes.

    """

    output_figure = {}
    output_df = {}

    D_Table = dummyColumns(df)
    # apply PCA
    data_scaled = pd.DataFrame(preprocessing.scale(D_Table), columns=D_Table.columns)
    pca = PCA(n_components=n_comp)
    PC = pca.fit_transform(data_scaled)

    if diagnose:
        components = pd.DataFrame(pca.components_, columns=data_scaled.columns)
        output_df['PCA'] = components

        var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)

        # Plot variance explaination
        fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.ylabel('% Variance Explained')
        plt.xlabel('# of Features')
        plt.title('PCA Analysis')
        plt.ylim(30, 100.5)
        plt.style.context('seaborn-whitegrid')
        plt.plot(np.arange(1, len(var) + 1), var)
        output_figure['PCA_varianceExplanation'] = fig

        # Plot graph PCA
        if n_comp == 2:  # if only two components, they can be plotted
            fig1 = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            plt.scatter(PC[:, 0], PC[:, 1], color='orange')
            plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
            plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
            plt.axis('equal')
            plt.tight_layout()
            plt.title('Principal Component plot')
            output_figure['PCA_plot'] = fig1
            plt.close('all')
    return pd.DataFrame(PC), output_figure, output_df
