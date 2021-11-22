import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def plot_confusion_matrix_fromAvecm(ave_cm,
                                    classes: list,
                                    normalize: bool = True,
                                    title: str = None,
                                    cmap: plt.cm = plt.cm.Blues) -> plt.Figure:
    """
    This function prints and plots the confusion matrix from an average-precomputed confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        ave_cm (TYPE): DESCRIPTION.
        classes (list): DESCRIPTION.
        normalize (bool, optional): DESCRIPTION. Defaults to True.
        title (str, optional): DESCRIPTION. Defaults to None.
        cmap (plt.cm, optional): DESCRIPTION. Defaults to plt.cm.Blues.

    Returns:
        fig (TYPE): DESCRIPTION.

    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = ave_cm

    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def analyseClassificationCoefficients(X: pd.DataFrame,
                                      y: pd.Series,
                                      D_learning_results: pd.DataFrame,
                                      outputPath: str) -> dict:
    """
    This function evaluates the importance coefficients of the input features of a model

    Args:
        X (pd.DataFrame): Input pandas dataFrame.
        y (pd.Series): Input pandas series sith target label.
        D_learning_results (pd.DataFrame): Results dataframe obstained from a grid search (analytics.learning.grids).
        outputPath (str): Output filename path to save the results.

    Returns:
        dict: DESCRIPTION.

    """

    output_figures = {}
    # define the confusion matrix
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    for index, row in D_learning_results.iterrows():

        y_pred = row['MODEL'].predict(x_test)
        cm = confusion_matrix(y_test, y_pred)

        # plot the confusion matrix
        fig = plt.figure(figsize=(9, 9))
        ax = fig.gca()
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]))
        ax.set_xticklabels(labels=row['MODEL'].classes_, rotation=45)
        ax.set_yticklabels(labels=row['MODEL'].classes_, rotation=45)

        all_sample_title = 'Accuracy Score: {0}'.format(np.round(row['SCORE_TEST'], 2))
        plt.title(f"Model: {row['MODEL_NAME']}, {all_sample_title}", size=15)
        output_figures[f"{row['MODEL_NAME']}_confusionMatrix"] = fig

        # analyse output for QDA
        if row['MODEL_NAME'] == 'quadratic_discriminant_analysis':
            # Print the mean for each class
            # create a dataframe with one row for each feature of X
            features_list = list(X.columns)

            # extract coefficients riprendere da qui
            fig = plt.figure(figsize=(12, 10))
            means = row['MODEL'].means_

            means_scaled = scale(means)

            plt.imshow(means_scaled, cmap='bwr')
            ax = fig.gca()

            # set xticks
            ax.set_xticks(range(0, len(features_list)))
            ax.set_xticklabels(features_list, rotation=90)

            # set yticks
            ax.set_yticks(range(0, len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_, rotation=45)

            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('QDA means per class')
            output_figures[f"{row['MODEL_NAME']}_means"] = fig

        # analyse output for LDA
        elif row['MODEL_NAME'] == 'linear_discriminant_analysis':
            # Print coefficients
            # create a dataframe with one row for each feature of X
            features_list = list(X.columns)

            # extract coefficients riprendere da qui
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].coef_

            coefficients_scaled = scale(coefficients)

            plt.imshow(coefficients_scaled, cmap='bwr')
            ax = fig.gca()

            # set xticks
            ax.set_xticks(range(0, len(features_list)))
            ax.set_xticklabels(features_list, rotation=90)

            # set yticks
            ax.set_yticks(range(0, len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_, rotation=45)

            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('LDA coefficients')
            output_figures[f"{row['MODEL_NAME']}_coefficients"] = fig

        # analyse output for logistic regression
        elif row['MODEL_NAME'] == 'logistic_regression':
            # Print coefficients
            # create a dataframe with one row for each feature of X
            features_list = list(X.columns)

            # extract coefficients riprendere da qui
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].coef_

            coefficients_scaled = scale(coefficients)

            plt.imshow(coefficients_scaled, cmap='bwr')
            ax = fig.gca()

            # set xticks
            ax.set_xticks(range(0, len(features_list)))
            ax.set_xticklabels(features_list, rotation=90)

            # set yticks
            ax.set_yticks(range(0, len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_, rotation=45)

            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('Logistic regression coefficients')
            output_figures[f"{row['MODEL_NAME']}_coefficients"] = fig
        elif row['MODEL_NAME'] == 'naive bayes':

            # Print coefficients
            # create a dataframe with one row for each feature of X
            features_list = list(X.columns)

            # print variance
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].sigma_

            coefficients_scaled = scale(coefficients)

            plt.imshow(coefficients_scaled, cmap='bwr')
            ax = fig.gca()

            # set xticks
            ax.set_xticks(range(0, len(features_list)))
            ax.set_xticklabels(features_list, rotation=90)

            # set yticks
            ax.set_yticks(range(0, len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_, rotation=45)

            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('Naive bayes sigma')
            output_figures[f"{row['MODEL_NAME']}_sigma"] = fig

            # print mean
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].theta_

            coefficients_scaled = scale(coefficients)

            plt.imshow(coefficients_scaled, cmap='bwr')
            ax = fig.gca()

            # set xticks
            ax.set_xticks(range(0, len(features_list)))
            ax.set_xticklabels(features_list, rotation=90)

            # set yticks
            ax.set_yticks(range(0, len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_, rotation=45)

            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('Naive bayes theta')
            output_figures[f"{row['MODEL_NAME']}_theta"] = fig

        elif row['MODEL_NAME'] == 'decision tree':

            # Print coefficients
            # create a dataframe with one row for each feature of X
            features_list = list(X.columns)

            # print variance
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].feature_importances_

            # coefficients_scaled = scale(coefficients)

            plt.bar(features_list, coefficients)
            ax = fig.gca()

            # set xticks
            # ax.set_xticks(range(0,len(features_list)))
            ax.set_xticklabels(features_list, rotation=45)

            plt.xlabel('Feature name')
            plt.ylabel('Feature importance')
            plt.title('Decision tree Gini importance')
            output_figures[f"{row['MODEL_NAME']}_Gini"] = fig

            # save the decision tree
            dotfile = open(f"{outputPath}//dt.dot", 'w')
            tree.export_graphviz(row['MODEL'],
                                 out_file=dotfile,
                                 feature_names=features_list,
                                 class_names=row['MODEL'].classes_,
                                 rounded=True,
                                 proportion=False,
                                 precision=2,
                                 filled=True)
            dotfile.close()

            # http://webgraphviz.com/

        else:
            print(f"{row['MODEL_NAME']}, model not considered")
    return output_figures
