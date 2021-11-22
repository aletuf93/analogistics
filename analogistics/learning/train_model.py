import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#import sklearn packages
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import tree


# %% TRAINING WITH GRIDSEARCH CV REGRESSION


#train all linear regression models
def train_models_regression(X,y,models_regression,test_size=0.33,cv=5):

    # split into train and validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # define evaluation dataframe
    D_trainingResults=pd.DataFrame(columns=['MODEL_NAME','MODEL','PARAMS','SCORE_TEST','SCORE_VALIDATION'])
    
    scores = 'neg_mean_squared_error'
    for model in models_regression.keys():
        estimator = models_regression[model]['estimator']
        param_grid = models_regression[model]['param']
        
        clf = GridSearchCV(estimator, param_grid,cv=cv,scoring=scores, verbose=10)
        clf.fit(X_train, y_train)
        #clf.cv_results_
        MODEL=clf.best_estimator_
        SCORE_TEST=clf.best_score_
        PARAMS=clf.best_params_
        
        #validation_set
        y_pred=MODEL.predict(X_test)
        #scorer=get_scorer(scores)
        SCORE_VALIDATION=-mean_squared_error(y_test, y_pred)
        D_trainingResults=D_trainingResults.append(pd.DataFrame([[model, MODEL, PARAMS, SCORE_TEST, SCORE_VALIDATION ]],columns = D_trainingResults.columns))
    return D_trainingResults

# %% TRAINING WITH GRIDSEARCH CV REGRESSION
#train all linear classification models

def train_models_classification(X,y,models_classification,test_size=0.33,cv=3,scores = 'accuracy'): 
    # split into train and validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # define evaluation dataframe
    D_trainingResults=pd.DataFrame(columns=['MODEL_NAME','MODEL','PARAMS','SCORE_TEST',
                                            'ACCURACY',
                                            'PRECISION',
                                            'RECALL',
                                            'F1',
                                            'REPORT'])
    
    
    for model in models_classification.keys():
        estimator = models_classification[model]['estimator']
        param_grid = models_classification[model]['param']
        
        clf = GridSearchCV(estimator, param_grid,cv=cv,scoring=scores, verbose=7,n_jobs=-1)
        clf.fit(X_train, y_train)
        #clf.cv_results_
        MODEL=clf.best_estimator_
        SCORE_TEST=clf.best_score_
        PARAMS=clf.best_params_
        
        #validation_set
        y_pred=MODEL.predict(X_test)
        #scorer=get_scorer(scores)
        ACCURACY=accuracy_score(y_test, y_pred)
        PRECISION=precision_score(y_test, y_pred, average='weighted')
        RECALL = recall_score(y_test, y_pred,average='weighted')
        F1=f1_score(y_test, y_pred,average='weighted')
        
        #get classification report
        D_rep = classification_report(y_test, y_pred, target_names=MODEL.classes_)
        
        D_trainingResults=D_trainingResults.append(pd.DataFrame([[model, MODEL, PARAMS, SCORE_TEST, ACCURACY, PRECISION, RECALL, F1, D_rep ]],columns = D_trainingResults.columns))
    return D_trainingResults



# In[1]: #confusion matrix


def plot_confusion_matrix_fromAvecm(ave_cm, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix from an average-precomputed confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = ave_cm
    
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
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

# %%
def analyseClassificationCoefficients(X,y,D_learning_results,outputPath):
    
    
    output_figures={}
    #define the confusion matrix
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    for index, row in D_learning_results.iterrows():
        
        
        y_pred = row['MODEL'].predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        
        #plot the confusion matrix
        fig= plt.figure(figsize=(9,9))
        ax= fig.gca()
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]))
        ax.set_xticklabels(labels=row['MODEL'].classes_, rotation=45)
        ax.set_yticklabels(labels=row['MODEL'].classes_, rotation=45)
        
          
           
        all_sample_title = 'Accuracy Score: {0}'.format(np.round(row['SCORE_TEST'],2))
        plt.title(f"Model: {row['MODEL_NAME']}, {all_sample_title}", size = 15)
        output_figures[f"{row['MODEL_NAME']}_confusionMatrix"]=fig
        
        
        #analyse output for QDA
        if row['MODEL_NAME']=='quadratic_discriminant_analysis':
            #Print the mean for each class
            #create a dataframe with one row for each feature of X
            features_list = list(X.columns)
            
            
            #extract coefficients riprendere da qui
            fig = plt.figure(figsize=(12, 10))
            means = row['MODEL'].means_
            
            means_scaled = scale(means)
            
            
            
            plt.imshow(means_scaled,cmap='bwr')
            ax = fig.gca()
            
            #set xticks
            ax.set_xticks(range(0,len(features_list)))
            ax.set_xticklabels(features_list,rotation=90)
            
            #set yticks
            ax.set_yticks(range(0,len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_,rotation=45)
            
            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('QDA means per class')
            output_figures[f"{row['MODEL_NAME']}_means"]=fig
        
        #analyse output for LDA
        elif row['MODEL_NAME']=='linear_discriminant_analysis':
            #Print coefficients
            #create a dataframe with one row for each feature of X
            features_list = list(X.columns)
            
            
            #extract coefficients riprendere da qui
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].coef_
            
            coefficients_scaled = scale(coefficients)
            
            
            
            plt.imshow(coefficients_scaled,cmap='bwr')
            ax = fig.gca()
            
            #set xticks
            ax.set_xticks(range(0,len(features_list)))
            ax.set_xticklabels(features_list,rotation=90)
            
            #set yticks
            ax.set_yticks(range(0,len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_,rotation=45)
            
            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('LDA coefficients')
            output_figures[f"{row['MODEL_NAME']}_coefficients"]=fig
        
        #analyse output for logistic regression
        elif row['MODEL_NAME']=='logistic_regression':
            #Print coefficients
            #create a dataframe with one row for each feature of X
            features_list = list(X.columns)
            
            
            #extract coefficients riprendere da qui
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].coef_
            
            coefficients_scaled = scale(coefficients)
            
            
            
            plt.imshow(coefficients_scaled,cmap='bwr')
            ax = fig.gca()
            
            #set xticks
            ax.set_xticks(range(0,len(features_list)))
            ax.set_xticklabels(features_list,rotation=90)
            
            #set yticks
            ax.set_yticks(range(0,len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_,rotation=45)
            
            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('Logistic regression coefficients')
            output_figures[f"{row['MODEL_NAME']}_coefficients"]=fig
        elif row['MODEL_NAME']=='naive bayes':
            
            #Print coefficients
            #create a dataframe with one row for each feature of X
            features_list = list(X.columns)
            
            
            #print variance
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].sigma_
            
            coefficients_scaled = scale(coefficients)
            
            
            
            plt.imshow(coefficients_scaled,cmap='bwr')
            ax = fig.gca()
            
            #set xticks
            ax.set_xticks(range(0,len(features_list)))
            ax.set_xticklabels(features_list,rotation=90)
            
            #set yticks
            ax.set_yticks(range(0,len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_,rotation=45)
            
            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('Naive bayes sigma')
            output_figures[f"{row['MODEL_NAME']}_sigma"]=fig
            
            #print mean
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].theta_
            
            coefficients_scaled = scale(coefficients)
            
            
            
            plt.imshow(coefficients_scaled,cmap='bwr')
            ax = fig.gca()
            
            #set xticks
            ax.set_xticks(range(0,len(features_list)))
            ax.set_xticklabels(features_list,rotation=90)
            
            #set yticks
            ax.set_yticks(range(0,len(row['MODEL'].classes_)))
            ax.set_yticklabels(row['MODEL'].classes_,rotation=45)
            
            plt.colorbar()
            plt.xlabel('Feature name')
            plt.ylabel('Classes')
            plt.title('Naive bayes theta')
            output_figures[f"{row['MODEL_NAME']}_theta"]=fig
            
        elif row['MODEL_NAME']=='decision tree':
            
            #Print coefficients
            #create a dataframe with one row for each feature of X
            features_list = list(X.columns)
            
            
            #print variance
            fig = plt.figure(figsize=(12, 10))
            coefficients = row['MODEL'].feature_importances_
            
            #coefficients_scaled = scale(coefficients)
                   
            
            plt.bar(features_list, coefficients)
            ax = fig.gca()
            
            #set xticks
            #ax.set_xticks(range(0,len(features_list)))
            ax.set_xticklabels(features_list,rotation=45)
            
           
            plt.xlabel('Feature name')
            plt.ylabel('Feature importance')
            plt.title('Decision tree Gini importance')
            output_figures[f"{row['MODEL_NAME']}_Gini"]=fig
            
            
            #save the decision tree
            dotfile = open(f"{outputPath}//dt.dot", 'w')
            tree.export_graphviz(row['MODEL'], 
                                 out_file=dotfile, 
                                 feature_names=features_list,
                                 class_names = row['MODEL'].classes_,
                                 rounded = True, 
                                 proportion = False, 
                                 precision = 2, 
                                 filled = True)
            dotfile.close()
            
            #http://webgraphviz.com/
            
           
        else:
            print(f"{row['MODEL_NAME']}, model not considered")
    return output_figures
    
# %% SAMPLE THe SAME NUMBER Of ELEMENTS PER CLASS

def sampleClassWithSameCardinality(df, targetColumn, numRecordsPerclass=np.inf, minRecordPerclass=100, includeLower=False):
    D_class_stat = df[targetColumn].value_counts()
    
    # if the minimum number of records is above the threshold
    if min(D_class_stat) > minRecordPerclass:
        numSample = min(numRecordsPerclass,min(D_class_stat))
        df_learning = pd.DataFrame()
        for target in set(df[targetColumn]):
            df_learning = df_learning.append(df[df[targetColumn] == target].sample(numSample))
    
    # if for some classes there are few records than minRecordPerclass
    else:
        #identify the classes with few records
        fewRecordLabelsList=[]
        for target in set(df[targetColumn]):
            if D_class_stat[target]<minRecordPerclass:
                fewRecordLabelsList.append(target)
        
        D_class_stat_less = D_class_stat[fewRecordLabelsList]
        D_class_stat_greater = D_class_stat[[(i not in fewRecordLabelsList) for i in D_class_stat.index]]
        
        
        df_learning = pd.DataFrame()
        
        # same as before for the greater than
        numSample = min(numRecordsPerclass,min(D_class_stat_greater))
        for target in set(D_class_stat_greater.index):
            df_learning = df_learning.append(df[df[targetColumn] == target].sample(numSample))
        if includeLower:
            # the maximum possible for the lower
            numSample = min(D_class_stat_less)
            for target in set(D_class_stat_less.index):
                df_learning = df_learning.append(df[df[targetColumn] == target].sample(numSample))
            
    return df_learning