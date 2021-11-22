
# import machine learning methods
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#import keras NN methods
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import keras.metrics

#import dataframe packages
import numpy as np
import pandas as pd

#import graph packages
import seaborn as sns
import matplotlib.pyplot as plt


# %% useful links
'''
to draw NN
http://alexlenail.me/NN-SVG/LeNet.html
'''


# %%
def plot_history(model_history,keys):
    m,val_m = keys
    fig = plt.figure()
    plt.plot(model_history.history[m])
    plt.plot(model_history.history[val_m])
    plt.ylabel(m)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title(keys[0])
    plt.show()
    
    return fig

# %% train classification neural network
def trainNeuralNetworkClassification(X,y,model,testSize=0.33):
    
    output_eval={}
    output_fig={}
    
    #compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=[keras.metrics.Accuracy(), 
                                                                         keras.metrics.Precision(),
                                                                         keras.metrics.Recall()])
      
    # %% encode labels for multiclass classification problems
    enc = OneHotEncoder()
    Y = enc.fit_transform(y[:, np.newaxis]).toarray()
    
    
    # %% scale the input table
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # %% split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=testSize, random_state=42)
    
    # %% set early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=10, patience=200)
    
    # %% fit the model
    base_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4000, verbose=10, callbacks=[es])
    
    
    # %% evaluate the output
    evals = model.evaluate(X_test, y_test)
    output_eval['loss'] = evals[0]
    output_eval['accuracy'] = evals[1]
    output_eval['precision'] = evals[2]
    output_eval['recall'] = evals[3]
    
    # %% evaluate the learning rate
    
   
    output_fig['loss_epoch'] = plot_history(base_history,[f"{list(base_history.history.keys())[0]}",f"val_{list(base_history.history.keys())[0]}"])
    output_fig['accuracy_epoch'] = plot_history(base_history,[f"{list(base_history.history.keys())[1]}",f"val_{list(base_history.history.keys())[1]}"])
    output_fig['precision_epoch'] = plot_history(base_history,[f"{list(base_history.history.keys())[2]}",f"val_{list(base_history.history.keys())[2]}"])
    output_fig['recall_epoch'] = plot_history(base_history,[f"{list(base_history.history.keys())[3]}",f"val_{list(base_history.history.keys())[3]}"])
    
    
    # %% evaluate the confusion matrix
    
    #convert the y_test into a pandas series
    y_test_pandas = pd.DataFrame(y_test, columns = [i for i in range(0,y_test.shape[1])])
    x = y_test_pandas.stack()
    y_test_pandas = pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1)))
    
    #make predictions on the test set
    y_pred = model.predict_classes(X_test, verbose=1)
    
    #plot the confusion matrix
    cm = confusion_matrix(y_test_pandas, y_pred)

    #plot the confusion matrix
    fig1= plt.figure(figsize=(9,9))
    ax= fig1.gca()
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(labels= enc.categories_[0], rotation=45)
    ax.set_yticklabels(labels=enc.categories_[0], rotation=45)
    
    plt.title("Neural network confusion matrix", size = 15)
    
    output_fig['confusion_matrix_NN'] = fig1
    
    
    
    
    return model, output_eval, output_fig