#!/usr/bin/env python
# coding: utf-8

# # Functions

# **for project: "miRNA Biomarker for Lung Cancer Diagnostics - Selecting a test panel for patient classification -"**

# ## Import Packages & Modules

# In[1]:


# Import all Packages & Modules

# IPython
from IPython.display import Image

# mlxtend
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.classifier import StackingCVClassifier

# SciPy
from scipy.stats import normaltest

# sklearn
import sklearn

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier 

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split 

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz  

# subprocess
from subprocess import call

# xgboost
from xgboost import XGBClassifier

# yellowbrick
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ROCAUC

# matplotlib
import matplotlib
import matplotlib.pyplot as plt

# missingno
import missingno

# numpy
import numpy as np

# pandas
import pandas as pd

# pickle (for saving and loading data)
import pickle # 

# seaborn
import seaborn as sns

# sys (enables to exit execution of code)
import sys


# ## Functions

# In[2]:


# Function #1
# Function for feature_selection from Feature Importance of Tree-based Classifiers

def feature_selection(results_dict, tree_clf, name_clf, X_train, y_train, n=20):
    
    # n is the number of Features + Importances added to the results dictionary (default=20)
    # and printed during process for each classifier
    # if n=None all Features + Importances are added and printed
    
    count = 0
    
    for classifier in tree_clf:
        # use classifiers to fit data
        classifier.fit(X_train, y_train)
    
    
        # calculate Feature Importances, rank by value and get Feature Name
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_name = X_train.columns[indices]
    
        # Write first n Ranks to dictionary results_dict as 
        # "name_clf: (feature_name, importance)", default is n=20
        results_dict[name_clf[count]] = list(zip(feature_name[0:n], importances[indices][0:n]))     
        
        count +=1
    
        # Print the feature ranking (only the first n Ranks), default is n=20
        print(80*"-") 
        print(classifier)
        print("Feature ranking:")
        for feature in range(n) if isinstance(n, int) else range(importances.size):
            print("%d. Feature: %s (%f)" % (feature + 1, X_train.columns[indices[feature]], importances[indices[feature]]))
        
    return results_dict


# In[3]:


# Function #2
# Function for RFE Feature Selection

def rfe_selection(results_dict, non_tree_clf, nt_name_clf, X_train, y_train):
    
    count = 0
    
    for classifier in non_tree_clf:
        
        # Initialize RFE Feature Selector
        rfe_selector = RFE(estimator=classifier, n_features_to_select=1, step=1) 
        
        # fit data
        rfe_selector.fit(X_train, y_train) 
        
        # get the indices of Feature Ranking
        indices = rfe_selector.ranking_
        
        # Create a Lists with ordered feature names 
        # from ranking (list with ordered positions of object in DataFrame)
        feature_name = []
        
        # to get the rank (index of ranking) another list is created 
        feature_rank = []
        
        for feature in indices:
            
            # [feature-1] because columns starts at index 0 = Position1 in Dataframe
            feature_name.append(X_train.columns[feature-1])
            
            # make a list of indices array and get the index of the feature 
            # (+1, because index of this list starts at 0) = rank of feature
            feature_rank.append(list(indices).index(feature)+1)
    
        # Write first 20 Ranks to dictionary results_dict as 
        # "nt_name_clf: (feature_name, feature_rank)"
        results_dict[nt_name_clf[count]] = list(zip(feature_name[0:20], feature_rank[0:20]))     
        
        count +=1
    
        # Print the feature ranking (only the first 20 Ranks)
        print(80*"-") 
        print(classifier)
        print("Feature ranking:")
        for feature in range(20):
            print("%d. Feature: %s" % (feature + 1, feature_name[feature]))
        
    return results_dict


# In[4]:


# Function #3
# Function to create Dataframe with selected Features

def dataframe_selection(results_fs, X_train):
    
    # First create a list of selected Features from Feature Selection Results
    selected_features = [feature_name[0] for feature_name in results_fs]

    # Then create exclusion list for dropping (default=X_train)
    exclusion_list = list(set(X_train.columns) - set(selected_features))

    # Now drop all from X_train except selected_features and create new dataframe
    X_new = X_train.drop(exclusion_list, axis=1)
    
    return X_new    


# In[5]:


# Function #4
# Function for model_evaluation

def model_evaluation(results_dict, model_clf, name_model, X_train, y_train):
    
    count = 0
    
    for model in model_clf:
        # use model to be cross validated with data and desired metrics
        validation = cross_validate(model, X_train, y_train, scoring = ('accuracy', 'roc_auc_ovr', 'precision_macro', 'f1_macro'))

        # get metric scores from validation dictionary and calculate mean
        accuracy = np.mean(validation['test_accuracy'])
        roc_auc = np.mean(validation['test_roc_auc_ovr'])
        precision = np.mean(validation['test_precision_macro'])
        f1 = np.mean(validation['test_f1_macro'])


        # Write all mean scores to dictionary results_cv as 
        # "name_model: {"Accuracy":accuracy,"Roc_AUC":roc_auc,"Precision":precision,"F1":f1}"
        results_dict[name_model[count]] = {"Accuracy":accuracy,"Roc_AUC":roc_auc,"Precision":precision,"F1":f1}

        # Print the scores for each model
        print(80*"-") 
        print(model)
        print("Scores:")
        print(results_dict[name_model[count]])

        count +=1
        
    return results_dict


# In[6]:


# Function #5
# Function to print the n best model + selection combinations (n highest values with their index, column) 
# for a certain dataframe (default n=3)

def top_model(dataframe, n=3):
    
    # create an numpy array of all values
    # (as copy to ensure original dateframe is not touched)
    values = dataframe.to_numpy(copy=True)
    
    # sort values to get the max per row (=model)
    values.sort()
    
    # make a list of the max values for each row (=model)

    top_model = []

    for val in range(values.shape[0]):
        top_model.append(values[val][-1])
    
    # sort the list descending to get the highest values in the whole dataframe first
    top_model.sort(reverse=True)
    
    # print out the Top n Values (of all in dataframe)
    for tm in range(n) if n<= values.shape[0] else range(values.shape[0]):
       
        # get index and colum name (=Model + Selection)
        idx, clm = np.where(dataframe == top_model[tm])
        
        print("TOP:", tm+1)
        print("Value:", top_model[tm])
        print("Model:", dataframe.index[idx][0])
        print("Feature Selection:", dataframe.columns[clm][0])
        print(65*"-")


# In[17]:


# Function #6
# Function for multibar plotting

def multibar_plot(bars, label_list, name, title, xlb, save_name, barWidth = 0.1, ):
    
    # gives an array with indeces of dataframe @ postion [0] in bars list
    r1 = np.arange(len(bars[0]))
    
    # Set position of other bars relative to r1 on X axis and save it in position_list
    # Set barWidth = 0.1 by default
    position_list = []
    for ps in list(range(1, len(bars))):
        r2 = [x + barWidth*ps for x in r1]
        position_list.append(r2)

    # Make the plot
    plt.bar(r1, bars[0], width=barWidth, edgecolor='white', label=label_list[0])
    
    for num in list(range(1, len(bars))):
        plt.bar(position_list[num-1], bars[num], width=barWidth, edgecolor='white', label=label_list[num])
        
    # doesn´t work if less labels than bars! make sure labels list is complete!
        
    # Add xticks on the middle of the group bars
    plt.xlabel(xlb, fontweight='bold') #, fontsize='large')
    
    # to set ticks on the middle bar
    plt.xticks([r + barWidth*(int((len(label_list))/2)) for r in range(len(bars[0]))], name)
 
    # Create legend & Show graphic
    plt.legend(loc='lower center', ncol=int(len(bars)/2), bbox_to_anchor=(0.5, -0.5))
    plt.title(label=title, fontweight='bold', fontsize='large')
    
    # Save figure and show
    plt.savefig(save_name, transparent=True, dpi=300);
    plt.show()


# In[8]:


# Function #7
# Function for RandomSearchCV + printing results

def random_searching(model, parameters, X_train, y_train, X_test, y_test, seed):
    
    # Perform Random search on the classifier using 'precision_micro' as the scoring method 
    #(micro = Calculate metrics globally by counting the total true positives, false negatives and false positives.)
    random_obj = RandomizedSearchCV(model, parameters, scoring='precision_micro', n_jobs = -1, verbose=5, n_iter=100, cv=5, random_state=seed)

    # Fit the Random search object to the training data and find the optimal parameters
    random_fit = random_obj.fit(X_train, y_train)
    
    # Fit the unoptimzed model
    model_fit = model.fit(X_train, y_train)

    # Get the estimators
    best_model = random_fit.best_estimator_

    # Make predictions using the unoptimized and optimized model
    predictions = model_fit.predict(X_test)
    best_predictions = best_model.predict(X_test)
    
    probabilities = model_fit.predict_proba(X_test)
    best_probabilities = best_model.predict_proba(X_test)
    
    # get all the metrics of optimized and unoptimized model
    accuracy = accuracy_score(y_test, predictions)
    accuracy_best = accuracy_score(y_test, best_predictions)
    
    roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr')
    roc_auc_best = roc_auc_score(y_test, best_probabilities, multi_class='ovr')
    
    precision = precision_score(y_test, predictions, average='micro')
    precision_best = precision_score(y_test, best_predictions, average='micro')
    
    f1 = f1_score(y_test, predictions, average='micro')
    f1_best = f1_score(y_test, best_predictions, average='micro')
    
    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print("Accuracy score Unoptimized:", accuracy)
    print("Roc_AUC score Unoptimized:", roc_auc)
    print("Precision score Unoptimized:", precision)
    print("F1 score Unoptimized:", f1)
   
    print("\nOptimized Model\n------")
    print("Accuracy score Optimized:", accuracy_best)
    print("Roc_AUC score Optimized:", roc_auc_best)
    print("Precision score Optimized:", precision_best)
    print("F1 score Optimized:", f1_best)
    print(best_model)


# In[9]:


# Function #8
# Function for GridSearchCV + printing and saving results

def grid_searching(results_dict, model, name_model, parameters, X_train, y_train, X_test, y_test):
    
    # Perform grid search on the classifier using 'precision_micro' as the scoring method 
    # (micro = Calculate metrics globally by counting the total true positives, false negatives and false positives.)
    grid_obj = GridSearchCV(model, parameters, scoring='precision_micro', n_jobs = -1, verbose=5, cv=5)

    # Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(X_train, y_train)
    
    # Fit the unoptimzed model
    model_fit = model.fit(X_train, y_train)

    # Get the estimators
    best_model = grid_fit.best_estimator_

    # Make predictions using the unoptimized and optimized model
    predictions = model_fit.predict(X_test)
    best_predictions = best_model.predict(X_test)
    
    probabilities = model_fit.predict_proba(X_test)
    best_probabilities = best_model.predict_proba(X_test)
    
    # get all the metrics of optimized and unoptimized model
    accuracy = accuracy_score(y_test, predictions)
    accuracy_best = accuracy_score(y_test, best_predictions)
    
    roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr')
    roc_auc_best = roc_auc_score(y_test, best_probabilities, multi_class='ovr')
    
    precision = precision_score(y_test, predictions, average='micro')
    precision_best = precision_score(y_test, best_predictions, average='micro')
    
    f1 = f1_score(y_test, predictions, average='micro')
    f1_best = f1_score(y_test, best_predictions, average='micro')
    
    
    # Write all mean scores to dictionary results_dict as 
    # "name_model: {"Accuracy":accuracy_best,"Roc_AUC":roc_auc,"Precision":precision,"F1":f1}"
    results_dict[name_model[0]] = {"Accuracy":accuracy,"Roc_AUC":roc_auc,"Precision":precision,"F1":f1}
    results_dict[name_model[1]] = {"Accuracy":accuracy_best,"Roc_AUC":roc_auc_best,"Precision":precision_best,"F1":f1_best}
    
    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print("Accuracy score Unoptimized:", accuracy)
    print("Roc_AUC score Unoptimized:", roc_auc)
    print("Precision score Unoptimized:", precision)
    print("F1 score Unoptimized:", f1)
   
    print("\nOptimized Model\n------")
    print("Accuracy score Optimized:", accuracy_best)
    print("Roc_AUC score Optimized:", roc_auc_best)
    print("Precision score Optimized:", precision_best)
    print("F1 score Optimized:", f1_best)
    print(best_model)


# In[10]:


# Function #9
# Function to print Value and Rank of certain metric in a results dictionary

def viz_summary(results_dict, metric):
    print(metric, "Value:")
    print(results_dict.T[metric])
    print(40*"-")
    print(metric, "Rank:")
    print(results_dict.T[metric].rank())


# In[11]:


# Function #10
# Function for recursive feature reduction

def feature_reduce(results_reduction, model_reduction, name_list, feature_list, elim_features, X_train, y_train, n=11):
        
    name_reduction = name_list.copy()
    feature_reduction = feature_list.copy()

    # set default for n=11 iterations =(From All20 to TOP10)
    for n in range(n):
        
        # Create exclusion list for dropping (default=X_train)
        exclusion_list = list(set(X_train.columns) - set(feature_reduction))
        
        # Now drop all from X_train except feature_reduction and create new dataframe (first iteration all)
        X_reduce = X_train.drop(exclusion_list, axis=1)
        
       
        # Use model_evaluation function with reduced Features Dataset
        model_evaluation(results_reduction, model_reduction, name_reduction, X_reduce, y_train)
    
        # return the list with features used (only for debugging)
        elim_features.append(X_reduce.columns)
    
        # reduce feature list by 1 Feature
        feature_reduction.pop()
    
        # get new key for results dictionary
        name_reduction.pop(0)
    
    return results_reduction


# In[12]:


# Function #11
# Function for plotting Classification Report and Final Precision

def score_eval(y_test, y_pred):
    
    # precision micro = Calculate metrics globally by counting the total true positives,
    # false negatives and false positives.
    score = precision_score(y_test, y_pred, average='micro')
    
    print('Final Precision Score (micro):', score)
    print('----' * 15)
    print('Classification Report')
    print(classification_report(y_test, y_pred))
    print('----' * 15)
    
    return score


# In[13]:


# Function #12
# Function to print Heatmap

def heatmap(correlation, name_savefig, cmap):

    # Create Correlation Heatmap for all values (with minimum value -1 for positive or negative correlation)
    # alternative cmaps: inferno, seismic, magma, icefire
    plt.figure(figsize=(20,20))
    heatmap = sns.heatmap(correlation, linewidths=0.5, vmin=-1, cmap=cmap, annot=True)
    
    # save figure
    plt.savefig(name_savefig, transparent=True, dpi=300)        


# In[14]:


# Function #13
# Function to create a dictionary with a TOP_n list from e.g. TOP20 List for e.g. dataframe selection

def TOP_n_from(TOP_n_dict, feature_top_list, n):
    
    # first copy the given list to leave it "untouched"
    TOP_n = feature_top_list.copy()

    # reverse the Features from Feature Selection Results list
    TOP_n_rev = list(reversed(TOP_n))

    # index how many or which features should be removed
    to_be_removed = TOP_n_rev[0:20-n]

    for tbr in to_be_removed:
        TOP_n.remove(tbr)

    # adding Rank to create Tuple List
    TOP_n_rank = list(zip(TOP_n, list(range(1, n+1))))

    # updating given dictionary
    TOP_n_dict.update({"TOP%d" % n:TOP_n_rank})

