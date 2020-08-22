#/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np
import statsmodels.api as sm
import operator
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re

def calc_kdes(training_data, classes, bw_method = 'cv_ml'):
    '''
    kernel density estimation for each class
    '''
    kdes = []
    for cl in classes:
        values = training_data[training_data[:, -1] == cl][:, : -1]
        class_kde = sm.nonparametric.KDEMultivariate(values,  var_type='c' * (training_data.shape[1] - 1), bw=bw_method)
        kdes.append(class_kde)
    return kdes

def predict_pdf(kdes, arr, classes):
    '''
    predict the label/probability for test data
    '''
    p_vals = []
    for k in kdes:
        p_vals.append(k.cdf(arr))
    idx = np.argmax(p_vals)
    membership = np.max(p_vals)
    return (membership, classes[idx], p_vals)

def plot_confusion_matrix(true_labels, predict_labels, labels, normalize=False):
    '''
    plot confusion matrix
    '''  
    cm = confusion_matrix(true_labels, predict_labels, labels = labels)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    sns.set(font_scale = 1.5)
    ax = sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels = labels, yticklabels = labels, fmt='g')
    ax.set_ylim(len(ax.get_yticklabels()), -0.0)
    plt.xlabel("predicted label", fontsize=20)
    plt.ylabel("true label", fontsize=20)
    return cm

def loo_cv(X, classes, bws):
    '''
    run leave-one-out cross-validation
    '''
    loo = LeaveOneOut()
    pred_cl = []
    pred_prob = []
    for train_index, test_index in loo.split(X):
        start = time.time()
        X_train, X_test = X[train_index], X[test_index]
        kdes = calc_kdes(X_train, classes, bw_method = bws)
        membership, cl, u = predict_pdf(kdes, X_test[0][:-1], classes)
        pred_cl.append(cl)
        pred_prob.append(membership)
        end = time.time()
        print('Time elapsed: {0}: validate data point {1} ==============> predicted: {2} actual: {3} maxProb: {4}'
              .format(end - start, test_index, cl, X_test[0][-1], membership))
    return pred_cl, pred_prob

def plot_class_result(data, col, conditions, titles, label_dict):
    '''
    plot classification results
    '''
    arr = []
    label_idx = label_dict.keys()
    for i in label_idx:
        tmp = []
        for c in conditions:
            tmp.append(data[(data[col] == i) & (data['condition'] == c)].shape[0])
        arr.append(tmp)
    df = pd.DataFrame(arr, 
                  columns = titles,
                 index = [label_dict[i] for i in label_idx])
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    ax = sns.heatmap(df, annot = True, fmt= 'g', cmap="YlGnBu")
    ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation = 45,
            horizontalalignment='left')

def plot_trends(training_data, test_data, conditions, titles):
    '''
    plot temporal trends
    '''
    training = training_data[list(filter(lambda x: x.startswith('min_'), training_data.columns)) + ['label']]
    training = training.rename(columns = dict(zip(training.columns[:-1],np.arange(len(training.columns[:-1])))))
    training['Category'] = ['Training Data'] * training.shape[0]
    arr = []
    for i in range(len(conditions)):
        tmp = test_data[(test_data['condition'] == conditions[i])][list(filter(lambda x: re.match(r'[0-9]*`', x), test_data.columns))+ ['label']]
        tmp['Category'] = [titles[i]] * tmp.shape[0]
        arr += tmp.values.tolist()
    testing = pd.DataFrame(arr, columns = training.columns)
    total = pd.concat([training, testing])
    plot_df = total.melt(id_vars = ['Category', 'label'], 
                         var_name = 'Stimulation time [min]', 
                         value_name = 'Log2 (SILAC-ratio)')
    sns.set(font_scale = 1.8)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    ax = sns.relplot(x = 'Stimulation time [min]', y = 'Log2 (SILAC-ratio)', 
                     hue = 'Category', col='label', col_wrap=3,
                     kind = 'line', data= plot_df)
    return ax