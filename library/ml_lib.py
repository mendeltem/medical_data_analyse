# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import h5py
import os
import json
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import missingno as mno
import seaborn as sns
from sklearn import linear_model
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
from sklearn import svm, datasets
from matplotlib.pyplot import figure
from scipy.stats import ttest_ind
import copy
import random
import shap
import re


def calc_shap(shap_values, X, expected_value):
    
    explanation = copy.copy(shap.Explanation(shap_values,
                 base_values=np.array(np.full((len(X), ), np.array(expected_value))),
                 data=X.values,
                 feature_names=list(X.columns)))
    
    return explanation

def get_importance(class_idx, beta, feature_names, intercepts=None):
    """
    Retrive and sort abs magnitude of coefficients from model.
    """

    # sort the absolute value of model coef from largest to smallest
    srt_beta_k = np.argsort(np.abs(beta[class_idx, :]))[::-1]
    feat_names = [feature_names[idx] for idx in srt_beta_k]
    feat_imp = beta[class_idx, srt_beta_k]
    # include bias among feat importances
    if intercepts is not None:
        intercept = intercepts[class_idx]
        bias_idx = len(feat_imp) - np.searchsorted(np.abs(feat_imp)[::-1], np.abs(intercept) )
        feat_imp = np.insert(feat_imp, bias_idx, intercept.item(), )
        intercept_idx = np.where(feat_imp == intercept)[0][0]
        feat_names.insert(intercept_idx, 'bias')

    return feat_imp, feat_names



def reorder_feats(vals_and_names, src_vals_and_names):
    """Given a two tuples, each containing a list of ranked feature
    shap values and the corresponding feature names, the function
    reorders the values in vals according to the order specified in
    the list of names contained in src_vals_and_names.
    """

    _, src_names = src_vals_and_names
    vals, names = vals_and_names
    reordered = np.zeros_like(vals)

    for i, name in enumerate(src_names):
        alt_idx = names.index(name)
        reordered[i] = vals[alt_idx]

    return reordered, src_names

def compare_avg_mag_shap(class_idx, comparisons, baseline, **kwargs):
    """
    Given a list of tuples, baseline, containing the feature values and a list with feature names
    for each class and, comparisons, a list of lists with tuples with the same structure , the
    function reorders the values of the features in comparisons entries according to the order
    of the feature names provided in the baseline entries and displays the feature values for comparison.
    """

    methods = kwargs.get("methods", [f"method_{i}" for i in range(len(comparisons) + 1)])

    n_features = len(baseline[class_idx][0])

    # bar settings
    bar_width = kwargs.get("bar_width", 0.05)
    bar_space = kwargs.get("bar_space", 2)

    # x axis
    x_low = kwargs.get("x_low", 0.0)
    x_high = kwargs.get("x_high", 1.0)
    x_step = kwargs.get("x_step", 0.05)
    x_ticks = np.round(np.arange(x_low, x_high + x_step, x_step), 3)

    # y axis (these are the y coordinate of start and end of each group
    # of bars)
    start_y_pos = np.array(np.arange(0, n_features))*bar_space
    end_y_pos = start_y_pos + bar_width*len(methods)
    y_ticks = 0.5*(start_y_pos + end_y_pos)

    # figure
    fig_x = kwargs.get("fig_x", 10)
    fig_y = kwargs.get("fig_y", 7)

    # fontsizes
    title_font = kwargs.get("title_fontsize", 20)
    legend_font = kwargs.get("legend_fontsize", 20)
    tick_labels_font = kwargs.get("tick_labels_fontsize", 20)
    axes_label_fontsize = kwargs.get("axes_label_fontsize", 10)

    # labels
    title = kwargs.get("title", None)
    ylabel = kwargs.get("ylabel", None)
    xlabel = kwargs.get("xlabel", None)

    # process input data
    methods = list(reversed(methods))
    base_vals = baseline[class_idx][0]
    ordering = baseline[class_idx][1]
    comp_vals = []

    # reorder the features so that they match the order of the baseline (ordering)
    for comparison in comparisons:
        vals, ord_ = reorder_feats(comparison[class_idx], baseline[class_idx])
        comp_vals.append(vals)
        assert ord_ is ordering

    all_vals = [base_vals] + comp_vals
    data = dict(zip(methods, all_vals))
    df = pd.DataFrame(data=data, index=ordering)

    # plotting logic
    fig, ax = plt.subplots(figsize=(fig_x, fig_y))

    for i, col in enumerate(df.columns):
        values = list(df[col])
        y_pos = [y + bar_width*i for y  in start_y_pos]
        ax.barh(y_pos, list(values), bar_width, label=col)

    # add ticks, legend and labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], rotation=45, fontsize=tick_labels_font)
    ax.set_xlabel(xlabel, fontsize=axes_label_fontsize)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(ordering, fontsize=tick_labels_font)
    ax.set_ylabel(ylabel, fontsize=axes_label_fontsize)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.legend(fontsize=legend_font)

    plt.grid(True)
    plt.title(title, fontsize=title_font)

    return ax, fig, df

def _flatten(X):
    img_size = X[0].shape
    X = X.reshape([X.shape[0], np.prod(img_size)])
    return(X)


def split_set(X, y, fraction, random_state=0):
    """
    Given a set X, associated labels y, splits a fraction y from X.
    """
    _, X_split, _, y_split = train_test_split(X,
                                              y,
                                              test_size=fraction,
                                              random_state=random_state,
                                             )
    print("Number of records: {}".format(X_split.shape[0]))
    print("Number of class {}: {}".format(0, len(y_split) - y_split.sum()))
    print("Number of class {}: {}".format(1, y_split.sum()))

    return X_split, y_split

def plot_importance(feat_imp, feat_names, class_idx, **kwargs):
    """
    Create a horizontal barchart of feature effects, sorted by their magnitude.
    """

    left_x, right_x = kwargs.get("left_x"), kwargs.get("right_x")
    eps_factor = kwargs.get("eps_factor", 4.5)

    fig, ax = plt.subplots(figsize=(20, 10))
    y_pos = np.arange(len(feat_imp))
    ax.barh(y_pos, feat_imp)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=9)
    ax.invert_yaxis()                  # labels read top-to-bottom
    ax.set_xlabel(f'Feature effects for class {class_idx}', fontsize=9)
    ax.set_xlim(left=left_x, right=right_x)

    for i, v in enumerate(feat_imp):
        eps = 0.03
        if v < 0:
            eps = -eps_factor*eps
        ax.text(v + eps, i + .25, str(round(v, 3)))

    return ax, fig

def permute_columns(X, feat_names, perm_feat_names):
    """
    Permutes the original dataset so that its columns
    (ordered according to feat_names) have the order
    of the variables after transformation with the
    sklearn preprocessing pipeline (perm_feat_names).
    """

    perm_X = np.zeros_like(X)
    perm = []
    for i, feat_name in enumerate(perm_feat_names):
        feat_idx = feat_names.index(feat_name)
        perm_X[:, i] = X[:, feat_idx]
        perm.append(feat_idx)
    return perm_X, perm

def create_category_map(X_data , limit = 10):
    """creates category map from a DataFrame and lists of columnnumbers
    
    
    input : DataFrame
    return: Dictionary  :  categorymap 
            List        :  column number of category columns
            List        :  column number of numerical columns
    """
    
    category_map = {}

    for i,col_name in enumerate(X_data.columns):
        unique_len = len(X_data[col_name].dropna().unique())

        if unique_len == 2 or unique_len < limit:
            print(col_name," : ", unique_len)

            category_map[i] = {}
            category_map[i] = []

            if np.isnan(list(X_data[col_name].unique())).any():
                category_map[i].append(np.nan)

            cat_values = list(X_data[col_name].dropna().unique())
            cat_values.sort()

            for cat_value in cat_values:
                category_map[i].append(cat_value)
                        
    ordinal_features          = [x for x in range(len(X_data.columns)) if x not in list(category_map.keys())]
    categorical_features      = list(category_map.keys())
                        
    return category_map,categorical_features,ordinal_features,list(X_data.columns)


def get_featnames(preprocessor, 
                  categorical_features,
                  ordinal_features,
                  feature_names):
    fts = [feature_names[x] for x in categorical_features]
    ohe = preprocessor.transformers_[1][1].named_steps['onehot']
    cat_enc_feat_names = ohe.get_feature_names(fts)
    feat_enc_dim = [len(cat_enc) - 1 for cat_enc in ohe.categories_]
    start=len(ordinal_features)
    cat_feat_start = [start]
    for dim in feat_enc_dim[:-1]:
        cat_feat_start.append(dim + cat_feat_start[-1])
    numerical_feats_idx  = preprocessor.transformers_[0][2]
    categorical_feats_idx  = preprocessor.transformers_[1][2]
    scaler = preprocessor.transformers_[0][1].named_steps['scaler']
    num_feats_names = [feature_names[i] for i in numerical_feats_idx]
    cat_feats_names = [feature_names[i] for i in categorical_feats_idx]
    perm_feat_names = num_feats_names + cat_feats_names
    
    return num_feats_names,cat_feats_names,perm_feat_names,cat_feat_start,feat_enc_dim,fts