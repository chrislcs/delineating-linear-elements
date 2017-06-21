# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from imblearn.metrics import geometric_mean_score
from clf_classifiers import BalancedRandomForest


def grid_search(df, features, class_column, param_dict, n_folds=3):
    """
    Perform a cross validated grid search to look for the best parameter
    settings.

    Parameters
    ----------
    df : DataFrame
        The feature values and the corrisponding classes.
    features : list of strings
        The names of the features (column names in the dataframe).
    class_column : string
        The name of the column with the class labels (the labels should be
        integers from 0 to n for n classes)
    param_dict : dictionary
        The parameters and the values that need to be checked.
    n_folds : int
        The amount of folds to perform the cross validation.

    Returns
    -------
    gs_scores : DataFrane
        A table with the Matthews Correlation Coefficient, the Area under
        ROC-curve, and geometric mean values of all the different combinations
        of parameter settings.
    param_grid : sklearn ParameterGrid
        All the different combinations of the parameters.
    """
    skf = StratifiedKFold(n_splits=n_folds)

    param_grid = ParameterGrid(param_dict)
    n_params = len(param_grid)
    gs_scores = pd.DataFrame(np.zeros((n_params, 3)),
                             columns=['mcc', 'roc_auc', 'gmean'])

    n = 0
    n_total = n_params * n_folds
    for train_index, test_index in skf.split(df[features], df[class_column]):
        for i in xrange(n_params):
            parameters = param_grid[i]
            train_data = df.iloc[train_index]
            test_data = df.iloc[test_index]

            clf = BalancedRandomForest(n_estimators=100, **parameters)
            clf.fit(train_data[features], train_data[class_column])

            preds = clf.predict(test_data[features])
            probas = clf.predict_proba(test_data[features])

            mcc = matthews_corrcoef(test_data[class_column], preds)
            roc_auc = roc_auc_score(test_data[class_column], probas[:, 1])
            gmean = geometric_mean_score(test_data[class_column], preds)
            gs_scores.loc[i, 'mcc'] += mcc
            gs_scores.loc[i, 'roc_auc'] += roc_auc
            gs_scores.loc[i, 'gmean'] += gmean

            n += 1
            print "Done %d of %d.." % (n, n_total)

    gs_scores['mcc'] /= n_folds
    gs_scores['roc_auc'] /= n_folds
    gs_scores['gmean'] /= n_folds

    return gs_scores, param_grid


def cross_validation(df, features, class_column, n_folds=10):
    """
    Perform a cross validation to evaluate the performance of a classification
    method.

    Parameters
    ----------
    df : DataFrame
        The feature values and the corrisponding classes.
    features : list of strings
        The names of the features (column names in the dataframe).
    class_column : string
        The name of the column with the class labels (the labels should be
        integers from 0 to n for n classes)
    n_folds : int
        The amount of folds to perform the cross validation.

    Returns
    -------
    cv_scores : DataFrane
        A table with the Matthews Correlation Coefficient, the Area under
        ROC-curve, and geometric mean values of all the folds and the
        average of those in the last row.
    confusion_matrices : list
        The confusion matrices of each fold.
    """
    skf = StratifiedKFold(n_splits=n_folds)

    cv_scores = pd.DataFrame(np.zeros((n_folds + 1, 3)),
                             columns=['mcc', 'roc_auc', 'gmean'])
    confusion_matrices = []

    i = 0
    for train_index, test_index in skf.split(df[features], df[class_column]):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        clf = BalancedRandomForest(n_estimators=1000, min_samples_leaf=5,
                                   min_samples_split=5, ratio=0.2)
        clf.fit(train_data[features], train_data[class_column])

        preds = clf.predict(test_data[features])
        probas = clf.predict_proba(test_data[features])

        mcc = matthews_corrcoef(test_data[class_column], preds)
        roc_auc = roc_auc_score(test_data[class_column], probas[:, 1])
        gmean = geometric_mean_score(test_data[class_column], preds)
        cv_scores.loc[i, 'mcc'] = mcc
        cv_scores.loc[i, 'roc_auc'] = roc_auc
        cv_scores.loc[i, 'gmean'] = gmean

        df_confusion = pd.crosstab(test_data[class_column], preds,
                                   rownames=['Actual'],
                                   colnames=['Predicted'],
                                   margins=True)
        confusion_matrices.append(df_confusion)

        i += 1
        print "Done %d of %d.." % (i, n_folds)

    cv_scores.loc[i, 'mcc'] = np.average(cv_scores['mcc'][:i])
    cv_scores.loc[i, 'roc_auc'] = np.average(cv_scores['roc_auc'][:i])
    cv_scores.loc[i, 'gmean'] = np.average(cv_scores['gmean'][:i])

    return cv_scores, confusion_matrices


def mean_decrease_impurity(clf, features, plot=False):
    """
    Assess the importance of the features by analysing the mean decrease in
    gini impurity for each feature.

    Parameters
    ----------
    clf : Classifier
        A classifier with a feature_importance_ attribute
    features : list of strings
        The names of the features used when fitting the classifier.
    plot : bool
        Plot the feature importances using a bar diagram.

    Returns
    -------
    scores : list of tuples
        The features and the corrisponding mean decrease in gini impurity.
    """
    scores = sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_),
                        features), reverse=True)

    if plot:
        widths, names = zip(*scores)
        xs = range(len(names))
        plt.figure()
        plt.barh(xs, widths[::-1], height=0.4, tick_label=names[::-1])
        plt.tight_layout()
        plt.show()

    return scores
