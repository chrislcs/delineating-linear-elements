# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
import pandas as pd


def merge_dataframes(dfs, key_field_name):
    """
    Merges dataframes containing data of one class into one dataframe with
    the class in a column.

    Parameters
    ----------
    dfs : dict of DataFrames
        Dictionary with the class as key and the value as the dataframes
        to be merged.

    Returns
    -------
    df : DataFrame
        The merged dataframe.
    """
    df = pd.DataFrame()
    for k, v in dfs.iteritems():
        v[key_field_name] = k
        df = df.append(v)
    return df


def correlated_features(df, features, corr_th=0.98):
    """
    Determines highly correlated features which can consequently be dropped.

    Parameters
    ----------
    df : DataFrame
        The feature values.
    features : list of strings
        The names of the features (column names in the dataframe)
        to be checked.
    corr_th : float
        The correlation coefficient threshold to determine what is highly
        correlated.

    Returns
    -------
    drops : list fo strings
        The names of the features which can be dropped.
    """
    df_corr = df[features].astype(np.float64).corr(method='pearson')
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr

    drops = []
    for col in df_corr.columns.values:
        if not np.in1d([col], drops):
            corr = df_corr[abs(df_corr[col]) > corr_th].index
            drops = np.union1d(drops, corr)

    return drops
