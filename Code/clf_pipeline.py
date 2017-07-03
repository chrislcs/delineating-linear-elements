# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import pandas as pd
import numpy as np
import time
from clf_preprocessing import merge_dataframes, correlated_features
from clf_classifiers import BalancedRandomForest, classify_vegetation
from clf_assessment import (grid_search, cross_validation,
                            mean_decrease_impurity)

# %% Load ground truth data
print "loading data.."
classes = ['veg', 'non_veg']
veg_pc = pd.read_csv('../Data/C_39CN1_veg.csv', delimiter=';', header=0)
non_veg_pc = pd.read_csv('../Data/C_39CN1_nonveg.csv', delimiter=';', header=0)
data = merge_dataframes({'veg': veg_pc, 'non_veg': non_veg_pc}, 'class')
data.rename(columns={'//X': 'X'}, inplace=True)
data.rename(columns=lambda x: x.replace(',', '_'), inplace=True)
del veg_pc, non_veg_pc
class_cat, class_indexer = pd.factorize(data['class'])
data['class_cat'] = class_cat
del class_cat
class_dtype = '|S%s' % max([len(x) for x in classes])
del x
class_indexer = np.array(class_indexer, dtype=class_dtype)

# %% Define the feature space
features = data.columns.drop(['class', 'class_cat', 'X', 'Y', 'Z',
                              'norm_x_50', 'norm_y_50', 'return_number'],
                             'ignore')
features = features.drop(correlated_features(data, features, corr_th=0.98))

# %% GridSearch (Cross Validated)
param_dict = {'min_samples_leaf': [5, 10],
              'min_samples_split': [5, 10],
              'ratio': [0.15, 0.1, 0.05]}

gs_scores, param_grid = grid_search(data, features, 'class_cat', param_dict)

# %% Cross Validation
cv_scores, conf_matrices = cross_validation(data, features, 'class_cat')

# %% Load all data
point_cloud = pd.read_csv("../Data/C_39CN1_ResearchArea_params.csv",
                          delimiter=',', header=0)


# %% Create final classifier
clf = BalancedRandomForest(n_estimators=1000, min_samples_leaf=5,
                           min_samples_split=5, ratio=0.2)
clf.fit(data[features], data['class_cat'])

# %% Assess feature importances
fi_scores = mean_decrease_impurity(clf, features)

# %% Classify vegetation / non-vegetation
classification = []
parts = 8
part = len(point_cloud)/parts
for i in xrange(parts):
    if i == parts-1:
        temp_pc = point_cloud.loc[point_cloud.index[i*part:]]
    else:
        temp_pc = point_cloud.loc[point_cloud.index[i*part:(i+1)*part]]
    preds = clf.predict(temp_pc[features])
    classification.extend(list(preds))

point_cloud['class'] = classification

# %% Classify trees / low vegetation
t = time.time()
points = point_cloud.loc[point_cloud['class'] == 1].as_matrix(columns=['X', 'Y', 'Z'])
radius = 2.0
tree_th = 4.0
classification = classify_vegetation(points, radius, tree_th)
point_cloud['veg_class'] = 'non_veg'
point_cloud.loc[point_cloud['class'] == 1, 'veg_class'] = classification
point_cloud['class'], _ = pd.factorize(point_cloud['veg_class'])
print "Done! Runtime: %s" % str(time.time() - t)

# %% Save results
point_cloud.to_csv('../Data/veg_classification.csv',
                   columns=['X', 'Y', 'Z', 'class'], index=False)
