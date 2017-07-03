# -*- coding: utf-8 -*-
"""
Computes point neighbourshood parameters and removes irrelevant points.

@author: Chris Lucas
"""

import pandas as pd
import time
import os
from scipy.spatial import cKDTree
from data_preprocessing import las_to_csv, downsample
from par_computation import neighbourhood_features

# %% file paths
las_path = "/Data/ResearchArea.las"
las2txt_path = "D:/MasterThesis/LAStools/LAStools/bin/las2txt.exe"
CloudCompare_path = "C:/Program Files/CloudCompare/CloudCompare.exe"

# %% Prepare data and load into python
# downsample point cloud and convert to csv
las = downsample(las_path, 0.3, tool_path=CloudCompare_path)
csv_path = las_to_csv(las, method='las2txt', tool_path=las2txt_path)

# Load the csv point cloud file
print "Loading point cloud csv file using pandas.."
point_cloud = pd.read_csv(csv_path, delimiter=';', header=None,
                          names=['X', 'Y', 'Z', 'intensity',
                                 'return_number', 'number_of_returns'])

points = point_cloud.as_matrix(columns=['X', 'Y', 'Z'])

# %% Compute nearest neighbours
print "Computing nearest neighbours.."
neighbours = [50]
kdtree = cKDTree(points)
distances, point_neighbours = kdtree.query(points, max(neighbours))
print "Done!"

# %% Compute point parameters
features = ['delta_z', 'std_z', 'radius', 'density', 'norm_z',
            'linearity', 'planarity', 'sphericity', 'omnivariance',
            'anisotropy', 'eigenentropy', 'sum_eigenvalues',
            'curvature']
feature_values = {}
for k in neighbours:
    print "Computing covariance features.."
    t = time.time()
    fv = neighbourhood_features(points, point_neighbours[:, :k],
                                features, distances[:, :k])
    print "Done! Runtime: %s" % str(time.time() - t)
    feature_values[k] = fv

for k in neighbours:
    for i, f in enumerate(features):
        key = f + '_' + str(k)
        point_cloud[key] = pd.Series(feature_values[k][:, i])

# %% Trim the data by deleting all non scatter points from the point cloud
print "Trimming data.."
point_cloud.query('sphericity_50 > 0.05 & planarity_50 < 0.7', inplace=True)
point_cloud.reset_index(drop=True, inplace=True)
print "Done!"

# %% Compute normalized return number
point_cloud['norm_returns'] = (point_cloud['return_number'] /
                               point_cloud['number_of_returns'])

# %% Output data
las_path_root = os.path.splitext(las_path)[0]
out_filename = '%s_params.csv' % (las_path_root)
point_cloud.to_csv(out_filename, index=False)
