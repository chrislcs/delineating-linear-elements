# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import pandas as pd
import time
import os
from lin_delineation import (segment_object, VegetationObject,
                             merge_objects, export_to_shapefile)
from data_preprocessing import downsample

# %%
cwd = os.getcwd()
CloudCompare_path = "D:/Chris/CloudCompare/CloudCompare.exe"
point_cloud = pd.read_csv('../Data/veg_classification.csv')

# Downsample low vegetation points
low_veg_path = '%s\\Data\\low_veg_2D.csv' % os.path.dirname(cwd)
point_cloud.loc[point_cloud['class'] == 1].to_csv(low_veg_path,
                                                  columns=['X', 'Y'],
                                                  index=False)
low_veg_path = downsample(low_veg_path, 1.0, CloudCompare_path)

# Downsample tree points
trees_path = '%s\\Data\\trees_2D.csv' % os.path.dirname(cwd)
point_cloud.loc[point_cloud['class'] == 2].to_csv(trees_path,
                                                  columns=['X', 'Y'],
                                                  index=False)
trees_path = downsample(trees_path, 2.0, CloudCompare_path)

# %% Load point cloud data
print 'Loading tree points..'
point_cloud = pd.read_csv('../Data/trees_2D_sub_2_0.csv',
                          delimiter=',', names=['X', 'Y', 'Z'], header=1)
point_cloud.drop('Z', axis=1, inplace=True)

points = point_cloud.as_matrix()
global_shift_t = (min(points[:, 0]), min(points[:, 1]))
points[:, 0] -= global_shift_t[0]
points[:, 1] -= global_shift_t[1]

# %% Segment the points into rectangular objects
min_size = 5
rect_th = 0.55
alpha = 0.4
k_init = 20
max_dist_init = 15.0
k = 8
max_dist = 5.0

print 'Growing rectangular regions..'
t = time.time()
segments = segment_object(points, min_size, rect_th, alpha=alpha,
                          k_init=k_init, max_dist_init=max_dist_init,
                          k=k, max_dist=max_dist)

linear_elements_t = []
for s in segments:
    l = VegetationObject(s, alpha)
    linear_elements_t.append(l)
print 'Done! Time elapsed: %.2f' % (time.time() - t)

# %% Merge neighbouring elongated objects if pointing in the same direction
print 'Merging objects..'
t = time.time()
max_dist = 5.0
max_dir_dif = math.radians(30)
min_elong = 1.3
max_c_dir_dif = math.radians(30)
max_width = 60
linear_elements_t_m = merge_objects(linear_elements_t, max_dist, max_dir_dif,
                                  max_c_dir_dif, min_elong, max_width)
print 'Done! Time elapsed: %.2f' % (time.time() - t)

# %% Export to shapefile
print 'Exporting to shapefile..'
filename = '../Data/linear_elements_t.shp'
epsg = 28992
export_to_shapefile(filename, linear_elements_t_m, epsg, global_shift_t)

# %% Load point cloud data
print 'Loading low vegetation points..'
point_cloud = pd.read_csv('../Data/low_veg_2D_sub_1_0.csv',
                          delimiter=',', names=['X', 'Y', 'Z'], header=1)
point_cloud.drop('Z', axis=1, inplace=True)

points = point_cloud.as_matrix()
global_shift_v = (min(points[:, 0]), min(points[:, 1]))
points[:, 0] -= global_shift_v[0]
points[:, 1] -= global_shift_v[1]

# %% Segment the points into rectangular objects
print 'Growing rectangular regions..'
t = time.time()
segments = segment_object(points, min_size, rect_th, alpha=alpha,
                          k_init=k_init, max_dist_init=max_dist_init,
                          k=k, max_dist=max_dist)

linear_elements_lv = []
for s in segments:
    l = VegetationObject(s, alpha)
    linear_elements_lv.append(l)
print 'Time elapsed: %.2f' % (time.time() - t)

# %% Merge neighbouring elongated regions if pointing in the same direction
print 'Merging objects..'
t = time.time()
linear_elements_lv = merge_objects(linear_elements_lv, max_dist, max_dir_dif,
                                   max_c_dir_dif, min_elong, max_width)
print 'Time elapsed: %.2f' % (time.time() - t)

# %% Export to shapefile
filename = '../Data/linear_elements_lv.shp'

export_to_shapefile(filename, linear_elements_lv, epsg, global_shift_v)
