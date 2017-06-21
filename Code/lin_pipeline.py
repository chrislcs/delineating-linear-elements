# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import pandas as pd
import time
from lin_delineation import (segment_object, VegetationObject,
                             merge_objects, export_to_shapefile)

# %% Load point cloud data
point_cloud = pd.read_csv('trees_2D_ds2.csv', delimiter=';', names=['X', 'Y', 'Z'])
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

t = time.time()
segments = segment_object(points, min_size, rect_th, alpha=alpha,
                          k_init=k_init, max_dist_init=max_dist_init,
                          k=k, max_dist=max_dist)
print 'Time elapsed: %.2f' % (time.time() - t)

linear_elements_t = []
for s in segments:
    l = VegetationObject(s, alpha)
    linear_elements_t.append(l)

# %% Merge neighbouring elongated regions if pointing in the same direction
max_dist = 5.0
max_dir_dif = math.radians(30)
min_elong = 1.5
max_c_dir_dif = math.radians(20)
max_width = 60
linear_elements_t = merge_objects(linear_elements_t, max_dist, max_dir_dif,
                                  max_c_dir_dif, min_elong, max_width)

# %% Export to shapefile
filename = 'linear_elements_t.shp'
epsg = 28992
export_to_shapefile(filename, linear_elements_t, epsg, global_shift_t)

# %% Load point cloud data
point_cloud = pd.read_csv('low_veg_2D_ds.csv', delimiter=';',
                          names=['X', 'Y', 'Z'])
point_cloud.drop('Z', axis=1, inplace=True)

points = point_cloud.as_matrix()
global_shift_v = (min(points[:, 0]), min(points[:, 1]))
points[:, 0] -= global_shift_v[0]
points[:, 1] -= global_shift_v[1]

# %% Segment the points into rectangular objects
t = time.time()
segments = segment_object(points, min_size, rect_th, alpha=alpha,
                          k_init=k_init, max_dist_init=max_dist_init,
                          k=k, max_dist=max_dist)
print 'Time elapsed: %.2f' % (time.time() - t)

linear_elements_lv = []
for s in segments:
    l = VegetationObject(s, alpha)
    linear_elements_lv.append(l)

# %% Merge neighbouring elongated regions if pointing in the same direction
linear_elements_lv = merge_objects(linear_elements_lv, max_dist, max_dir_dif,
                                   max_c_dir_dif, min_elong, max_width)

# %% Export to shapefile
filename = 'linear_elements_lv.shp'

export_to_shapefile(filename, linear_elements_lv, epsg, global_shift_v)
