# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np
import fiona
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay, cKDTree
from fiona.crs import from_epsg


class BoundingBox(object):
    """
    Compute the minimum oriented bounding box around a set of points
    by rotating caliphers.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    hull : (Mx2) array
        The indices of the points of the edges which form the
        convex hull of the points. (as computed by the Delauney triangulation
        of SciPy)

    Attributes
    ----------
    hull : (Mx2x2) array
        The coordinates of the points of the edges which form the
        convex hull of the points.
    hull_points : (Mx2) array
        The points of the convex hull.
    corner_points : (4x1) array
        The coordinates of the corner points of the bounding box.
    length : float
        The length of the long side of the bounding box.
    width : float
        The length of the short side of the bounding box.
    area : area
        The area of the bounding box.
    """
    def __init__(self, points, hull):
        self.hull = points[hull]
        self._to_unique_points()
        angles = self._compute_edge_angles()
        self._compute_bbox(angles)

    @staticmethod
    def _rotate(points, angle):
        """
        Rotate points in a coordinate system using a rotation matrix based on
        an angel.

        Parameters
        ----------
        points : (Mx2) array
            The coordinates of the points.
        angle : float
            The angle by which the points will be rotated (in radians).

        Returns
        -------
        points_rotated : (Mx2) array
            The coordinates of the rotated points.
        """
        # Compute rotation matrix
        rot_matrix = np.array(((math.cos(angle), -math.sin(angle)),
                               (math.sin(angle), math.cos(angle))))
        # Apply rotation matrix to the points
        points_rotated = np.dot(points, rot_matrix)
        return np.array(points_rotated)

    def _to_unique_points(self):
        """
        Extracts the unique points present in the convex hull edges.
        """
        hull_points = []
        for s in self.hull:
            for p in s:
                hull_points.append(p)
        hull_points = np.array(hull_points)
        self.hull_points = np.unique(hull_points.view(np.dtype((np.void,
                   hull_points.dtype.itemsize*
                   hull_points.shape[1])))).view(hull_points.dtype).reshape(-1,
                                                         hull_points.shape[1])

    def _compute_edge_angles(self):
        """
        Compute the angles between the edges of the convex hull and the x-axis.

        Returns
        -------
        edge_angles : (Mx1) array
            The angles between the edges and the x-axis.
        """
        edges_count = len(self.hull)
        edge_angles = np.zeros(edges_count)
        for i in xrange(edges_count):
            edge_x = self.hull[i][1][0] - self.hull[i][0][0]
            edge_y = self.hull[i][1][1] - self.hull[i][0][1]
            edge_angles[i] = math.atan2(edge_y, edge_x)
        return np.unique(edge_angles)

    def _compute_bbox(self, angles):
        """
        Compute the oriented minimum bounding box.

        Parameters
        ----------
        angles : (Mx1) array-like
            The angles the edges of the convex hull and the x-axis.
        """
        # Start with basic rectangle around the points
        min_x = min(self.hull_points[:, 0])
        max_x = max(self.hull_points[:, 0])
        min_y = min(self.hull_points[:, 1])
        max_y = max(self.hull_points[:, 1])
        width = max_x - min_x
        height = max_y - min_y
        min_bbox = {'angle': 0, 'minmax': (min_x, max_x, min_y, max_y),
                    'width': width, 'height': height, 'area': width * height}

        for a in angles:
            # Rotate the points and compute the new bounding box
            rotated_points = self._rotate(self.hull_points, a)
            min_x = min(rotated_points[:, 0])
            max_x = max(rotated_points[:, 0])
            min_y = min(rotated_points[:, 1])
            max_y = max(rotated_points[:, 1])
            width = max_x - min_x
            height = max_y - min_y
            area = width * height

            # Save if the new bounding box is smaller than the current smallest
            if area < min_bbox['area']:
                min_bbox = {'angle': a, 'minmax': (min_x, max_x, min_y, max_y),
                            'width': width, 'height': height,
                            'area': area}

        # Extract the rotated corner points of the minimum bounding box
        c1 = (min_bbox['minmax'][0], min_bbox['minmax'][2])
        c2 = (min_bbox['minmax'][0], min_bbox['minmax'][3])
        c3 = (min_bbox['minmax'][1], min_bbox['minmax'][3])
        c4 = (min_bbox['minmax'][1], min_bbox['minmax'][2])
        rotated_corner_points = [c1, c2, c3, c4]

        # Set the length, width and area of the minimum bounding box
        self.length = max(min_bbox['width'], min_bbox['height'])
        self.width = min(min_bbox['width'], min_bbox['height'])
        self.area = min_bbox['area']
        # Rotate the corner points back to the original system
        self.corner_points = np.array(self._rotate(rotated_corner_points,
                                                   2*np.pi-min_bbox['angle']))


class AlphaShape(object):
    """
    Compute the alpha shape (or concave hull) of points.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    alpha : float
        Influences the shape of the alpha shape. Higher values lead to more
        edges being deleted.
    inc : bool
        If True adding and deleting points is allowed (at the cost of some
        efficiency).

    Attributes
    ----------
    points : (Mx2) array
        The coordinates of the points used to compute the alpha shape.
    alpha : float
        The alpha used to compute the alpha shape.
    tri : SciPy Delaunay object
        The Delaunay triangulation of the points.
    edges : set
        The edges between the points resulting from the Delaunay
        triangulation.
    edge_points : list of arrays
        Each array contains the points which are connected by the edges
        resulting from the Delaunay triangulation.
    m : Shapely MultiLineString
        The Delaunay triangulation edges converted to a Shapely
        MultiLineString.
    triangles : array of Shapely polygons
        The Delaunay triangles converted to Shapely polygons.
    shape : Shapely polygon
        The resulting alpha shape of the to_shape method.

    Methods
    -------
    add_points(points)
        Adds points to the alpha shape. Can only be used if 'inc' has been set
        to True.
    remove_last_added()
        Removes the last added points by the 'add_points' method and
        recreates the alpha shape.
    """
    def __init__(self, points, alpha, inc=False):
        if len(points) < 4:
            raise ValueError('Not enough points to compute an alpha shape.')
        self.saved = None
        self.area = 0
        self.alpha = alpha
        self.points = points
        self.tri = Delaunay(points, incremental=inc)
        self.edges = set()
        self.edge_points = []
        self._add_edges(self.tri.vertices)

    @staticmethod
    def _find_outliers(data, m):
        """
        Find outliers by Median Absolute Deviation (MAD).

        Parameters
        ----------
        data : array
            Data to find outliers of.
        m : float or int
            Threshold value of the median deviation of a data entry
            to be considered an outlier.

        Returns
        -------
        outliers : array
            Boolean array where outliers are marked as False.
        """
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return s < m

    @staticmethod
    def remove_array_from_list(L, arr):
        """
        Removes an array from a list.

        Parameters
        ----------
        L : list
            A list with arrays of which one has to be removed.
        arr : array
            The array which needs to be removed from the list.
        """
        i = 0
        size = len(L)
        while i != size and not np.array_equal(L[i], arr):
            i += 1
        if i != size:
            L.pop(i)
        else:
            raise ValueError('array not found in list.')

    def _triangle_geometry(self, triangle):
        """
        Compute the area and circumradius of a triangle.

        Parameters
        ----------
        triangle : (1x3) array-like
            The indices of the points which form the triangle.

        Returns
        -------
        area : float
            The area of the triangle
        circum_r : float
            The circumradius of the triangle
        """
        pa = self.points[triangle[0]]
        pb = self.points[triangle[1]]
        pc = self.points[triangle[2]]
        # Lengths of sides of triangle
        a = math.hypot((pa[0]-pb[0]), (pa[1]-pb[1]))
        b = math.hypot((pb[0]-pc[0]), (pb[1]-pc[1]))
        c = math.hypot((pc[0]-pa[0]), (pc[1]-pa[1]))
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        if area != 0:
            circum_r = a*b*c/(4.0*area)
        else:
            circum_r = 0
        return area, circum_r

    def _add_edge(self, i, j):
        """
        Add an edge (line) between two points.

        Parameters
        ----------
        i : int
            Index of a point.
        j : int
            Index of a point.
        """
        if (i, j) in self.edges or (j, i) in self.edges:
            return
        self.edges.add((i, j))
        self.edge_points.append(self.points[[i, j]])

    def _add_edges(self, vertices):
        """
        Add the edges between the given vertices if the circumradius of the
        triangle is bigger than 1/alpha.

        Parameters
        ----------
        vertices : (Mx3) array
            Indices of the points forming the vertices of a triangulation.
        """
        for ia, ib, ic in vertices:
            area, circum_r = self._triangle_geometry((ia, ib, ic))
            if area != 0:
                if circum_r < 1.0/self.alpha:
                    self._add_edge(ia, ib)
                    self._add_edge(ib, ic)
                    self._add_edge(ic, ia)
                    self.area += area

    def _remove_edge(self, i, j):
        """
        Remove an edge (line) between two points.

        Parameters
        ----------
        i : int
            Index of a point.
        j : int
            Index of a point.
        """
        if (i, j) not in self.edges and (j, i) not in self.edges:
            return
        if (i, j) in self.edges:
            self.edges.remove((i, j))
            self.remove_array_from_list(self.edge_points, self.points[[i, j]])
        if (j, i) in self.edges:
            self.edges.remove((j, i))
            self.remove_array_from_list(self.edge_points, self.points[[j, i]])

    def _remove_edges(self, vertices):
        """
        Remove the edges between the given vertices if the circumradius of the
        triangle is bigger than 1/alpha.

        Parameters
        ----------
        vertices : (Mx3) array
            Indices of the points forming the vertices of a triangulation.
        """
        for ia, ib, ic in vertices:
            area, circum_r = self._triangle_geometry((ia, ib, ic))
            if area != 0:
                if circum_r < 1.0/self.alpha:
                    self.area -= area
                    self._remove_edge(ia, ib)
                    self._remove_edge(ib, ic)
                    self._remove_edge(ic, ia)

    def to_shape(self, remove_interior=None):
        """
        Convert the alpha shape to a Shapely polygon object.

        Parameters
        ----------
        remove_interior : float or int
            If remove_interior parameter is set it will attempt to remove a
            possible interior of a alpha shape by detecting outlying polygons
            in 'triangles', since interiors are often way bigger than the
            actual triangles. Outliers detected using a Median Absolute
            Deviation (MAD).
        """
        self.m = geometry.MultiLineString(self.edge_points)
        self.triangles = np.array(list(polygonize(self.m)))
        if remove_interior is not None:
            tri_area = np.array([t.area for t in self.triangles])
            outliers = self._find_outliers(tri_area, m=remove_interior)
            if type(outliers) is bool:
                outliers = np.array([outliers])
            self.triangles = self.triangles[outliers]

        self.shape = cascaded_union(self.triangles)

    def add_points(self, points):
        """
        Adds points to the alpha shape.

        Parameters
        ----------
        points : (Mx2) array
            The coordinates of the points.
        """
        # Save current properties
        self.saved = {'points': self.points.copy(),
                      'edge_points': self.edge_points[:],
                      'edges': self.edges.copy(),
                      'area': self.area}
        # Determine old and new triangles
        old = self.tri.vertices.copy()
        old_rows = old.view(np.dtype((np.void,
                                      old.dtype.itemsize*old.shape[1])))
        self.tri.add_points(points)
        new = self.tri.vertices
        new_rows = new.view(np.dtype((np.void,
                                      new.dtype.itemsize*new.shape[1])))
        old_tri = np.setdiff1d(old_rows,
                               new_rows).view(new.dtype).reshape(-1,
                                                                 new.shape[1])
        new_tri = np.setdiff1d(new_rows,
                               old_rows).view(new.dtype).reshape(-1,
                                                                 new.shape[1])
        # Remove old triangles and add new
        self._remove_edges(old_tri)
        self.points = np.vstack((self.points, points))
        self._add_edges(new_tri)

    def remove_last_added(self):
        """
        Removes the last added points.
        """
        if self.saved is None:
            raise RuntimeError('No points were added.')
        self.points = self.saved['points']
        self.edge_points = self.saved['edge_points']
        self.edges = self.saved['edges']
        self.area = self.saved['area']
        self.tri = Delaunay(self.points, incremental=True)
        self.saved = None


class VegetationObject(AlphaShape, BoundingBox):
    """
    A vegetation object.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points of the vegetation object.
    alpha : float
        Influences the shape of the alpha shape. Higher values lead to more
        edges being deleted.

    Attributes
    ----------
    area : float
        The area of the vegetation object.
    bbox_area : float
        The area of the minimum bounding box around the vegetation object.
    elongatedness : float
        The elongatedness of the vegetation object, defined as the length
        divided by the width.
    direction : float
        The angle between the long side of the object and the x-axis in
        radians.

    Methods
    -------
    shift_polygons(shift=(0,0))
        Returns the polygons shifted by the given values.
    export_to_shp(filename, epsg, shift=(0, 0))
        Export the object to a shapefile.
    """
    def __init__(self, points, alpha=0.4):
        self.points = points
        AlphaShape.__init__(self, points, alpha)
        self.to_shape(100)
        area = self.area
        BoundingBox.__init__(self, points, self.tri.convex_hull)
        self.bbox_area = self.area
        self.area = area
        self.elongatedness = self.length / self.width
        self.direction = self._compute_direction()

    def _compute_direction(self):
        """
        Compute the angle between the long side of the object and the x-axis.
        """
        idx = np.argmin(self.corner_points[:, 0])
        [c1, c2, c3, c4] = np.roll(self.corner_points, 4-idx, axis=0)
        dist12 = math.hypot((c1[0]-c2[0]), (c1[1]-c2[1]))
        dist23 = math.hypot((c2[0]-c3[0]), (c2[1]-c3[1]))
        if dist12 > dist23:
            x = c1[0] - c2[0]
            y = c1[1] - c2[1]
        else:
            x = c2[0] - c3[0]
            y = c2[1] - c3[1]
        return math.atan2(y, x)

    def shift_polygons(self, shift=(0, 0)):
        """
        Returns the polygons shifted by the given values.

        Parameters
        ----------
        shift : list or tuple
            The [X, Y] shift to be added to the coordinates.

        Returns
        -------
        polygons : List of polygons
            Polygons shifted by the given values
        """
        polygons = []
        if type(self.shape) is geometry.multipolygon.MultiPolygon:
            for geom in self.shape.geoms:
                ext_coords = np.array(geom.exterior)
                ext_coords += shift
                if len(geom.interiors) > 0:
                    int_rings = []
                    for i in geom.interiors:
                        int_coords = np.array(i)
                        int_coords += shift
                        int_rings.append(geometry.LinearRing(int_coords))
                    poly = geometry.Polygon(ext_coords, int_rings)
                else:
                    poly = geometry.Polygon(ext_coords)
                polygons.append(poly)
        elif type(self.shape) is geometry.polygon.Polygon:
            ext_coords = np.array(self.shape.exterior)
            ext_coords += shift
            if len(self.shape.interiors) > 0:
                int_rings = []
                for i in self.shape.interiors:
                    int_coords = np.array(i)
                    int_coords += shift
                    int_rings.append(geometry.LinearRing(int_coords))
                poly = geometry.Polygon(ext_coords, int_rings)
            else:
                poly = geometry.Polygon(ext_coords)
            polygons.append(poly)
        else:
            raise TypeError('Shape not the correct type.')

        return polygons

    def export_to_shp(self, filename, epsg, shift=(0, 0)):
        """
        Export the object to a shapefile.

        Parameters
        ----------
        filename : string
            The path and filename where the shapefile will be saved.
        epsg : int
            The epsg number of the coordinate system to use for the shapefile.
        global_shift : list or tuple
            The [X, Y] shift in coordinates to be shift back before
            saving.

        Output
        ------
        Shapefile
        """
        crs = from_epsg(epsg)

        polygons = self.shift_polygons(shift)

        if type(self.shape) is geometry.multipolygon.MultiPolygon:
            poly = geometry.MultiPolygon(polygons)
        elif type(self.shape) is geometry.polygon.Polygon:
            poly = geometry.Polygon(polygons[0])

        schema = {'geometry': 'Polygon',
                  'properties': {'area': 'float',
                                 'elongatedness': 'float',
                                 'length': 'float',
                                 'width': 'float'}, }

        with fiona.open(filename, 'w', 'ESRI Shapefile',
                        schema=schema, crs=crs) as c:
            c.write({
                'geometry': geometry.mapping(poly),
                'properties': {'area': float(self.area),
                               'elongatedness': float(self.elongatedness),
                               'length': float(self.length),
                               'width': float(self.width)}, })


def compute_rectangularity(alpha_shape):
    """
    Computes the rectangularity of an point cloud object by computing
    the ratio between the area of the object (area of the concave hull) and
    the area of the oriented bounding box.

    Parameters
    ----------
    alpha_shape : AlphaShape object
        The AlphaShape of the object of which to compute the rectangularity.

    Returns
    -------
    rectangularity : float
        The ratio between the concave hull area and the bounding box area.
    """
    bbox = BoundingBox(alpha_shape.points, alpha_shape.tri.convex_hull)
    return alpha_shape.area / bbox.area


def segment_object(points, min_size, threshold, alpha=0.4,
                   k_init=30, max_dist_init=float('inf'),
                   k=8, max_dist=float('inf')):
    """
    Segment an object into smaller rectangular objects.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    min_size : int
        The minimum size (number of points) an object needs to be.
    threshold : float
        The threshold value the rectangularity needs to be for the points to
        be considered of the same object.
    alpha : float
        The alpha value used for the computation of the alpha shapes.
    k_init : int
        The size of the initial object when segmenting.
    max_dist_init : float or int
        The max distance points can be from the starting point for them to be
        added to the initial object
    k : int
        The number of neighbours considered when growing a segment.
    max_dist : float or int
        The max distance a point can be from another point and still be
        considered from the same neighbourhood.

    Returns
    -------
    segments : list of arrays
        The points belonging to each rectangular segment.
    """
    segments = []
    kdtree = cKDTree(points)

    points_remain = np.array(range(len(points)))
    while len(points_remain) > 3:
        seg = grow_rectangular(points, points_remain, kdtree, threshold, alpha,
                               k_init=k_init, max_dist_init=max_dist_init,
                               k=k, max_dist=max_dist)
        if len(seg) > min_size:
            points_segment = points[seg]
            segments.append(points_segment)

        seg_i = [np.argwhere(points_remain == i)[0][0] for i in seg]
        points_remain = np.delete(points_remain, seg_i)

    return segments


def grow_rectangular(points, points_remain, kdtree, threshold, alpha=0.4,
                     k_init=30, max_dist_init=float('inf'),
                     k=8, max_dist=float('inf')):
    """
    Grow a region based on rectangularity

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of all the points.
    points_remain : (Mx2) array
        The coordinates of the remaining points which need to be segmented.
    kdtree : SciPy KDtree object
        A SciPy KDtree data structure objects of the points.
    threshold : float
        The threshold value the rectangularity needs to be for the points to
        be considered of the same object.
    alpha : float
        The alpha value used for the computation of the alpha shapes.
    k_init : int
        The size of the initial object when segmenting.
    max_dist_init : float or int
        The max distance points can be from the starting point for them to be
        added to the initial object
    k : int
        The number of neighbours considered when growing a segment.
    max_dist : float or int
        The max distance a point can be from another point and still be
        considered from the same neighbourhood.

    Returns
    -------
    cluster : list
        The indices of the points belonging to the same region according to
        the rectangularity constraint.
    """
    # Initiate starting region
    # Locate point with the smallest x value
    min_x = np.argmin(points[points_remain, 0])
    p0_i = np.where(np.logical_and(points[:, 0] ==
                                   points[points_remain][min_x][0],
                                   points[:, 1] ==
                                   points[points_remain][min_x][1]))[0][0]
    p0 = points[p0_i]
    # Determine neighbourhood
    p0_nn_dist, p0_nn_i = kdtree.query(p0, k=k_init,
                                       distance_upper_bound=max_dist_init)
    p0_nn_i = np.delete(p0_nn_i, np.argwhere(p0_nn_dist == float('inf')))
    cluster = list(set(p0_nn_i).intersection(points_remain))
    # Check if there are enough points nearby
    if len(cluster) < 4:
        return [p0_i]
    # Compute initial rectangularity and check if it is above the treshold
    alpha_shape = AlphaShape(points[cluster], alpha, inc=True)
    rectangularity = compute_rectangularity(alpha_shape)
    if rectangularity < threshold:
        return [p0_i]

    # Start region growing
    point_queue = cluster[:]
    while point_queue != []:
        # Get next point in queue and determine its neighbourhood
        p = point_queue.pop(0)
        p_nn_dist, p_nn_i = kdtree.query(points[p], k=k,
                                         distance_upper_bound=max_dist)
        p_nn_i = np.delete(p_nn_i, np.argwhere(p_nn_dist == float('inf')))
        p_nn_i = list(set(p_nn_i).intersection(points_remain))
        # for each point in the neighbourhood check if the region would still
        # rectangular if the point was added. Add point to region and point
        # queue if taht is the case.
        for pi in p_nn_i:
            if pi not in cluster:
                alpha_shape.add_points(np.array([points[pi]]))
                rectangularity = compute_rectangularity(alpha_shape)
                if rectangularity > threshold:
                    cluster.append(pi)
                    point_queue.append(pi)
                else:
                    alpha_shape.remove_last_added()

    alpha_shape.tri.close()

    return cluster


def angle_difference(angle1, angle2):
    """
    Compute the difference between two angles.

    Parameters
    ----------
    angle1 : float
        One of the angles to compare in radians.
    angle2 : float
        One of the angles to compare in radians.

    Returns
    -------
    angle_dif : float
        The differece between the angles.
    """
    return 180 - abs(abs(angle1 - angle2) - 180)


def merge_objects(objects, max_dist, max_dir_dif,  max_c_dir_dif,
                  min_elong=0, max_width=float('inf'), alpha=0.4):
    """
    Merges objects if they are in proximity, have the same direction and are
    aligned with each other.

    Parameters
    ----------
    objects : list of LinearElement objects
        The linear elements to be merged.
    max_dist : float or int
        The maximum distance two objects can be from each other to be
        considered for merging
    max_dir_dif : float
        The maximum difference of the directions of the objects in radians.
        Ensures the objects point in the same direction.
    max_c_dir_dif : float
        The maximum difference of the angle between the center points and
        the directions of the objects in radians. Ensures the objects are
        aligned.
    min_elong : float or int
        Minimum elongatedness of the objects to be considered for merging.
        To ensure only elongated objects will be merged.
    max_width : float or int
        The maximum width of the objects to be considered for merging. To
        ensure only narrow objects will be merged.
    alpha : float
        The alpha used to compute the alpha shape of the newly created merged
        objects.

    Returns
    -------
    objects : list of LinearElement objects
        The merged linear objects.
    """
    objects = objects[:]
    n_objects = len(objects)

    # Calculate distances between all polygons
    distances = np.full((n_objects, n_objects), np.inf)
    for i in xrange(n_objects):
        poly1 = objects[i].shape
        for j in xrange(i+1, n_objects):
            poly2 = objects[j].shape
            dist = poly1.distance(poly2)
            distances[i, j] = dist

    # Select the polygons which are close to each other as candidates
    candidates = distances < max_dist
    candidates = np.transpose(np.where(candidates == True))

    # Determine if the candidates meet the criteria for merging
    to_merge = {}
    for c in candidates:
        if (objects[c[0]].width < max_width and
                objects[c[1]].width < max_width):
            # Check if the objects point in the same direction
            dir_dif = angle_difference(objects[c[0]].direction,
                                       objects[c[1]].direction)
            # Check if the objects are aligned by comparing the angle between
            # the two center points and the directions the objects face
            x = objects[c[0]].shape.centroid.x-objects[c[1]].shape.centroid.x
            y = objects[c[0]].shape.centroid.y-objects[c[1]].shape.centroid.y
            centroid_dir = math.atan2(y, x)
            c_dir_dif1 = angle_difference(objects[c[0]].direction,
                                          centroid_dir)
            c_dir_dif2 = angle_difference(objects[c[1]].direction,
                                          centroid_dir)
            # Set to merge if all criteria are met
            if (dir_dif < max_dir_dif and
                    objects[c[0]].elongatedness > min_elong and
                    objects[c[1]].elongatedness > min_elong and
                    c_dir_dif1 < max_c_dir_dif and c_dir_dif2 < max_c_dir_dif):
                for k, v in to_merge.iteritems():
                    if c[0] in v:
                        to_merge.setdefault(k, []).append(c[1])
                        break
                    if c[1] in v:
                        to_merge.setdefault(k, []).append(c[0])
                        break
                else:
                    to_merge.setdefault(c[0], []).append(c[1])

    # Create the new objects from the objects to be merged
    to_remove = []
    for k, v in to_merge.iteritems():
        merge_objects_idx = [k]
        merge_objects_idx.extend(v)
        merge_objects = [objects[i] for i in merge_objects_idx]
        points = merge_objects[0].points
        for x in merge_objects[1:]:
            points = np.append(points, x.points, axis=0)
        new_object = VegetationObject(points, alpha)
        new_object.width = max([x.width for x in merge_objects])
        new_object.length = sum([x.length for x in merge_objects])
        new_object.elongatedness = new_object.length / new_object.width
        new_object.direction = (sum([x.direction for x in merge_objects]) /
                                len(merge_objects))
        objects.append(new_object)
        to_remove.extend(merge_objects_idx)

    # Remove the objects which are now part of a merged object
    objects = [x for i, x in enumerate(objects) if i not in to_remove]

    return objects


def export_to_shapefile(filename, linear_elements, epsg, global_shift=(0, 0)):
    """
    Export linear elements to a shapefile.

    Parameters
    ----------
    filename : string
        The path and filename where the shapefile will be saved.
    linear_elements : list of LinearElement objects
        The linear elements to be saved in the shapefile.
    epsg : int
        The epsg number of the coordinate system to use for the shapefile.
    global_shift : list or tuple
        The [X, Y] shift in coordinates to be shift back before
        saving.

    Output
    ------
    Shapefile
    """
    crs = from_epsg(epsg)
    schema = {'geometry': 'Polygon',
              'properties': {'id': 'int',
                             'area': 'float',
                             'elongatedness': 'float',
                             'length': 'float',
                             'width': 'float',
                             'dir': 'float'
                             }, }

    with fiona.open(filename, 'w', 'ESRI Shapefile',
                    schema=schema, crs=crs) as c:
        for i, l in enumerate(linear_elements):
            polygons = l.shift_polygons(global_shift)
            if type(l.shape) is geometry.multipolygon.MultiPolygon:
                p = geometry.MultiPolygon(polygons)
            elif type(l.shape) is geometry.polygon.Polygon:
                p = geometry.Polygon(polygons[0])

            c.write({
                'geometry': geometry.mapping(p),
                'properties': {'id': i,
                               'area': float(l.area),
                               'elongatedness': float(l.elongatedness),
                               'length': float(l.length),
                               'width': float(l.width),
                               'dir': float(math.degrees(l.direction))
                               }, })
