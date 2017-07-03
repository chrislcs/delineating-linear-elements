# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import subprocess
import os


def las_to_csv(file_path, method='CloudCompare',
               tool_path='CloudCompare', overwrite=False):
    """
    Uses either CloudCompare or las2txt to convert a LAS to a CSV file.

    Parameters
    ----------
    file_path : string
        Path to the input file.
    method : string
        The tool to be used to clip the points cloud. Valid methods:
        'CloudCompare' or 'las2txt'
    tool_path : string
        Path to the tool executable.
    overwrite : bool
        Force recalculation if output file already exist.

    Returns
    -------
    out_filename : string
        Returns the output file path based on the input files.

    Output
    ------
    out_file : CSV file
         The resulting CSV file.
    """
    file_path_root = os.path.splitext(file_path)[0]
    out_filename = '%s.csv' % file_path_root

    if os.path.isfile(out_filename) and overwrite is False:
        print 'CSV output file already exists.'
    elif method == 'CloudCompare':
        subprocess.call([tool_path, '-SILENT', '-C_EXPORT_FMT', 'ASC',
                         '-PREC', '3', '-SEP', 'SEMICOLON', '-ADD_HEADER',
                         '-o', '-GLOBAL_SHIFT', '0', '0', '0', file_path,
                         '-NO_TIMESTAMP', '-SAVE_CLOUDS'])
        os.rename('%s.asc' % file_path_root, out_filename)
    elif method == 'las2txt':
        subprocess.call([tool_path, '-i', file_path, '-parse', 'xyzirn',
                         '-sep', 'semicolon'])
        os.rename('%s.txt' % file_path_root, out_filename)

    return out_filename


def downsample(file_path, distance, tool_path='CloudCompare',
               overwrite=False):
    """
    Downsample a point cloud by specifying a minimum distance between points.

    Parameters
    ----------
    file_path : string
        Path to the input file.
    distance : float or int
        Minimum distance between points
    tool_path : string
        Path to the CloudCompare executable.
    overwrite : bool
        Force recalculation if output file already exist.

    Returns
    -------
    out_filename : string
        Returns the output file path based on the input files.

    Output
    ------
    out_file : file
         The downsampled point cloud
    """
    path_root, ext = os.path.splitext(file_path)
    out_filename = '%s_sub_%s%s' % (path_root,
                                    str(distance).replace('.', '_'), ext)

    if os.path.isfile(out_filename) and overwrite is False:
        print 'Subsampled output file already exists.'
    else:
        if ext.lower() == '.csv':
            subprocess.call([tool_path, '-SILENT', '-C_EXPORT_FMT', 'ASC',
                             '-SEP', 'COMMA', '-EXT', 'csv',
                             '-o', '-GLOBAL_SHIFT', '0', '0', '0', file_path,
                             '-NO_TIMESTAMP', '-SS', 'SPATIAL', str(distance)])
        elif ext.lower() == '.las':
            subprocess.call([tool_path, '-SILENT', '-C_EXPORT_FMT', 'LAS',
                             '-o', '-GLOBAL_SHIFT', '0', '0', '0', file_path,
                             '-NO_TIMESTAMP', '-SS', 'SPATIAL', str(distance)])
        else:
            raise IOError('Specified file not of valid type.')

        os.rename('%s_SPATIAL_SUBSAMPLED%s' % (path_root, ext), out_filename)

    return out_filename
