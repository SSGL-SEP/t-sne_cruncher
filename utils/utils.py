import os
import errno
import csv

import numpy as np


def all_files(folder_path: str, exts: list):
    """
    Gathers all files conforming to provided extensions.

    :param folder_path: Path to folder
    :type folder_path: str
    :param exts: list of file extensions to accept
    :type exts: List[str]
    """
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            base, ext = os.path.splitext(f)
            joined = os.path.join(root, f)
            if ext.lower() in exts:
                yield joined


def mkdir_p(path: str):
    """
    Attempts ot create a given folder

    :param path: Folder path 
    :type path: str
    """
    try:
        os.makedirs(path)
    except OSError as ex:
        v = os.path.isdir(path)
        w = ex.errno == errno.EEXIST
        if v and w:
            return "ex"
            pass
        else:
            raise


def normalize(x: np.ndarray, min_value: int, max_value: int):
    """
    Normalize values in given numpy array to be between 2 given values.

    :param x: numpy array containing values to normalize 
    :type x: numpy.ndarray
    :param min_value: Smallest allowed value
    :type min_value: int
    :param max_value: Larges allowed value
    :type max_value: int
    :return: Normalized numpy array
    :rtype: numpy.ndarray
    """
    x -= min([min(y) for y in x])
    x /= (max([max(y) for y in x]) / (max_value - min_value))
    x += min_value
    return x


def parse_metadata(file_path: str):
    """
    Parse .csv file containing metadata into a dictionary.

    :param file_path: .csv file to read
    :type file_path: str
    :return: Dictionary of metadata values.
    :rtype: Dict{str: List[Dict{str: str}]}
    """
    d = {}
    h = None
    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if not h:
                h = row
            else:
                _parse_row(d, h, row)
    return d


def _parse_row(d, h, row):
    if row[0] not in d:
        d[row[0]] = []
    for i in range(len(row)):
        if row[i]:
            d[row[0]].append({"key": h[i], "val": row[i]})


def insert_suffix(file_path, suffix):
    prefix, ext = os.path.splitext(file_path)
    return "".join([prefix, suffix, ext])
