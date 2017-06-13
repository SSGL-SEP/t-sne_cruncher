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


def parse_metadata(file_path: str, index_dict: dict, unfilterables: list, ignorables: list) -> dict:
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
                for t in h:
                    if t not in ignorables:
                        d[t] = {"__filterable": t not in unfilterables}
            else:
                _parse_row(d, h, row, index_dict, ignorables)
    return d


def _parse_row(d: dict, h: list, row: list, index_dict: dict, ignorables: list) -> None:
    """
    Add csv row data to dictionary

    :param d: dictionary to add data to
    :type d: Dict[str, List[Dict[str, str]]]
    :param h: list of header data
    :type h: List[str]
    :param row: data to add to dictionary
    :type row: List[str]
    """
    if row[0] not in index_dict:
        return
    fi = index_dict[row[0]]
    for i in range(len(row)):
        if h[i] in ignorables:
            continue
        if row[i] in d[h[i]]:
            d[h[i]][row[i]]["points"].append(fi)
        else:
            d[h[i]][row[i]] = {"points": [fi]}


def insert_suffix(file_path: str, suffix: str) -> str:
    """
    Insert suffix into file path foo/bar.json, _1 -> foo/bar_1.json

    :param file_path: file path to insert suffix into
    :type file_path: str
    :param suffix: suffix to insert
    :type suffix: str
    :return: modified file path
    :rtype: str
    """
    prefix, ext = os.path.splitext(file_path)
    return "".join([prefix, suffix, ext])


class UnionFind:
    def __init__(self, items):
        self.parents = {i: i for i in items}
        self.sizes = {i: 1 for i in items}
        self.components = len(items)

    def find(self, a, b):
        if (a not in self.parents) or (b not in self.parents):
            raise ValueError("{} or {} not present in union-find structure".format(a, b))
        return self[a] == self[b]

    def root(self, item):
        child = item
        item = self.parents[item]
        while item != child:
            self.parents[child] = self.parents[item]
            child = item
            item = self.parents[item]
        return item

    def union(self, a, b):
        if (a not in self.parents) or (b not in self.parents):
            raise ValueError("{} or {} not present in union-find structure".format(a, b))
        a = self[a]
        b = self[b]
        if a == b:
            return False
        if self.sizes[a] < self.sizes[b]:
            self.parents[a] = b
            self.sizes[b] += self.sizes[a]
        else:
            self.parents[b] = a
            self.sizes[a] += self.sizes[b]
        self.components -= 1
        return True

    def __getitem__(self, item):
        return self.root(item)
