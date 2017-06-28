import os
import errno
import csv
from typing import List, Dict, TypeVar, Any
from argparse import Namespace

import numpy as np


T = TypeVar("T")


def all_files(folder_path: str, exts: List[str]) -> str:
    """
    Gather all files conforming to provided extensions.

    :param folder_path: Path to folder
    :type folder_path: str
    :param exts: list of file extensions to accept
    :type exts: List[str]
    :rtype: Union[str, None]
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
        if not (os.path.isdir(path) and ex.errno == errno.EEXIST):
            raise


def normalize(x: np.ndarray, min_value: int, max_value: int) -> np.ndarray:
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
    x -= x.min()
    x /= (x.max() / (max_value - min_value))
    x += min_value
    return x


def parse_metadata(args: Namespace, index_dict: Dict[str, int]) -> Dict[str, dict]:
    """
    Generate metadata dictionary based on csv.
    :param args: Command line arguments
    :type args: argparse.Namespace
    :param index_dict: Dictionary mapping file name to index.
    :type index_dict: Dict[str, int]
    :return: Metadata Dictionary
    :rtype: Dict[str, Dict[str, bool]]
    """
    file_path = args.collect_metadata
    ignorables = args.tags_to_ignore
    unfilterables = args.unfilterables
    d = {}
    td = {}
    h = None
    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if not h:
                h = row
                for t in h:
                    if t not in ignorables:
                        d[t] = {"__filterable": t not in unfilterables}
                        td[t] = {}
            else:
                _parse_row(td, h, row, index_dict, ignorables)
    for tag in td.keys():
        vl = sorted(td[tag].keys())
        for v in vl:
            d[tag][v] = td[tag][v]
    return d


def _parse_row(d: Dict[str, Any], h: List[str], row: List[str], index_dict: Dict[str, int], ignorables: List[str]) -> None:
    """
    Add csv row data to dictionary.

    :param d: Dictionary of tags
    :type d: Dict[str, Dict]
    :param h: List of column headers
    :type h: List[str]
    :param row: List of row values
    :type row: List[str]
    :param index_dict: Dictionary mapping file names to index
    :type index_dict: Dict[str, int]
    :param ignorables: List of tag types to ignore.
    :type ignorables: List[str]
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
    Insert suffix into file path eg. foo/bar.json, _1 -> foo/bar_1.json

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
    def __init__(self, items: List[T]):
        """
        Create instance of UnionFind
        :param items: List of "nodes"
        :type items: List[T]
        """
        self.parents = {i: i for i in items}
        self.sizes = {i: 1 for i in items}
        self.components = len(items)

    def find(self, a: T, b: T) -> bool:
        """
        Find out if objects a and b are in the same subset.
        :param a: An instance of T in UnionFind
        :type a: T
        :param b: Another instance of T in UnionFind
        :type b: T
        :return: True if both objects are in the same subset.
        :rtype: bool
        """
        if (a not in self.parents) or (b not in self.parents):
            raise ValueError("{} or {} not present in union-find structure".format(a, b))
        return self[a] == self[b]

    def root(self, item: T) -> T:
        """
        Find root of subset that item is in.
        :param item: item to find root of.
        :type item: T
        :return: Root of set that item is in.
        :rtype: T
        """
        if item not in self.parents:
            raise ValueError("{} not present in union find structure".format(item))
        child = item
        item = self.parents[item]
        while item != child:
            self.parents[child] = self.parents[item]
            child = item
            item = self.parents[item]
        return item

    def union(self, a: T, b: T) -> bool:
        """
        Combine subsets of a and b.
        :param a: An object in UnionFind
        :type a: T
        :param b: Another object in UnionFind
        :type b: T
        :return: True if a union was made.
        :rtype: bool
        """
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

    def __getitem__(self, item: T) -> T:
        return self.root(item)
