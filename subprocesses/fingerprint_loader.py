from typing import Tuple, List
import csv

import numpy


def load_numpy_fingerprints(path: str) -> Tuple[List[numpy.ndarray], List[str]]:
    """
    Load fingerprint data from numpy file

    :param path: Path to file
    :type path: str
    :return: Tuple containing list of numpy arrays and a list of file names
    :rtype: Tuple[List[numpy.ndarray], List[str]
    """
    data = numpy.load(path)
    results = [c[1] for c in data]
    file_data = [c[0] for c in data]
    return results, file_data


def _load_row(row: List[str], results: List[numpy.ndarray], file_data: List[str]) -> None:
    """
    Raad csv row and add contenst to the given lists

    :param row: Row to read
    :type row: List[str]
    :param results: List of numpy arrays to add to
    :type results: List[numpy.ndarray]
    :param file_data: List of file name strings to add to
    :type file_data: List[str]
    """
    file_data.append(row[0])
    results.append(numpy.asarray([int(x) for x in row[1:]]))


def load_delimited_fingerprints(path: str, delimiter: str) -> Tuple[List[numpy.ndarray], List[str]]:
    """
    Load fingerprint data from csv file with given delimiter.

    :param path: Path to csv file
    :type path: str
    :param delimiter: Cell delimiter used in file
    :type delimiter: str
    :return: Tuple containing list of numpy arrays and a list of file names
    :rtype: Tuple[List[numpy.ndarray], List[str]
    """
    results = []
    file_data = []
    with open(path) as in_file:
        csv_reader = csv.reader(in_file, delimiter=delimiter)
        for row in csv_reader:
            _load_row(row, results, file_data)
    return results, file_data


def load_tsv_fingerprints(path: str) -> Tuple[List[numpy.ndarray], List[str]]:
    """
    Load fingerprint data from tsv file.

    :param path: Path to tsv file
    :type path: str
    :return: Tuple containing list of numpy arrays and a list of file names
    :rtype: Tuple[List[numpy.ndarray], List[str]
    """
    return load_delimited_fingerprints(path, '\t')


def load_csv_fingerprints(path: str) -> Tuple[List[numpy.ndarray], List[str]]:
    """
    Load fingerprint data from tsv file.

    :param path: Path to csv file
    :type path: str
    :return: Tuple containing list of numpy arrays and a list of file names
    :rtype: Tuple[List[numpy.ndarray], List[str]
    """
    return load_delimited_fingerprints(path, ',')
