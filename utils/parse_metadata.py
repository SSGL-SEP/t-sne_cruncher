"""Provides a function for parsing a specific type of metadata csv into a python dictionary of dictionarys."""
import csv


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
    with open(file_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if not h:
                h = row
            else:
                if row[0] not in d:
                    d[row[0]] = []
                for i in range(len(row)):
                    if row[i]:
                        d[row[0]].append({"key": h[i], "val": row[i]})
    return d
