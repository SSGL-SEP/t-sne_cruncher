import csv


def parse_metadata(fp):
    d = {}
    h = None
    with open(fp, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if not h:
                h = row
            else:
                d[row[0]] = {h[i]: row[i] for i in range(len(row))}
    return d
