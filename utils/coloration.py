import colorsys
from math import sqrt

import numpy

from utils import UnionFind


def _get_color(i: int, max_value: int) -> str:
    rgb = colorsys.hsv_to_rgb(i / max_value, 1, 255)
    return "#{0:02x}{1:02x}{2:02x}".format(
        int(max(0, min(rgb[0], 255))),
        int(max(0, min(rgb[1], 255))),
        int(max(0, min(rgb[2], 255))))


def get_edges(d, value_list, x_3d):
    center_points = {}
    edges = []
    for i in range(len(value_list)):
        key = value_list[i]
        points = numpy.asarray([x_3d[x] for x in d[key]["points"]]).astype(numpy.float64)
        center_points[key] = sum(points) / len(points)
        for neighbour in value_list[:i]:
            edges.append(Edge(key, center_points[key], neighbour, center_points[neighbour]))
    edges = sorted(edges)
    return edges


def select_colors(e, colors, tag_dict):
    if "color" in tag_dict[e.start] and "color" in tag_dict[e.end]:
        return
    elif "color" in tag_dict[e.start]:
        c = tag_dict[e.start]["color"]
        tag_dict[e.end]["color"] = colors.assign_distant(c)
    elif "color" in tag_dict[e.end]:
        c = tag_dict[e.end]["color"]
        tag_dict[e.start]["color"] = colors.assign_distant(c)
    else:
        c = colors.assign()
        tag_dict[e.start]["color"] = c
        tag_dict[e.end]["color"] = colors.assign_distant(c)


def _color_by_tag(d: dict, tag: str, x_3d: numpy.ndarray) -> None:
    value_list = [x for x in d[tag].keys() if not x.startswith("__")]
    edges = get_edges(d[tag], value_list, x_3d)
    union_find = UnionFind(value_list)
    colors = ColorData(len(value_list))
    for e in edges:
        if union_find[e.start] == union_find[e.end]:
            continue
        select_colors(e, colors, d[tag])


def add_color(metadata: dict, x_3d: numpy.ndarray) -> None:
    for tag in metadata:
        _color_by_tag(metadata, tag, x_3d)


class Edge:

    def __init__(self, sn, sc, en, ec):
        self.start = sn
        self.end = en
        self.start_coordinate = sc
        self.end_coordinate = ec
        self.weight = sqrt(sum([(sc[i]-ec[i])**2 for i in range(len(sc))]))

    def __lt__(self, other):
        self.type_check(other)
        return self.weight < other.weight

    def __le__(self, other):
        self.type_check(other)
        return self.weight <= other.weight

    def __gt__(self, other):
        self.type_check(other)
        return self.weight > other.weight

    def __ge__(self, other):
        self.type_check(other)
        return self.weight >= other.weight

    def __eq__(self, other):
        self.type_check(other)
        return self.weight == other.weight

    def type_check(self, other):
        if type(self) != type(other):
            raise TypeError("Can not compare {} and {}".format(type(self), type(other)))


class ColorData:
    def __init__(self, max_value):
        self.start_index = 0
        self.assigned = 0
        self.colors = [_get_color(i, max_value) for i in range(max_value)]
        self.color_usages = {c: True for c in self.colors}
        self.color_indexes = {self.colors[i]: i for i in range(len(self.colors))}

    def available(self, idx):
        return self.color_usages[self.colors[idx]]

    def give(self, idx):
        c = self.colors[idx]
        self.color_usages[c] = False
        self.assigned += 1
        return c

    def assign(self):
        if self.assigned >= len(self.colors):
            print("No more unique colors to assign. Something went wrong. Assigning #ffffff")
            return "#ffffff"
        while not self.available(self.start_index):
            self.start_index += 1
        return self.give(self.start_index)

    def assign_distant(self, color):
        if self.assigned >= len(self.colors):
            print("No more unique colors to assign. Something went wrong. Assigning #ffffff")
            return "#ffffff"
        c_count = len(self.colors)
        start_idx = self.color_indexes[color]
        idx = (start_idx + c_count // 2) % c_count
        if self.available(idx):
            return self.give(idx)
        mi = (idx - c_count // 4) % c_count
        ma = (idx + c_count // 4) % c_count
        while True:
            if self.available(mi):
                return self.give(mi)
            if self.available(ma):
                return self.give(ma)
            mi = (mi + 1) % c_count
            ma = (ma - 1) % c_count
            if mi == idx or ma == idx:
                break
        mi = (idx - c_count // 4) % c_count
        ma = (idx + c_count // 4) % c_count
        while mi != start_idx and ma != start_idx:
            mi = (mi - 1) % c_count
            ma = (ma + 1) % c_count
            if self.available(mi):
                return self.give(mi)
            if self.available(ma):
                return self.give(ma)
        # safety catch. The lines below should never get called.
        print("No more unique colors to assign. Something went wrong. Assigning #ffffff")
        return "#ffffff"
