import colorsys
from math import sqrt
from random import choice
from typing import Tuple, List, TypeVar, Any, Dict

import numpy

from utils import UnionFind

T = TypeVar('T', bound="Edge")


class Edge:

    def __init__(self, sn: str, sc: numpy.ndarray, en: str, ec: numpy.ndarray):
        """
        Create instance of Edge.

        :param sn: Name associated with one end of the edge.
        :type sn: str
        :param sc: Coordinates associated with one end of the edge.
        :type sc: numpy.ndarray
        :param en: Name associated with other end of the edge.
        :type en: str
        :param ec: Coordinates associated with other end of the edge.
        :type ec: numpy.ndarray
        """
        self.start = sn
        self.end = en
        self.start_coordinate = sc
        self.end_coordinate = ec
        self.weight = sqrt(sum([(sc[i]-ec[i])**2 for i in range(len(sc))]))

    def __lt__(self, other: T):
        self.type_check(other)
        return self.weight < other.weight

    def __le__(self, other: T):
        self.type_check(other)
        return self.weight <= other.weight

    def __gt__(self, other: T):
        self.type_check(other)
        return self.weight > other.weight

    def __ge__(self, other: T):
        self.type_check(other)
        return self.weight >= other.weight

    def __eq__(self, other: T):
        self.type_check(other)
        return self.weight == other.weight

    def type_check(self, other: Any):
        if type(self) != type(other):
            raise TypeError("Can not compare {} and {}".format(type(self), type(other)))


class ColorData:
    def __init__(self, max_value: int):
        """
        Create instance of ColorData
        :param max_value: Number of colors needed.
        :type max_value: int
        """
        self.start_index = 0
        self.assigned = 0
        self.colors = [_get_color(i, max_value) for i in range(max_value)]
        self.color_usages = {c: True for c in self.colors}
        self.random_assign = False
        if len(self.colors) != len(self.color_usages):
            self.random_assign = True
            nc = [self.colors[0]]
            for i in range(1, len(self.colors)):
                c = self.colors[i]
                if c == nc[0] or c == nc[-1]:
                    continue
                nc.append(c)
            self.colors = nc
        self.color_indexes = {self.colors[i]: i for i in range(len(self.colors))}

    def available(self, idx: int) -> bool:
        """
        Check if color is available for assignment
        :param idx: index of color
        :type idx: int
        :return: True if color is available
        :rtype: bool
        """
        return self.color_usages[self.colors[idx]]

    def give(self, idx: int) -> str:
        """
        Assign specified color
        :param idx: Index of color.
        :type idx: int
        :return: Html hex representation of assigned color.
        :rtype: str
        """
        c = self.colors[idx]
        if self.random_assign:
            return c
        self.color_usages[c] = False
        self.assigned += 1
        return c

    @staticmethod
    def default() -> str:
        """
        Assigns default color (white).
        :return: Html hex representation of white.
        :rtype: str
        """
        print("Could not find unique value. Returning default '#ffffff'")
        return "#ffffff"

    def assign(self) -> str:
        """
        Assign a new hopefully unused color.
        :return: Html hex representation of assigned color.
        :rtype: str
        """
        if self.random_assign:
            return choice(self.colors)
        if self.assigned >= len(self.colors):
            return self.default()
        while not self.available(self.start_index):
            self.start_index += 1
        return self.give(self.start_index)

    def assign_distant(self, color: str) -> str:
        """
        Assign as distant a color as possible compared to given color.
        :param color: Html hex representation of color.
        :type color: str
        :return: Html hex representation of assigned color.
        :rtype: str
        """
        if self.assigned >= len(self.colors):
            return self.default()
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

    def __str__(self) -> str:
        return """Current index: {}. Colors assigned: {}. Total colors {}.
        Colors: {}
        Usages: {}
        Indexes: {}""".format(self.start_index, self.assigned, len(self.colors),
                              self.colors, self.color_usages, self.color_indexes)


def _get_color(i: int, max_value: int) -> str:
    """
    Generate color with hue at point i between 0 and max_value (normalized).
    :param i: Desired hue (compared to max_value).
    :type i: int
    :param max_value: Number of potential color "steps".
    :type max_value: int
    :return: Html hex representation of color.
    :rtype: str
    """
    rgb = colorsys.hsv_to_rgb(i / max_value, 1, 255)
    return "#{0:02x}{1:02x}{2:02x}".format(
        int(max(0, min(rgb[0], 255))),
        int(max(0, min(rgb[1], 255))),
        int(max(0, min(rgb[2], 255))))


def html_hex_to_rgb(input_html: str) -> Tuple[float, float, float]:
    """
    Convert html hex color string to rgb tuple.
    :param input_html: Html hex string.
    :type input_html: str
    :return: Tuple containing decimal r, g and b values of color.
    :rtype: Tuple[float, float, float]
    """
    return int(input_html[1:3], 16) / 255, int(input_html[3:5], 16) / 255, int(input_html[5:], 16) / 255


def _get_edges(d: Dict[str, dict], value_list: List[str], x_3d: numpy.ndarray) -> List[Edge]:
    """
    Generate graph representation of color cluster centers of gravity.
    :param d: Dictionary of tag used for coloration.
    :type d: Dict[str, dict]
    :param value_list: List of usable keys in d.
    :type value_list: List[str]
    :param x_3d: Coordinates for all points in map.
    :type x_3d: numpy.ndarray
    :return: List of edges (distances between color clusters).
    :rtype: List[Edge]
    """
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


def select_colors(e: Edge, colors: ColorData, tag_dict: Dict[str, Any]) -> None:
    """
    Assing colors to nodes based on edge.
    :param e: Edge with ends to be colored.
    :type e: Edge
    :param colors: ColorData object for color assignments.
    :type colors: ColorData
    :param tag_dict: Dictionary of tag used for color assignemnt.
    :type tag_dict: Dict[str, bool]
    """
    if "color" in tag_dict[e.start] and "color" in tag_dict[e.end]:
        return
    elif "color" in tag_dict[e.start]:
        c = tag_dict[e.start]["color"]
        clr = colors.assign_distant(c)
        tag_dict[e.end]["color"] = clr
    elif "color" in tag_dict[e.end]:
        c = tag_dict[e.end]["color"]
        clr = colors.assign_distant(c)
        tag_dict[e.start]["color"] = clr
    else:
        c = colors.assign()
        tag_dict[e.start]["color"] = c
        clr = colors.assign_distant(c)
        tag_dict[e.end]["color"] = clr


def _color_by_tag(d: Dict[str, dict], tag: str, x_3d: numpy.ndarray) -> None:
    """
    Assign colors to values of tag
    :param d: Dictionary of tags.
    :type d: Dict[str, Dict[str, bool]]
    :param tag: Tag to color by.
    :type tag: str
    :param x_3d: 3d or 2d data to use in center of gravity calculations.
    :type x_3d: numpy.ndarray
    """
    value_list = [x for x in d[tag].keys() if not x.startswith("__")]
    edges = _get_edges(d[tag], value_list, x_3d)
    union_find = UnionFind(value_list)
    colors = ColorData(len(value_list))
    for e in edges:
        if union_find.union(e.start, e.end):
            select_colors(e, colors, d[tag])


def add_color(metadata: Dict[str, dict], x_3d: numpy.ndarray) -> None:
    """
    Assign colors to metadata tags
    :param metadata: Dictionary of tags
    :type metadata: Dict[str, dict]
    :param x_3d: 3d or 2d data to use in center of gravity calculations.
    :type x_3d: numpy.ndarray
    """
    for tag in metadata:
        _color_by_tag(metadata, tag, x_3d)
