import colorsys

import numpy


def _get_color(i: int, max_value: int) -> str:
    rgb = colorsys.hsv_to_rgb(i / max_value, 1, 255)
    return "#{0:02x}{1:02x}{2:02x}".format(
        int(max(0, min(rgb[0], 255))),
        int(max(0, min(rgb[1], 255))),
        int(max(0, min(rgb[2], 255))))


def _color_by_tag(d: dict, tag: str, x_3d: numpy.ndarray) -> None:
    value_list = sorted(list(d[tag].keys()))
    for i in range(len(value_list)):
        if value_list[i] == "__filterable":
            continue
        d[tag][value_list[i]]["color"] = _get_color(i, len(value_list))


def add_color(metadata: dict, x_3d: numpy.ndarray) -> None:
    for tag in metadata:
        _color_by_tag(metadata, tag, x_3d)
