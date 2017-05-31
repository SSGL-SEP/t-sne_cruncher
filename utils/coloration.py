import colorsys
from functools import reduce


def _append_color(e, rgb):
    e.append("#{0:02x}{1:02x}{2:02x}".format(
        int(max(0, min(rgb[0], 255))),
        int(max(0, min(rgb[1], 255))),
        int(max(0, min(rgb[2], 255)))))


def _set_color(e, color_by, s, c):
    for d in e[5]:
        if d["key"] == color_by:
            rgb = colorsys.hsv_to_rgb(s[d["val"]] / c, 1, 255)
            _append_color(e, rgb)
            return
    _append_color(e, (255, 255, 255))


def _color_by_tag(data_list, color_by):
    s = {d["val"]: 0 for e in data_list for d in e[5] if d["key"] == color_by}
    li = sorted(list(s.keys()))
    for i in range(len(li)):
        s[li[i]] = i
    for e in data_list:
        _set_color(e, color_by, s, len(li))


def _color_by_manhattan(data_list):
    m = max([e[1] + e[2] + e[3] for e in data_list])
    for e in data_list:
        rgb = colorsys.hsv_to_rgb((e[1] + e[2] + e[3])/m, 1, 255)
        _append_color(e, rgb)


def add_color(data_list, color_by):
    if color_by:
        _color_by_tag(data_list, color_by)
    else:
        _color_by_manhattan(data_list)

