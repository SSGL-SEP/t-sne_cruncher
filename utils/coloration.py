import colorsys


def _append_color(audio_list, rgb):
    audio_list.append("#{0:02x}{1:02x}{2:02x}".format(
        int(max(0, min(rgb[0], 255))),
        int(max(0, min(rgb[1], 255))),
        int(max(0, min(rgb[2], 255)))))


def _set_color(audio_list, color_by, value_set, color_count):
    for tag_dict in audio_list[5]:
        if tag_dict["key"] == color_by:
            rgb = colorsys.hsv_to_rgb(value_set[tag_dict["val"]] / color_count, 1, 255)
            _append_color(audio_list, rgb)
            return
    _append_color(audio_list, (255, 255, 255))


def _color_by_tag(data_list, color_by):
    value_set = {d["val"]: 0 for e in data_list for d in e[5] if d["key"] == color_by}
    value_list = sorted(list(value_set.keys()))
    for i in range(len(value_list)):
        value_set[value_list[i]] = i
    for audio_list in data_list:
        _set_color(audio_list, color_by, value_set, len(value_list))


def _color_by_manhattan(data_list):
    max_value = max([audio_list[1] + audio_list[2] + audio_list[3] for audio_list in data_list])
    for audio_list in data_list:
        rgb = colorsys.hsv_to_rgb((audio_list[1] + audio_list[2] + audio_list[3])/max_value, 1, 255)
        _append_color(audio_list, rgb)


def add_color(data_list, color_by=None):
    if color_by:
        _color_by_tag(data_list, color_by)
    else:
        _color_by_manhattan(data_list)

