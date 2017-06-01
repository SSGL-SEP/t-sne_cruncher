import colorsys


def _append_color(audio_list: list, rgb: tuple) -> None:
    """
    Appends the given rgb color as html hex color to audio_list
    
    :param audio_list: List to append color to 
    :type audio_list: List[any]
    :param rgb: rgb color tuple
    :type rgb: Tuple[float, float, float]
    """
    audio_list.append("#{0:02x}{1:02x}{2:02x}".format(
        int(max(0, min(rgb[0], 255))),
        int(max(0, min(rgb[1], 255))),
        int(max(0, min(rgb[2], 255)))))


def _set_color(audio_list: list, color_by: str, value_dict: dict, color_count: int) -> None:
    """
    Adds color o the audio_list based on the other parameters
    
    :param audio_list: Audio list to append color to. 
    :type audio_list: List[any]
    :param color_by: Tag (in list) to apply color by
    :type color_by: str
    :param value_dict: dict containing tags ant their values
    :type value_dict: Dict[str, int]
    :param color_count: number of colors to use.
    :type color_count: int
    """
    for tag_dict in audio_list[5]:
        if tag_dict["key"] == color_by:
            rgb = colorsys.hsv_to_rgb(value_dict[tag_dict["val"]] / color_count, 1, 255)
            _append_color(audio_list, rgb)
            return
    _append_color(audio_list, (255, 255, 255))


def _color_by_tag(data_list: list, color_by: str) -> None:
    """
    Add coloring to all audio data lists in data_list based on color_by
    
    :param data_list: List of audio data lists 
    :type data_list: List[List[any]]
    :param color_by: Name of tag to base coloring on.
    :type color_by: str
    """
    value_set = {d["val"]: 0 for e in data_list for d in e[5] if d["key"] == color_by}
    value_list = sorted(list(value_set.keys()))
    for i in range(len(value_list)):
        value_set[value_list[i]] = i
    for audio_list in data_list:
        _set_color(audio_list, color_by, value_set, len(value_list))


def _color_by_manhattan(data_list: list) -> None:
    """
    Adds color to all audio data list in data_list based on manhattan distance from origin 
    
    :param data_list: List of audio data lists
    :type data_list: List[List[any]]
    """
    max_value = max([audio_list[1] + audio_list[2] + audio_list[3] for audio_list in data_list])
    for audio_list in data_list:
        rgb = colorsys.hsv_to_rgb((audio_list[1] + audio_list[2] + audio_list[3])/max_value, 1, 255)
        _append_color(audio_list, rgb)


def add_color(data_list: list, color_by: str = None) -> None:
    """
    Adds color to all audio data lists in data_list based on tag or manhattan distance from origin

    :param data_list: List of audio data lists
    :type data_list: List[List[any]]
    :param color_by: Optional tag to do coloring by
    :type color_by: str
    """
    if color_by:
        _color_by_tag(data_list, color_by)
    else:
        _color_by_manhattan(data_list)

