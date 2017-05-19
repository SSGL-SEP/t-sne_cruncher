def normalize(x, min_value, max_value):
    x -= min([min(y) for y in x])
    x /= (max([max(y) for y in x]) / (max_value - min_value))
    x += min_value
    return x
