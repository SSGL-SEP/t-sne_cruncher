import numpy as np


def normalize(x: np.ndarray, min_value: int, max_value: int):
    """
    Normalize values in given numpy array to be between 2 given values.
    
    :param x: numpy array containing values to normalize 
    :type x: numpy.ndarray
    :param min_value: Smallest allowed value
    :type min_value: int
    :param max_value: Larges allowed value
    :type max_value: int
    :return: Normalized numpy array
    :rtype: numpy.ndarray
    """
    x -= min([min(y) for y in x])
    x /= (max([max(y) for y in x]) / (max_value - min_value))
    x += min_value
    return x
