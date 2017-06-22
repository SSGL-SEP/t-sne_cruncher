import os
from multiprocessing.pool import Pool
from argparse import Namespace
from typing import Iterable, Any, Tuple, List, Dict, Union, Callable

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import *


def t_sne(data: np.ndarray, no_dims: int = 3, args: Namespace = None,
          a_func: Callable = None, a_params: Iterable[Any] = None) -> List[Tuple[np.ndarray, str]]:
    """
    Run t-SNE dimensionality reduction on data.

    :param data: numpy array of shape (n, m) containing n m-dimensional vectors.
    :type data: numpy.ndarray
    :param no_dims: Number of output dimensions.
    :type no_dims: int
    :param args: Command line parameters.
    :type args: argparse.Namespace
    :param a_func: Function to call with reduced data and a_params after reduction.
    :type a_func: Callable
    :param a_params: Parameters to be appended to a_func call after dimensionality reduction.
    :type a_params: Iterable[Any]
    :return: List of tuples containing dimensionally reduced data and perplexity string.
    :rtype: List[Tuple[numpy.ndarray, str]]
    """
    perplexity = args.perplexity if args else [30]
    if args and args.parallel:
        with Pool() as p:
            l_nd = list(p.starmap(_t_sne_job, [(data, x, no_dims, a_func, a_params) for x in perplexity]))
    else:
        l_nd = []
        for p in perplexity:
            l_nd.append(_t_sne_job(data, p, no_dims, a_func, a_params))
    return l_nd


def _t_sne_job(data: np.ndarray, perplexity: int, no_dims: int,
               a_func: Callable = None, a_params: Iterable[Any] = None) -> Tuple[np.ndarray, str]:
    """
    Run t-SNE dimensionality reduction on data with given preplexity.
    :param data: numpy array of shape (n, m) containing n m-dimensional vectors.
    :type data: numpy.ndarray
    :param perplexity: Perplexity to run t-SNE with.
    :type perplexity: int
    :param no_dims: Number of output dimensions.
    :type no_dims: int
    :param a_func: Function to call with reduced data and a_params after reduction.
    :type a_func: Callable
    :param a_params: Parameters to be appended to a_func call after dimensionality reduction.
    :type a_params: Iterable[Any]
    :return: Tuple containing dimensionally reduced data and perplexity string.
    :rtype: Tuple[numpy.ndarray, str]
    """
    print("Running t-SNE with perplexity {}".format(perplexity))
    model = TSNE(n_components=no_dims, perplexity=perplexity, method='exact', verbose=2)
    x_nd = model.fit_transform(data), str(perplexity)
    if a_func and a_params:
        a_func(x_nd, *a_params)
    return x_nd


def pca(data: np.ndarray, output_dimensions: int, args: Namespace = None,
        a_func: Callable = None, a_params: Iterable[Any] = None) -> List[Tuple[np.ndarray, str]]:
    """
    Run PCA dimensionality reduction on data.
    :param data: numpy array of shape (n, m) containing n m-dimensional vectors.
    :type data: numpy.ndarray
    :param output_dimensions: Number of output dimensions.
    :type output_dimensions: int
    :param args: Command line parameters.
    :type args: argparse.Namespace
    :param a_func: Function to call with reduced data and a_params after reduction.
    :type a_func: Callable
    :param a_params: Parameters to be appended to a_func call after dimensionality reduction.
    :type a_params: Iterable[Any]
    :return: List of length 1 containing tuple with dimensionally reduced data and description string.
    :rtype: List[Tuple[numpy.ndarray, str]]
    """
    model = PCA(n_components=output_dimensions, svd_solver='full')
    x_nd = model.fit_transform(data), "pca"
    if a_func and a_params:
        a_func(x_nd, *a_params)
    return [x_nd]


def _get_colors(x_nd: np.ndarray, metadata: Dict[str, Any] = None,
                color_by: str = None) -> List[Union[int, Tuple[float, float, float]]]:
    """
    Map points to colors based on metadata or manhattan distance from origin.
    :param x_nd: dimensionally reduced data
    :type x_nd: numpy.ndarray
    :param metadata: Metadata dictionary containing coloration data.
    :type metadata: Dict[str, Dict[str, bool]]
    :param color_by: Name of tag to color by
    :type color_by: str
    :return: List of rgb colors if metadata is present or list of manhattan distances if metadata is not present.
    :rtype: List[Union[int, Tuple[float, float, float]]]
    """
    if (not metadata) or (not color_by):
        return [(x[0] + x[1] + (x[2] if len(x) > 2 else 0)) for x in x_nd]
    c = []
    c_dict = {}
    for val in [v for v in metadata[color_by].keys() if not v.startswith("__")]:
        for i in metadata[color_by][val]["points"]:
            c_dict[i] = html_hex_to_rgb(metadata[color_by][val]["color"])
    i = 0
    while i in c_dict:
        c.append(c_dict[i])
        i += 1
    return c


def plot_results(x_nd: np.ndarray, output_file: str = os.path.join(os.getcwd(), 'prints.png'),
                 metadata: Dict[str, Any] = None, color_by: str = None):
    """
    Generate pyplot image of dimensionally reduced data.
    :param x_nd: Dimensionally reduced data
    :type x_nd: numpy.ndarray
    :param output_file: File to write plot to.
    :type output_file: str
    :param metadata: Metadata containing coloration information
    :type metadata: Dict[str, Dict[str, bool]]
    :param color_by: tag type to color by
    :type color_by: str
    """
    fig_size = (16, 16)
    point_size = 20

    plt.figure(figsize=fig_size)
    x_coords = [x[0] for x in x_nd]
    y_coords = [y[1] for y in x_nd]
    colors = _get_colors(x_nd, metadata, color_by)
    plt.scatter(x_coords, y_coords, c=colors, s=point_size)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print("Wrote plot to {}.".format(output_file))
