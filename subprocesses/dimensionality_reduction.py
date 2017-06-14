"""Provides functions for performing a t-SNE reduction on audio fingerprint data"""
import os
from multiprocessing.pool import Pool

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import *


def save_tsv(data: np.ndarray, file_name: str):
    """
    Save reduced data to a specified .tsv file.
    
    :param data: 2-dimensional numpy array containing reduced data
    :type data: numpy.ndarray
    :param file_name: File name of file to write
    :type file_name: str
    """
    np.savetxt(file_name, data, fmt="%.5f", delimiter="\t")


def t_sne(data: np.ndarray, no_dims: int = 3, args=None):
    if args:
        perplexity = args.perplexity
    else:
        perplexity = [30]
    if args and args.parallel:
        with Pool() as p:
            l_nd = list(p.map(t_sne_job, [(data, x, no_dims) for x in perplexity]))
    else:
        l_nd = []
        for p in perplexity:
            l_nd += t_sne_job((data, p, no_dims)), p
    if args and args.output_file:
        for x_nd in l_nd:
            save_tsv(x_nd[0], insert_suffix(args.output_file, x_nd[1]))
    return l_nd


def t_sne_job(params: tuple) -> tuple:
    """
    run r_sne fit transform on specified data in a specified way

    :param params: tuple of data, perplexity, output dimension
    :type params: tuple(numpy.ndarray, int, int)
    :return: typle of reduced data and perplexity
    :rtype: tuple(numpy.ndarray, str)
    """
    print("Running t-SNE with perplexity {}".format(params[1]))
    model = TSNE(n_components=params[2], perplexity=params[1], method='exact', verbose=2)
    return model.fit_transform(params[0]), str(params[1]), params[1]


def pca(output_dimensions: int, data: np.ndarray, args):
    model = PCA(n_components=output_dimensions, svd_solver='full')
    return model.fit_transform(data), "pca"


def _get_colors(x_nd: np.ndarray, metadata: dict, color_by: str):
    if (not metadata) or (not color_by):
        return [(c[0] + c[1] + (c[2] if len(c) > 2 else 0)) for c in x_nd]
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
                 metadata: dict = None, color_by: str = None):
    fig_size = (16, 16)
    point_size = 100

    plt.figure(figsize=fig_size)
    x_coords = [x[0] for x in x_nd]
    y_coords = [y[1] for y in x_nd]
    colors = _get_colors(x_nd, metadata, color_by)
    plt.scatter(x_coords, y_coords, c=colors, s=point_size)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
