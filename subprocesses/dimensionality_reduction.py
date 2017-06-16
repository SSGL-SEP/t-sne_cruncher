"""Provides functions for performing a t-SNE reduction on audio fingerprint data"""
import os
from multiprocessing.pool import Pool

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import *


def t_sne(data: np.ndarray, no_dims: int = 3, args=None, a_func=None, a_params=None):
    if args:
        perplexity = args.perplexity
    else:
        perplexity = [30]
    if args and args.parallel:
        with Pool() as p:
            l_nd = list(p.map(t_sne_job, [(data, x, no_dims, a_func, a_params) for x in perplexity]))
    else:
        l_nd = []
        for p in perplexity:
            l_nd.append(t_sne_job((data, p, no_dims, a_func, a_params)))
    return l_nd


def t_sne_job(params: tuple) -> tuple:
    print("Running t-SNE with perplexity {}".format(params[1]))
    model = TSNE(n_components=params[2], perplexity=params[1], method='exact', verbose=2)
    x_nd = model.fit_transform(params[0]), str(params[1])
    if len(params) >= 5 and params[3] and params[4]:
        params[3](x_nd, *params[4])
    return x_nd


def pca(data: np.ndarray, output_dimensions: int, args=None, a_func=None, a_params=None):
    model = PCA(n_components=output_dimensions, svd_solver='full')
    x_nd = model.fit_transform(data), "pca"
    if a_func and a_params:
        a_func(x_nd, *a_params)
    return x_nd


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
    print("Wrote plot to {}.".format(output_file))
