"""Provides functions for performing a t-SNE reduction on audio fingerprint data"""
import os
from time import time
from multiprocessing.pool import Pool

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils import *


def calculate_tsne(file_path: str = os.path.join(os.getcwd(), 'fingerprints.npy')):
    """
    Calculate a 3d reduction of n-dimensional vectors.
    
    :param file_path: Path to npy file describing a 3 dimensional nparray 
    :type file_path: str
    """
    data = np.load(file_path)
    data = data.reshape(len(data), -1)
    data = data.astype(np.float64)
    start = time()
    x_3d = t_sne(data)
    # noinspection PyTypeChecker
    for to_plot in x_3d:
        plot_t_sne(to_plot[0])
    print('perplexity={}, {} seconds'.format(30, time() - start))
    return x_3d


def save_tsv(data: np.ndarray, file_name: str):
    """
    Save reduced data to a specified .tsv file.
    
    :param data: 2-dimensional numpy array containing reduced data
    :type data: numpy.ndarray
    :param file_name: File name of file to write
    :type file_name: str
    """
    np.savetxt(file_name, data, fmt="%.5f", delimiter="\t")


def t_sne(data: np.ndarray, perplexity=None, output_file: str = None, no_dims: int = 3):
    """
    Use t-SNE to create a 3-dimensional projection of a 2-dimensional numpy array
    
    :param no_dims: Number of output dimensions. Typically 3 or 2
    :type no_dims: int
    :param data: Two-dimensional numpy array to reduce.
    :type data: numpy.ndarray
    :param perplexity: Initial perplexity value
    :type perplexity: List[int]
    :param output_file: Optional .tsv output file for the projected data.
    :type output_file: str
    :return: Two-dimensional numpy array containing 3-dimensional coordinates for elements.
    :rtype: List[numpy.ndarray]
    """
    if perplexity is None:
        perplexity = [30]
    with Pool() as p:
        l_3d = list(p.map(t_sne_job, [(data, x, no_dims) for x in perplexity]))
    for x_3d in l_3d:
        if output_file:
            save_tsv(x_3d[0], insert_suffix(output_file, x_3d[1]))
    return l_3d


def t_sne_job(params: tuple) -> tuple:
    """
    run r_sne fit transform on specified data in a specified way

    :param params: tuple of data, perplexity, output dimension
    :type params: tuple(numpy.ndarray, int, int)
    :return: typle of reduced data and perplexity
    :rtype: tuple(numpy.ndarray, str)
    """
    print("Running t-SNE with perplexity {}".format(params[1]))
    model = TSNE(n_components=params[2], perplexity=params[1], method='exact')
    return model.fit_transform(params[0]), str(params[1])


def plot_t_sne(x_3d: np.ndarray, output_file: str = os.path.join(os.getcwd(), 'prints.png')):
    """
    Create a scatter plot of a t-SNE 3d projection
    
    :param x_3d: Two-dimensional numpy array containing 2 or 3-dimensional coordinates for elements.
    :type x_3d: numpy.ndarray
    :param output_file: .png file to write scatter plot to.
    :type output_file: str
    """
    fig_size = (16, 16)
    point_size = 100

    plt.figure(figsize=fig_size)
    x_coords = [x[0] for x in x_3d]
    y_coords = [y[1] for y in x_3d]
    colors = [(c[0] + c[1] + (c[2] if len(c) > 2 else 0)) for c in x_3d]
    plt.scatter(x_coords, y_coords, c=colors, s=point_size)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    calculate_tsne()
