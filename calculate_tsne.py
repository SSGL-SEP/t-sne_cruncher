"""Provides functions for performing a t-SNE reduction on audio fingerprint data"""
import os
from time import time
import numpy as np
from matplotlib import pyplot as plt
from utils import *


def calculate_tsne(file_path: str = os.path.join(os.getcwd(), 'fingerprints.npy')):
    """
    Calculate a 3d reduction of n-dimensional vectors.
    
    :param file_path: Path to npy file describing a 3 dimensional nparray 
    :type file_path: str
    """
    data = np.load(file_path)
    data = data.reshape(len(data), -1)
    print(data.shape)
    data = data.astype(np.float64)
    start = time()
    x_3d = t_sne(data)
    plot_t_sne(x_3d)
    print('initial_dims={}, perplexity={}, {} seconds'.format(30, 30, time() - start))


def save_tsv(data: np.ndarray, file_name: str):
    """
    Save reduced data to a specified .tsv file.
    
    :param data: 2-dimensional numpy array containing reduced data
    :type data: numpy.ndarray
    :param file_name: File name of file to write
    :type file_name: str
    """
    np.savetxt(file_name, data, fmt="%.5f", delimiter="\t")


def t_sne(data: np.ndarray, initial_dims: int = 30, perplexity: int = 30, output_file: str = None):
    """
    Use t-SNE to create a 3-dimensional projection of a 2-dimensional numpy array
    
    :param data: Two-dimensional numpy array to reduce.
    :type data: numpy.ndarray
    :param initial_dims: Number of initial dimensions.
    :type initial_dims: int
    :param perplexity: Initial perplexity value
    :type perplexity: int
    :param output_file: Optional .tsv output file for the projected data.
    :type output_file: str
    :return: Two-dimensional numpy array containing 3-dimensional coordinates for elements.
    :rtype: numpy.ndarray
    """
    x_3d = tsne(data, no_dims=3, initial_dims=initial_dims, perplexity=perplexity)
    if output_file:
        save_tsv(x_3d, output_file)
    return x_3d


def plot_t_sne(x_3d: np.ndarray, output_file: str = os.path.join(os.getcwd(), 'prints.png')):
    """
    Create a scatter plot of a t-SNE 3d projection
    
    :param x_3d: Two-dimensional numpy array containing 3-dimensional coordinates for elements.
    :type x_3d: numpy.ndarray
    :param output_file: .png file to write scatter plot to.
    :type output_file: str
    """
    fig_size = (16, 16)
    point_size = 100

    plt.figure(figsize=fig_size)
    plt.scatter([y[0] for y in x_3d], [y[1] for y in x_3d], c=[sum(y) for y in x_3d], s=point_size)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    # If run as a script: attempts to create a 3d projection of data read from 'fingerprints.npy'
    calculate_tsne()
