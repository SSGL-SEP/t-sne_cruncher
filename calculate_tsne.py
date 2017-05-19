from matplotlib import pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
from utils import *
from time import time


def calculate_tsne(file_path=os.path.join(os.getcwd(), 'fingerprints.npy')):
    data = np.load(file_path)
    data = data.reshape(len(data), -1)
    print(data.shape)
    data = data.astype(np.float64)
    start = time()
    x_3d = t_sne(data)
    plot_t_sne(x_3d)
    print('initial_dims={}, perplexity={}, {} seconds'.format(30, 30, time() - start))


def save_tsv(data, fn):
    np.savetxt(fn, data, fmt="%.5f", delimiter="\t")


def t_sne(data, initial_dims=30, perplexity=30, output_file=None):
    x_3d = tsne(data, no_dims=3, initial_dims=initial_dims, perplexity=perplexity)
    if output_file:
        save_tsv(x_3d, output_file)
    return x_3d


def plot_t_sne(x_3d, output_file=os.path.join(os.getcwd(), 'prints.png')):
    fig_size = (16, 16)
    point_size = 100

    plt.figure(figsize=fig_size)
    plt.scatter([y[0] for y in x_3d], [y[1] for y in x_3d], c=[sum(y) for y in x_3d], s=point_size)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    calculate_tsne()
