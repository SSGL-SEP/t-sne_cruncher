from matplotlib import pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
from utils import *
from time import time


def main():
    data = np.load(os.path.join(os.getcwd(), 'fingerprints.npy'))
    data = data.reshape(len(data), -1)
    print(data.shape)
    data = data.astype(np.float64)
    start = time()
    with Pool() as p:
        p.map(t_sne, [data])
    print('initial_dims={}, perplexity={}, {} seconds'.format(30, 30, time() - start))


def save_tsv(data, fn):
    np.savetxt(fn, data, fmt="%.5f", delimiter="\t")


def t_sne(data, initial_dims=30, perplexity=30):
    mkdir_p(os.getcwd() + "tsne")
    mkdir_p(os.getcwd() + "plot")
    figsize = (16, 16)
    pointsize = 2

    x_3d = list(tsne(data, no_dims=3, initial_dims=initial_dims, perplexity=perplexity))
    # x_3d = normalize(x_3d)
    save_tsv(x_3d, os.path.join(os.getcwd(), 'tsne/fingerprint.{}.{}.3d.tsv'.format(initial_dims, perplexity)))

    plt.figure(figsize=figsize)
    plt.scatter([y[0] for y in x_3d], [y[1] for y in x_3d])#, edgecolor='', s=pointsize, c=x_3d)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plot/fingerprint.{}.{}.png'.format(initial_dims, perplexity)))
    plt.close()

if __name__ == "__main__":
    main()
