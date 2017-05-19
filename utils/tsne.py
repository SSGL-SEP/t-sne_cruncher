#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab as plot


def h_beta(d=np.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    p = np.exp(-d.copy() * beta)
    sum_p = sum(p)
    h = np.log(sum_p) + beta * np.sum(d * p) / sum_p
    p = p / sum_p
    return h, p


def x2p(x=np.array([]), tol=1e-5, perplexity=30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = x.shape
    sum_x = np.sum(np.square(x), 1)
    d = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    p = np.zeros((n, n))
    beta = np.ones((n, 1))
    log_u = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        di = d[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = h_beta(di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        hdiff = H - log_u
        tries = 0
        while np.abs(hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = h_beta(di, beta[i])
            hdiff = H - log_u
            tries = tries + 1

        # Set the final row of P
        p[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return p


def pca(x=np.array([]), no_dims=50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    print("Preprocessing the data using PCA...")
    (n, d) = x.shape
    x = x - np.tile(np.mean(x, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(x.T, x))
    y = np.dot(x, M[:, 0:no_dims])
    return y


def tsne(x=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    x = pca(x, initial_dims).real
    (n, d) = x.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    y = np.random.randn(n, no_dims)
    d_y = np.zeros((n, no_dims))
    i_y = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    p = x2p(x, 1e-5, perplexity)
    p = p + np.transpose(p)
    p = p / np.sum(p)
    p = p * 4  # early exaggeration
    p = np.maximum(p, 1e-12)

    # Run iterations
    for it in range(max_iter):

        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        q = num / np.sum(num)
        q = np.maximum(q, 1e-12)

        # Compute gradient
        p_q = p - q
        for i in range(n):
            d_y[i, :] = np.sum(np.tile(p_q[:, i] * num[:, i], (no_dims, 1)).T * (y[i, :] - y), 0)

        # Perform the update
        if it < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((d_y > 0) != (i_y > 0)) + (gains * 0.8) * ((d_y > 0) == (i_y > 0))
        gains[gains < min_gain] = min_gain
        i_y = momentum * i_y - eta * (gains * d_y)
        y = y + i_y
        y = y - np.tile(np.mean(y, 0), (n, 1))

        # Compute current value of cost function
        if (it + 1) % 10 == 0:
            c = np.sum(p * np.log(p / q))
            print("Iteration ", (it + 1), ": error is ", c)

        # Stop lying about P-values
        if it == 100:
            p = p / 4

    # Return solution
    return y


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 50, 20.0)
    plot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plot.show()
