import numpy as np

from scipy.optimize import minimize_scalar
from scipy.spatial import distance_matrix


def generate_similarity_space(X, y, sigma, kernel="gaussian"):
    """
    Generate a similarity space using a specified kernel function.

    Parameters:
    X : ndarray of shape (n_samples, n_features)
        The input data points.
    y : ndarray of shape (n_samples,)
        The class labels corresponding to each data point.
    sigma : float
        The width parameter for the kernel function. Must be greater than zero.
    kernel : str, optional (default="gaussian")
        The kernel function to use. Options are "gaussian" or "laplacian".

    Returns:
    similarities : ndarray of shape (n_samples, n_classes)
        The computed similarity values for each sample with respect to each class.
    """

    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero")

    dist_matrix = distance_matrix(X, X)
    if kernel == "gaussian":
        gram_matrix = np.exp(-.5 * (dist_matrix**2)/(sigma**2))
    elif kernel == "laplacian":
        gram_matrix = np.exp(-.5 * np.abs(dist_matrix)/sigma)
    else:
        raise ValueError(
            "Unsupported kernel type. Use 'gaussian' or 'laplacian'.")

    classes = sorted(np.unique(y))
    N = gram_matrix.shape[0]

    gram_matrix = gram_matrix * (1 - np.eye(N))

    similarities = [
        np.sum(gram_matrix[
            :, np.where(y == c)[0]
        ], axis=1)/(N-1) for c in classes
    ]

    similarities = np.array(similarities).transpose()
    return similarities


def similarity_distance_loss(X, y, sigma, kernel="gaussian"):
    """
    Compute the RBF loss based on similarity space.

    Parameters:
    X : ndarray of shape (n_samples, n_features)
        The input data points.
    y : ndarray of shape (n_samples,)
        The class labels corresponding to each data point.
    sigma : float
        The width parameter for the kernel function.
    kernel : str, optional (default="gaussian")
        The kernel function to use. Options are "gaussian" or "laplacian".

    Returns:
    float
        The negative sum of the upper triangular part of the distance matrix
        between class mean similarities.
    """
    if sigma == 0.:
        return 0.

    classes = sorted(np.unique(y))
    similarities = generate_similarity_space(X, y, sigma, kernel)
    mean_sims = np.array([
        similarities[
            np.where(y == c)[0]
        ].mean(axis=0) for c in classes
    ])

    return -1. * np.triu(
        distance_matrix(mean_sims, mean_sims)
    ).sum()


def optimize_width(X, y, kernel="gaussian"):
    """
    Optimize the sigma (bwidth) parameter for the RBF kernel to minimize
    the similarity distance loss.

    Parameters:
    X : ndarray of shape (n_samples, n_features)
        The input data points.
    y : ndarray of shape (n_samples,)
        The class labels corresponding to each data point.
    kernel : str, optional (default="gaussian")
        The kernel function to use. Options are "gaussian" or "laplacian".

    Returns:
    float
        The optimal value of sigma found by the optimizer.
    """
    def fun(sigma): return similarity_distance_loss(
        X, y, sigma, kernel=kernel
    )
    result = minimize_scalar(fun)
    return result.x
