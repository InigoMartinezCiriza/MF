from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance


def linear_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
) -> np.ndarray:
    return X @ X_prime.T


def exponential_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    l: float
) -> np.ndarray:
    d = distance.cdist(X, X_prime, metric='minkovski', p=1.0)
    return A * np.exp(- d / l)


def rbf_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_prime:
        Data matrix
    A:
        Output variance
    ls:
        Kernel lengthscale

    Returns
    -------
    kernel matrix

    Notes
    -------
    Alternative parametrization (e.g. en sklearn)
    gamma = 0.5 / ls**2

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_process_regression as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    """
    d = distance.cdist(X, X_prime, metric='euclidean')
    return A * np.exp(-0.5 * (d / ls)**2)


def kernel_pca(
    X: np.ndarray,
    X_test: np.ndarray,
    kernel: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_test:
        data matrix
    A:
        output variance
    ls:
        kernel lengthscale

    Returns
    -------
    X_test_hat:
        Projection of X_test on the principal components
    lambda_eigenvals:
        Eigenvalues of the centered kernel
    alpha_eigenvecs:
        Principal components. These are the eigenvectors
        of the centered kernel with the RKHS normalization

    Notes
    -------
    In the corresponding method of sklearn the eigenvectors
    are normalized in l2.

    """
    # Number of training samples
    n = X.shape[0]

    # Compute the kernel matrix on the training data
    K = kernel(X, X)

    # Center the kernel matrix
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # Perform eigen-decomposition on the centered kernel matrix
    eigenvals, eigenvecs = np.linalg.eigh(K_centered)

    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Remove components with near-zero eigenvalues to avoid numerical issues
    tol = 1e-12
    nonzero_indices = eigenvals > tol
    lambda_eigenvals = eigenvals[nonzero_indices]
    alpha_eigenvecs = eigenvecs[:, nonzero_indices]

    # Normalization
    alpha_eigenvecs = alpha_eigenvecs / np.sqrt(lambda_eigenvals)

    # Compute the kernel between test data and training data
    K_test = kernel(X_test, X)

    # Center the test kernel matrix using the training kernel means
    # Compute the mean of the training kernel (over columns) and total mean
    K_train_mean = np.mean(K, axis=0)
    K_total_mean = np.mean(K)
    K_test_mean = np.mean(K_test, axis=1, keepdims=True)
    K_test_centered = K_test - K_train_mean - K_test_mean + K_total_mean

    # Project the test data onto the principal components.
    X_test_hat = K_test_centered @ alpha_eigenvecs

    return X_test_hat, lambda_eigenvals, alpha_eigenvecs