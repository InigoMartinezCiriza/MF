# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: <alberto.suarez@uam.es>
"""
# Load packages

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance
from numpy.random import Generator


def rbf_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    """Vectorized RBF kernel (covariance) function.

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
    >>> import gaussian_processes as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    [[3.         2.88236832]
     [2.88236832 3.        ]
     [2.55643137 2.88236832]]

    """
    d = distance.cdist(X, X_prime, metric='euclidean')
    return A * np.exp(-0.5 * (d / ls)**2)


def simulate_gp(
    t: np.ndarray,
    mean_fn: Callable[[np.ndarray], np.ndarray],
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    M: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a Gaussian process.

        X(t) ~ GP(mean_fn,kernel_fn)

    Parameters
    ----------
    t :
        Times at which the process is monitored.

    mean_fn:
        Mean function of the Gaussian process (vectorized).

    kernel_fn:
        Covariance functions of the Gaussian process (vectorized).

    M :
        Number of trajectories that are simulated.

    Returns
    -------
    X:
        Simulated trajectories as an np.ndarray with M rows and len(t) columns.
        Each trajectory is a row of the matrix consisting of the
        values of the process for each value of t.

    mean_vector:
        Vector with the values of the mean for each value of t.
        It is a np.ndarray with len(t) columns.

    kernel_matrix:
        Kernel matrix as an np.ndarray with len(t) rows and len(t)  columns.


    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_processes as gp
    >>> def mean_fn(t):
    ...     return np.zeros(np.shape(t))
    >>> def BB_kernel(s,t):
    ...     return (np.minimum(s,t) - s * t)
    >>> M, N  = (20, 1000)
    >>> t0, t1 = (0.0, 1.0)
    >>> t = np.linspace(t0, t1, N)
    >>> BB, _, _ = gp.simulate_gp(t, mean_fn, BB_kernel, M)
    >>> _ = plt.plot(t, BB.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('BB(t)')
    >>> _= plt.title('Standard Brownian Bridge process')
    >>> plt.show()
    """
    #  NOTE Use np.meshgrid for the arguments of
    #  kernel_fn to compute the kernel matrix.
    #  Do not use numpy.random.multivariate_normal
    #  Use np.linalg.svd
    #

    # Compute mean vector
    mean_vector = mean_fn(t)
    
    # Compute kernel matrix using np.meshgrid
    S, T = np.meshgrid(t, t, indexing='ij')
    kernel_matrix = kernel_fn(S, T)
    
    # Perform SVD decomposition of kernel matrix
    U, S, Vt = np.linalg.svd(kernel_matrix)
    
    # Generate standard normal samples
    Z = np.random.randn(M, len(t))
    
    # Compute square root of kernel matrix using SVD
    sqrt_K = U @ np.diag(np.sqrt(S))
    
    # Generate GP samples
    X = mean_vector + (sqrt_K @ Z.T).T
    
    return X, mean_vector, kernel_matrix


def simulate_conditional_gp(
    t: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    mean_fn: Callable[[np.ndarray], np.ndarray],
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    M: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a Gaussian process conditined to observed values.

        X(t) ~ GP(mean_fn,kernel_fn)

        condition to having observed  X(t_obs) = x_obs at t_obs

    Parameters
    ----------
    t :
        Times at which the process is monitored.

    t_obs :
        Times at which the values of the process have been observed.
        The Gaussian process has the value x_obs at t_obs.

    x_obs :
        Values of the process at t_obs.

    mean_fn :
        Mean function of the Gaussian process [vectorized].

    kernel_fn :
        Covariance functions of the Gaussian process.

    M :
        Number of trajectories in the simulation.

    Returns
    -------
    X:
        Simulated trajectories as an np.ndarray with M rows and len(t) columns.
        Each trajectory is a row of the matrix consisting of the
        values of the process for each value of t.

    mean_vector:
        Vector with the values of the mean for each value of t.
        It is a np.ndarray with len(t) columns.

    kernel_matrix:
        Kernel matrix as an np.ndarray with len(t) rows and len(t)  columns.

    Example
    -------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_processes as gp
    >>> def mean_fn(t, mu=1.0):
    ...     return mu*t
    >>> def BB_kernel(s,t):
    ...     return np.minimum(s,t) - s * t
    >>> M, N  = (30, 1000)
    >>> t0, t1 = (0.0, 1.0)
    >>> t = np.linspace(t0, t1, N)
    >>> t_obs = np.array([0.25, 0.5, 0.75])
    >>> x_obs = np.array([0.3, -0.3, -1.0])
    >>> B, _, _ = gp.simulate_conditional_gp(
    ...     t,
    ...     t_obs,
    ...     x_obs,
    ...     mean_fn,
    ...     BB_kernel,
    ...     M,
    ... )
    >>> _ = plt.plot(t, B.T)
    >>> _ = plt.xlabel('t')
    >>> _ =  plt.ylabel('B(t)')

    """
    # NOTE Use 'multivariate_normal' from numpy with "'method = 'svd'".
    # 'svd' is slower, but numerically more robust than 'cholesky'
    
    # Compute mean vectors
    mean_vector = mean_fn(t)
    mean_obs = mean_fn(t_obs)
    
    # Compute kernel matrices
    S, T = np.meshgrid(t, t, indexing='ij')
    kernel_matrix = kernel_fn(S, T)
    
    S_obs, T_obs = np.meshgrid(t_obs, t_obs, indexing='ij')
    kernel_obs = kernel_fn(S_obs, T_obs)
    
    S_cross, T_cross = np.meshgrid(t, t_obs, indexing='ij')
    kernel_cross = kernel_fn(S_cross, T_cross)
    
    # Compute conditional mean and covariance
    kernel_obs_inv = np.linalg.inv(kernel_obs + 1e-6 * np.eye(len(t_obs)))
    conditional_mean = mean_vector + kernel_cross @ kernel_obs_inv @ (x_obs - mean_obs)
    conditional_cov = kernel_matrix - kernel_cross @ kernel_obs_inv @ kernel_cross.T
    
    # Create an instance of Generator
    rng = np.random.default_rng()

    # Sample from the conditional distribution
    X = rng.multivariate_normal(conditional_mean, conditional_cov, size=M, method='svd')
    
    return X, conditional_mean, kernel_matrix


def gp_regression(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    sigma2_noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Gaussian process regression.

    Parameters
    ----------
    X:
        :math:`N \times D` data matrix for training

    y:
        vector of output values

    X_test:
        :math:`L \times D` data matrix for testing.

    kernel_fn:
        Kernel (covariance) function.

    sigma2_noise:
        Variance of the noise.
        It is a hyperparameter of GP regression.

    Returns
    -------
        prediction_mean:
            Predictions at the test points.

        prediction_variance:
            Uncertainty of the predictions.
    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_processes as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> y = [1, 2, 3]
    >>> X_test = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> sigma2_noise = 0.01
    >>> def kernel (X, X_prime):
    ...     return gp.rbf_kernel(X, X_prime, A, l)
    >>> predictions, _ = gp.gp_regression(X, y, X_test, kernel, sigma2_noise)
    >>> print(predictions)
    [1.00366515 2.02856104]
    """

    # NOTE use 'np.linalg.solve' instead of inverting the matrix.
    # This procedure is numerically more robust.

    # Compute kernel matrices
    K = kernel_fn(X, X) + sigma2_noise * np.eye(len(X))
    K_s = kernel_fn(X, X_test)
    K_ss = kernel_fn(X_test, X_test)
    
    # Solve for alpha using np.linalg.solve
    alpha = np.linalg.solve(K, y)
    
    # Compute predictive mean
    prediction_mean = K_s.T @ alpha
    
    # Compute predictive variance
    v = np.linalg.solve(K, K_s)
    prediction_variance = K_ss - K_s.T @ v
    
    return prediction_mean, prediction_variance


if __name__ == "__main__":
    import doctest
    doctest.testmod()
