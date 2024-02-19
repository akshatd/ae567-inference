from typing import Callable, Any
import numpy as np

def compute_normalizing_constant(data: np.ndarray,
                                 model: Any,
                                 prior: Callable[[np.ndarray], np.ndarray],
                                 param_vals: np.ndarray):
    """Compute normalizing constant
         -
        /
        |  P(data | \theta) P(\theta) d \theta.
       _/

    Args:
        Model: model defining the sampling distribution
        data: the observed data
        prior: a function returning the pdf for the prior
        param_vals: a set of inputs at which to discretize the above integral

    Notes:

        Uses trapezoid rule on a fine discretization of theta
    """

    prior_vals = prior(param_vals)
    like_vals = np.array([model.likelihood(data, p) for p in param_vals])

    unnormalized_posterior = prior_vals * like_vals

    # Compute the integral using trapezoid rule
    constant = np.trapz(unnormalized_posterior, param_vals)
    return constant, unnormalized_posterior

def compute_posterior(data: np.ndarray,
                      model: Any,
                      prior: Callable[[np.ndarray], np.ndarray],
                      param_vals: np.ndarray):
    r"""Compute the posterior over a range of parameters"""
    Z, unnormalized_posterior = compute_normalizing_constant(data, model, prior, param_vals)
    return unnormalized_posterior / Z


def func_inv(xvals: np.ndarray, cdf_vals: np.ndarray, u: np.ndarray) -> np.ndarray:
    """A function to compute the inverse cdf from a discrete set of values

    Args:

        xvals: (N, ) array of locations at which cdf_vals is obtained (sorted)
        cdf_vals: (N, ) array of values of the cdf (sorted)
        u: (M, )Locations at which to evaluate the inverse CDF. an array of numbers between 0 and 1.

    Returns:
        ret: (M, ) array of locations of the inverse CDF
    """

    assert np.all(u < 1) and np.all(u > 0)

    M = u.shape[0]
    ret = np.zeros((M))
    for jj in range(M):
        inds = (cdf_vals - u[jj] > 0).nonzero()
        ret[jj] = xvals[inds[0][0]]
    return ret