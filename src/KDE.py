'''
Author: Rajdeep Pathak
Date: July 8, 2025
If you use this implementation, we request you to kindly cite our paper:
'''

import cupy as cp
import numpy as np
from math import sqrt, pi
import gc

class KDE_GPU:
    """
    Simple Kernel Density Estimation on GPU.
    """
    
    def __init__(self, bandwidth=None, rule=None, kernel='gaussian'):
        """
        bandwidth: float or None. If None, use rule to compute bandwidth.
        rule: None, 'scott', or 'silverman'
        kernel: 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', or 'cosine'
        """
        self.bandwidth = bandwidth
        self.rule = rule
        self.kernel = kernel
        self.data = None

    def fit(self, data):
        """Fit the Kernel Density model on the data. 
        If bandwidth = 'scott' or 'silverman', then the bandwidths are computed on GPU.

        Parameters
        ----------
        data: array-like of shape (n_samples, n_features).
            List of n_features-dimensional data points. Each row corresponds to a single data point.
        """
        self.data = cp.asarray(data)
        n = self.data.size
        if self.bandwidth is None and self.rule is not None:
            if self.rule == 'scott':
                # based on scikit-learn's implementation
                if len(self.data.shape) == 1:
                    self.bandwidth = n ** (-1/5)
                else:
                    self.bandwidth = n ** (-1/(self.data.shape[1]+4)) 
            elif self.rule == 'silverman':
                # based on scikit-learn's implementation
                if len(self.data.shape) == 1:
                    self.bandwidth = (n * 3/4) ** (-1/5)
                else:
                    self.bandwidth = (n * (self.data.shape[1] + 2) / 4) ** (-1 / (self.data.shape[1] + 4)) 
            else:
                raise ValueError("Unknown rule: choose 'scott' or 'silverman'")

    def _kernel_func(self, x):
        # All kernels assume input x is a CuPy array
        abs_x = cp.abs(x)
        if self.kernel == 'gaussian':
            return (1 / sqrt(2 * pi)) * cp.exp(-0.5 * x**2)
        elif self.kernel == 'tophat':
            return 0.5 * (abs_x <= 1)
        elif self.kernel == 'epanechnikov':
            return 0.75 * (1 - x**2) * (abs_x <= 1)
        elif self.kernel == 'exponential':
            return 0.5 * cp.exp(-abs_x)
        elif self.kernel == 'linear':
            return (1 - abs_x) * (abs_x <= 1)
        elif self.kernel == 'cosine':
            return (cp.pi / 4) * cp.cos((cp.pi / 2) * x) * (abs_x <= 1)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def evaluate(self, points, batch_size=1000):
        """
        Evaluate the densities under the fitted KDE at each test point.
        
        Parameters
        ----------
        points: array-like of shape (n_samples, n_features)
                An array of points to query. Last dimension should match dimension of training data (n_features).
        batch_size: process the points in smaller batches to avoid GPU memory outage.

        Returns
        -------
        densities: ndarray of shape (n_samples,)
                   Density of each point under the fitted KDE.
        """
        points = cp.asarray(points, dtype=cp.float32)
        n = self.data.size
        bw = self.bandwidth
        densities = []
        for start in range(0, len(points), batch_size):
            end = start + batch_size
            batch_points = points[start:end]
            diffs = (batch_points[:, None] - self.data[None, :]) / bw
            kernels = self._kernel_func(diffs)
            batch_density = cp.sum(kernels, axis=1) / (n * bw)
            densities.append(cp.asnumpy(batch_density))

        # Free GPU memory
        del batch_points, diffs, kernels, batch_density
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        
        return np.concatenate(densities)
