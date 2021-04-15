"""
==================================
Compare Initializations Strategies
==================================

Expectation Maximization for Gaussian Mixture Models does not have a unique
solution. The result depends on the initialization. It is particularly
important to either normalize the training data or set the covariances
accordingly. In addition, k-means++ initialization helps to find a good
initial distribution of means.

Here is a numerically challenging example in which we find a better
distribution of individual Gaussians with k-means++.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from gmr.utils import check_random_state
from gmr import GMM, plot_error_ellipses, kmeansplusplus_initialization, covariance_initialization


random_state = check_random_state(1)

n_samples = 300
n_features = 2
X = np.ndarray((n_samples, n_features))
X[:n_samples // 3, :] = random_state.multivariate_normal(
    [0.0, 1.0], [[0.5, -1.0], [-1.0, 5.0]], size=(n_samples // 3,))
X[n_samples // 3:-n_samples // 3, :] = random_state.multivariate_normal(
    [-2.0, -2.0], [[3.0, 1.0], [1.0, 1.0]], size=(n_samples // 3,))
X[-n_samples // 3:, :] = random_state.multivariate_normal(
    [3.0, 1.0], [[3.0, -1.0], [-1.0, 1.0]], size=(n_samples // 3,))

# artificial scaling, makes standard implementation fail
# either the initial covariances have to be adjusted or we have
# to normalize the dataset
X[:, 1] *= 10000.0

plt.figure(figsize=(10, 10))

n_components = 3
initial_covs = np.empty((n_components, n_features, n_features))
initial_covs[:] = np.eye(n_features)
gmm = GMM(n_components=n_components, random_state=random_state)
gmm.from_samples(X, init_params="random", n_iter=0)

plt.subplot(2, 2, 1)
plt.title("Default initialization")
plt.xlim((-10, 10))
plt.ylim((-100000, 100000))
plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"], alpha=0.15)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(gmm.means[:, 0], gmm.means[:, 1], color=["r", "g", "b"])

gmm.from_samples(X)

plt.subplot(2, 2, 2)
plt.title("Trained Gaussian Mixture Model")
plt.xlim((-10, 10))
plt.ylim((-100000, 100000))
plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"], alpha=0.15)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(gmm.means[:, 0], gmm.means[:, 1], color=["r", "g", "b"])

initial_means = kmeansplusplus_initialization(X, n_components, random_state)
initial_covs = covariance_initialization(X, n_components)
gmm = GMM(n_components=n_components, random_state=random_state)
gmm.from_samples(X, init_params="kmeans++", n_iter=0)

plt.subplot(2, 2, 3)
plt.title("k-means++ and inital covariance scaling")
plt.xlim((-10, 10))
plt.ylim((-100000, 100000))
plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"], alpha=0.15)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(gmm.means[:, 0], gmm.means[:, 1], color=["r", "g", "b"])

gmm.from_samples(X)

plt.subplot(2, 2, 4)
plt.title("Trained Gaussian Mixture Model")
plt.xlim((-10, 10))
plt.ylim((-100000, 100000))
plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"], alpha=0.15)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(gmm.means[:, 0], gmm.means[:, 1], color=["r", "g", "b"])

plt.show()
