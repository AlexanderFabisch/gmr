"""
============================================
Estimate Gaussian Mixture Model from Samples
============================================

The maximum likelihood estimate (MLE) of a GMM cannot be computed directly.
Instead, we have to use expectation-maximization (EM). Then we can sample from
the estimated distribution or compute conditional distributions.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from gmr.utils import check_random_state
from gmr import GMM, plot_error_ellipses, kmeansplusplus_initialization


random_state = check_random_state(0)

n_samples = 300
n_features = 2
X = np.ndarray((n_samples, n_features))
X[:n_samples // 3, :] = random_state.multivariate_normal(
    [0.0, 1.0], [[0.5, -1.0], [-1.0, 5.0]], size=(n_samples // 3,))
X[n_samples // 3:-n_samples // 3, :] = random_state.multivariate_normal(
    [-2.0, -2.0], [[3.0, 1.0], [1.0, 1.0]], size=(n_samples // 3,))
X[-n_samples // 3:, :] = random_state.multivariate_normal(
    [3.0, 1.0], [[3.0, -1.0], [-1.0, 1.0]], size=(n_samples // 3,))

initial_means = kmeansplusplus_initialization(X, 3, random_state)
gmm = GMM(n_components=3, means=np.copy(initial_means),
          random_state=random_state)
gmm.from_samples(X)
cond = gmm.condition(np.array([0]), np.array([1.0]))

plt.figure(figsize=(5, 5))

plt.subplot(1, 1, 1)
plt.title("Gaussian Mixture Model")
plt.xlim((-10, 10))
plt.ylim((-10, 10))
plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"], alpha=0.15)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(initial_means[:, 0], initial_means[:, 1], s=100,
            label="Initial means")
plt.legend(loc="best")

plt.show()
