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
from gmr import GMM, plot_error_ellipses


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

gmm = GMM(n_components=3, random_state=random_state)
gmm.from_samples(X)
cond = gmm.condition(np.array([0]), np.array([1.0]))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Gaussian Mixture Model")
plt.xlim((-10, 10))
plt.ylim((-10, 10))
plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"])
plt.scatter(X[:, 0], X[:, 1])

plt.subplot(1, 3, 2)
plt.title("Probability Density and Samples")
plt.xlim((-10, 10))
plt.ylim((-10, 10))
x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.vstack((x.ravel(), y.ravel())).T
p = gmm.to_probability_density(X_test)
p = p.reshape(*x.shape)
plt.contourf(x, y, p)
X_sampled = gmm.sample(100)
plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c="r")

plt.subplot(1, 3, 3)
plt.title("Conditional PDF $p(y | x = 1)$")
X_test = np.linspace(-10, 10, 100)
plt.plot(X_test, cond.to_probability_density(X_test[:, np.newaxis]))

plt.show()
