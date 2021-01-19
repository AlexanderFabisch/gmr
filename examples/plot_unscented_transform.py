"""
===================
Unscented Transform
===================

The `unscented transform <https://en.wikipedia.org/wiki/Unscented_transform>`_
can be used to to transform a Gaussian distribution through a nonlinear
function.

In this example we transform an MVN from 2D Cartesian coordinates (1) to polar
coordinates (2) and back to Cartesian coordinates (3). It does not work
perfectly because the transformations are highly nonlinear.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from gmr import MVN, plot_error_ellipse


def cartesian_to_polar(X):
    Y = np.empty_like(X)
    Y[:, 0] = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    Y[:, 1] = np.arctan2(X[:, 1], X[:, 0])
    return Y


def polar_to_cartesian(Y):
    X = np.empty_like(Y)
    X[:, 0] = Y[:, 0] * np.cos(Y[:, 1])
    X[:, 1] = Y[:, 0] * np.sin(Y[:, 1])
    return X


plt.figure(figsize=(12, 4))

# parameters of unscented transform, these are the defaults:
alpha = 1e-3
beta = 2.0  # lower values give better estimates
kappa = 0.0

ax = plt.subplot(131)
ax.set_title("(1) Cartesian coordinates")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim((-8, 8))
ax.set_ylim((-8, 8))
mvn_cartesian = MVN(
    mean=np.array([2.5, 1.3]),
    covariance=np.array([[1.0, -1.5], [-1.5, 4.0]]),
    random_state=0)
plot_error_ellipse(ax, mvn_cartesian)
samples_cartesian = mvn_cartesian.sample(1000)
ax.scatter(samples_cartesian[:, 0], samples_cartesian[:, 1], s=1)

ax = plt.subplot(132)
ax.set_title("(2) Polar coordinates")
ax.set_xlabel("$r$")
ax.set_ylabel("$\phi$")
ax.set_xlim((-8, 8))
ax.set_ylim((-8, 8))
sigma_points_cartesian = mvn_cartesian.sigma_points(alpha=alpha, kappa=kappa)
sigma_points_polar = cartesian_to_polar(sigma_points_cartesian)
mvn_polar = mvn_cartesian.estimate_from_sigma_points(sigma_points_polar, alpha=alpha, beta=beta, kappa=kappa)
plot_error_ellipse(ax, mvn_polar)
samples_polar = cartesian_to_polar(samples_cartesian)
ax.scatter(samples_polar[:, 0], samples_polar[:, 1], s=1)

ax = plt.subplot(133)
ax.set_title("(3) Cartesian coordinates")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim((-8, 8))
ax.set_ylim((-8, 8))
sigma_points_polar2 = mvn_polar.sigma_points(alpha=alpha, kappa=kappa)
sigma_points_cartesian2 = polar_to_cartesian(sigma_points_polar2)
mvn_cartesian2 = mvn_polar.estimate_from_sigma_points(sigma_points_cartesian2, alpha=alpha, beta=beta, kappa=kappa)
plot_error_ellipse(ax, mvn_cartesian2)
samples_cartesian2 = polar_to_cartesian(samples_polar)
ax.scatter(samples_cartesian2[:, 0], samples_cartesian2[:, 1], s=1)

plt.tight_layout()
plt.show()
