"""
========================================
Sample from Confidence Interval of a GMM
========================================

Sometimes we want to avoid sampling regions of low probability. We will see
how this can be done in this example. We compare unconstrained sampling with
sampling from the 95.45 % and 68.27 % confidence regions. In a one-dimensional
Gaussian these would correspond to the 2-sigma and sigma intervals
respectively.
"""
import numpy as np
import matplotlib.pyplot as plt
from gmr import GMM, plot_error_ellipses


random_state = np.random.RandomState(100)
gmm = GMM(
    n_components=2,
    priors=np.array([0.2, 0.8]),
    means=np.array([[0.0, 0.0], [0.0, 5.0]]),
    covariances=np.array([[[1.0, 2.0], [2.0, 9.0]], [[9.0, 2.0], [2.0, 1.0]]]),
    random_state=random_state)

n_samples = 1000

plt.figure(figsize=(15, 5))

ax = plt.subplot(131)
ax.set_title("Unconstrained Sampling")
samples = gmm.sample(n_samples)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.9, s=1, label="Samples")
plot_error_ellipses(ax, gmm, factors=(1.0, 2.0), colors=["orange", "orange"])
ax.set_xlim((-8, 8))
ax.set_ylim((-10, 10))

ax = plt.subplot(132)
ax.set_title(r"95.45 % Confidence Region ($2\sigma$)")
samples = gmm.sample_confidence_region(n_samples, 0.9545)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.9, s=1, label="Samples")
plot_error_ellipses(ax, gmm, factors=(1.0, 2.0), colors=["orange", "orange"])
ax.set_xlim((-5, 5))
ax.set_ylim((-10, 10))

ax = plt.subplot(133)
ax.set_title(r"68.27 % Confidence Region ($\sigma$)")
samples = gmm.sample_confidence_region(n_samples, 0.6827)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.9, s=1, label="Samples")
plot_error_ellipses(ax, gmm, factors=(1.0, 2.0), colors=["orange", "orange"])
ax.set_xlim((-5, 5))
ax.set_ylim((-10, 10))
ax.legend()

plt.show()