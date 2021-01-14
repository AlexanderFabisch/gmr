"""
========================================
Sample from Confidence Interval of a MVN
========================================

Sometimes we want to avoid sampling regions of low probability. We will see
how this can be done in this example. We compare unconstrained sampling with
sampling from the 95.45 % and 68.27 % confidence regions. In a one-dimensional
Gaussian these would correspond to the 2-sigma and sigma intervals
respectively.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from gmr import MVN, plot_error_ellipse


random_state = np.random.RandomState(100)
mvn = MVN(
    mean=np.array([0.0, 0.0]),
    covariance=np.array([[1.0, 2.0], [2.0, 9.0]]),
    random_state=random_state)

n_samples = 1000

plt.figure(figsize=(15, 5))

ax = plt.subplot(131)
ax.set_title("Unconstrained Sampling")
samples = mvn.sample(n_samples)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.9, s=1, label="Samples")
plot_error_ellipse(ax, mvn, factors=(1.0, 2.0), color="orange")
ax.set_xlim((-5, 5))
ax.set_ylim((-10, 10))

ax = plt.subplot(132)
ax.set_title(r"95.45 % Confidence Region ($2\sigma$)")
samples = mvn.sample_confidence_region(n_samples, 0.9545)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.9, s=1, label="Samples")
plot_error_ellipse(ax, mvn, factors=(1.0, 2.0), color="orange")
ax.set_xlim((-5, 5))
ax.set_ylim((-10, 10))

ax = plt.subplot(133)
ax.set_title(r"68.27 % Confidence Region ($\sigma$)")
samples = mvn.sample_confidence_region(n_samples, 0.6827)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.9, s=1, label="Samples")
plot_error_ellipse(ax, mvn, factors=(1.0, 2.0), color="orange")
ax.set_xlim((-5, 5))
ax.set_ylim((-10, 10))
ax.legend()

plt.show()