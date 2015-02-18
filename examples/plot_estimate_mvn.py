"""
======================================================
Estimate Multivariate Normal Distribution from Samples
======================================================

The maximum likelihood estimate (MLE) of an MVN can be computed directly. Then
we can sample from the estimated distribution or compute the marginal
distributions.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from gmr.utils import check_random_state
from gmr import MVN, plot_error_ellipse


random_state = check_random_state(0)
mvn = MVN(random_state=random_state)
X = random_state.multivariate_normal([0.0, 1.0], [[0.5, 1.5], [1.5, 5.0]],
                                     size=(100,))
mvn.from_samples(X)
X_sampled = mvn.sample(n_samples=100)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.xlim((-10, 10))
plt.ylim((-10, 10))
plot_error_ellipse(plt.gca(), mvn)
plt.scatter(X[:, 0], X[:, 1], c="g", label="Training data")
plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c="r", label="Samples")
plt.title("Bivariate Gaussian")
plt.legend(loc="best")

x = np.linspace(-10, 10, 100)
plt.subplot(1, 3, 2)
plt.xticks(())
marginalized = mvn.marginalize(np.array([1]))
plt.plot(marginalized.to_probability_density(x[:, np.newaxis]), x)
plt.title("Marginal distribution over y")

plt.subplot(1, 3, 3)
plt.yticks(())
marginalized = mvn.marginalize(np.array([0]))
plt.plot(x, marginalized.to_probability_density(x[:, np.newaxis]))
plt.title("Marginal distribution over x")

plt.show()
