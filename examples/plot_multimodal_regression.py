"""
=====================
Multimodal Regression
=====================

In multimodal regression we do not try to fit a function f(x) = y but a
probability distribution p(y|x) with more than one peak in the probability
density function.

The dataset that we use to illustrate multimodal regression by Gaussian
mixture regression is from Section 5 of

C. M. Bishop, "Mixture Density Networks", 1994,
https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf

On the left side of the figure you see the training data and the fitted
GMM indicated by ellipses corresponding to its components. On the right
side you see the predicted probability density p(y|x=0.5). There are
three peaks that correspond to three different valid predictions. Each
peak is represented by at least one of the Gaussians of the GMM.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from gmr import GMM, plot_error_ellipses


def f(y, random_state):
    eps = random_state.rand(*y.shape) * 0.2 - 0.1
    return y + 0.3 * np.sin(2.0 * np.pi * y) + eps


y = np.linspace(0, 1, 1000)
random_state = np.random.RandomState(3)
x = f(y, random_state)

XY_train = np.column_stack((x, y))
gmm = GMM(n_components=4, random_state=random_state)
gmm.from_samples(XY_train)

plt.figure(figsize=(10, 5))

ax = plt.subplot(121)
ax.set_title("Dataset and GMM")
ax.scatter(x, y, s=1)
colors = ["r", "g", "b", "orange"]
plot_error_ellipses(ax, gmm, colors=colors)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax = plt.subplot(122)
ax.set_title("Conditional Distribution")
Y = np.linspace(0, 1, 1000)
Y_test = Y[:, np.newaxis]
X_test = 0.5
conditional_gmm = gmm.condition([0], [X_test])
p_of_Y = conditional_gmm.to_probability_density(Y_test)
ax.plot(Y, p_of_Y, color="k", label="GMR", lw=3)
for component_idx in range(conditional_gmm.n_components):
    p_of_Y = (conditional_gmm.priors[component_idx]
              * conditional_gmm.extract_mvn(
                component_idx).to_probability_density(Y_test))
    ax.plot(Y, p_of_Y, color=colors[component_idx],
            label="Component %d" % (component_idx + 1))
ax.set_xlabel("y")
ax.set_ylabel("$p(y|x=%.1f)$" % X_test)
ax.legend(loc="best")

plt.tight_layout()
plt.show()
