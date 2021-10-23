"""
========================================================
Confidence Interval of a 1D Standard Normal Distribution
========================================================

We plot the 0.6827 confidence interval of a standard normal distribution in
one dimension. The confidence interval is marked by green lines and the
region outside of the confidence interval is marked by red lines.
"""
print(__doc__)
import matplotlib.pyplot as plt
import numpy as np
from gmr import MVN


mvn = MVN(mean=[0.0], covariance=[[1.0]])
alpha = 0.6827
X = np.linspace(-3, 3, 101)[:, np.newaxis]
P = mvn.to_probability_density(X)

for x, p in zip(X, P):
    conf = mvn.is_in_confidence_region(x, alpha)
    color = "g" if conf else "r"
    plt.plot([x[0], x[0]], [0, p], color=color)

plt.plot(X.ravel(), P)

plt.xlabel("x")
plt.ylabel("Probability Density $p(x)$")
plt.show()
