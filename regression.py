import numpy as np
import matplotlib.pyplot as plt
from multivariate_normal import MultivariateNormal


if __name__ == "__main__":
    mvn = MultivariateNormal(random_state=0)

    n_samples = 10
    X = np.ndarray((n_samples, 2))
    X[:, 0] = np.linspace(0, np.pi, n_samples)
    X[:, 1] = 1 - 3 * X[:, 0] + np.random.randn(n_samples)
    mvn.from_samples(X)

    plt.scatter(X[:, 0], X[:, 1])
    X_test = np.linspace(0, np.pi, 100)
    mean, covariance = mvn.predict(np.array([0]), X_test[:, np.newaxis])
    y = mean.ravel()
    s = covariance.ravel()
    plt.plot(X_test, y)
    plt.fill_between(X_test, y - s, y + s, alpha=0.2)
    plt.show()
