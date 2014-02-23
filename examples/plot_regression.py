import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from gmr import MVN, GMM, plot_error_ellipses


if __name__ == "__main__":
    random_state = check_random_state(0)

    n_samples = 10
    X = np.ndarray((n_samples, 2))
    X[:, 0] = np.linspace(0, 2 * np.pi, n_samples)
    X[:, 1] = 1 - 3 * X[:, 0] + random_state.randn(n_samples)

    mvn = MVN(random_state=0)
    mvn.from_samples(X)

    plt.scatter(X[:, 0], X[:, 1])
    X_test = np.linspace(0, 2 * np.pi, 100)
    mean, covariance = mvn.predict(np.array([0]), X_test[:, np.newaxis])
    y = mean.ravel()
    s = covariance.ravel()
    plt.plot(X_test, y)
    plt.fill_between(X_test, y - s, y + s, alpha=0.2)

    n_samples = 100
    X = np.ndarray((n_samples, 2))
    X[:, 0] = np.linspace(0, 2 * np.pi, n_samples)
    X[:, 1] = np.sin(X[:, 0]) + random_state.randn(n_samples) * 0.1

    plt.figure()
    gmm = GMM(n_components=3, random_state=0)
    gmm.from_samples(X)
    Y = gmm.predict(np.array([0]), X_test[:, np.newaxis])
    plt.plot(X_test, Y.ravel())
    plt.scatter(X[:, 0], X[:, 1])
    plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b", "y"])

    plt.show()
