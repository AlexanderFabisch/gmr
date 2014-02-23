import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from gmr import MVN, plot_error_ellipse


if __name__ == "__main__":
    random_state = check_random_state(0)
    mvn = MVN(random_state=random_state)
    X = random_state.multivariate_normal([0.0, 1.0], [[0.5, -2.0], [-2.0, 5.0]],
                                         size=(10000,))
    mvn.from_samples(X)
    print(mvn.to_moments())
    print(mvn.to_probability_density(X))
    X = mvn.sample(n_samples=100)

    plt.figure()
    plot_error_ellipse(plt.gca(), mvn)
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))

    plt.figure()
    x = np.linspace(-5, 5, 100)
    marginalized = mvn.marginalize(np.array([0]))
    plt.plot(x, marginalized.to_probability_density(x[:, np.newaxis]))

    plt.figure()
    for x in np.linspace(-2, 2, 100):
        conditioned = mvn.condition(np.array([0]), np.array([x]))
        y = np.linspace(-6, 6, 100)
        plt.plot(y, conditioned.to_probability_density(y[:, np.newaxis]).ravel())

    plt.show()