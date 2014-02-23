import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import pinvh


def invert_indices(n_features, indices):
    inv = np.ones(n_features, dtype=np.bool)
    inv[indices] = False
    inv, = np.where(inv)
    return inv


class MultivariateNormal(object):
    """Multivariate normal distribution.

    Some utility functions to deal with MVNs. See
    http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    for more details.
    """
    def __init__(self, mean=None, covariance=None, random_state=None):
        self.mean = mean
        self.covariance = covariance
        self.random_state = check_random_state(random_state)

    def from_samples(self, X, bessels_correction=True):
        """MLE of the mean and covariance."""
        self.mean = np.mean(X, axis=0)
        self.covariance = np.cov(X, rowvar=0,
                                 bias=0 if bessels_correction else 1)

    def to_moments(self):
        """Get mean and covariance."""
        return self.mean, self.covariance

    def sample(self, n_samples):
        """Sample from multivariate normal distribution."""
        return self.random_state.multivariate_normal(
            self.mean, self.covariance, size=(n_samples,))

    def to_probability_density(self, X):
        """Compute probability density."""
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        precision = pinvh(self.covariance)
        d = X - self.mean
        normalization = 1 / np.sqrt((2 * np.pi) ** n_features *
                                    np.linalg.det(self.covariance))
        p = np.ndarray(n_samples)
        for n in range(n_samples):
            p[n] = normalization * np.exp(-0.5 * d[n].dot(precision).dot(d[n]))
        return p

    def marginalize(self, indices):
        """Marginalize over given indices."""
        return MultivariateNormal(
            mean=self.mean[indices],
            covariance=self.covariance[np.ix_(indices, indices)])

    def condition(self, indices, x):
        """Conditional distribution over given indices."""
        mean, covariance = self._condition(invert_indices(self.mean.shape[0],
                                                          indices), indices, x)
        return MultivariateNormal(mean=mean, covariance=covariance,
                                  random_state=self.random_state)

    def predict(self, indices, X):
        """Predict means and covariance of posteriors."""
        return self._condition(invert_indices(self.mean.shape[0], indices),
                               indices, X)

    def _condition(self, i1, i2, X):
        cov_12 = self.covariance[np.ix_(i1, i2)]
        cov_11 = self.covariance[np.ix_(i1, i1)]
        cov_22 = self.covariance[np.ix_(i2, i2)]
        prec_22 = pinvh(cov_22)
        regression_coeffs = cov_12.dot(prec_22)

        if X.ndim == 2:
            mean = self.mean[i1] + regression_coeffs.dot(
                (X - self.mean[i2]).T).T
        elif X.ndim == 1:
            mean = self.mean[i1] + regression_coeffs.dot(X - self.mean[i2])
        else:
            raise ValueError("%d dimensions are not allowed for X!" % X.ndim)
        covariance = cov_11 - regression_coeffs.dot(cov_12.T)
        return mean, covariance

    def to_ellipse(self, factor=1.0):
        """Compute error ellipse.

        An ellipse of equiprobable points.
        """
        vals, vecs = np.linalg.eigh(self.covariance)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.arctan2(*vecs[:, 0][::-1])
        width, height = 2 * factor * np.sqrt(vals)
        return angle, width, height


def plot_error_ellipse(ax, mvn):
    from matplotlib.patches import Ellipse
    for factor in np.linspace(0.25, 2.0, 8):
        angle, width, height = mvn.to_ellipse(factor)
        ell = Ellipse(xy=mvn.mean, width=width, height=height,
                        angle=np.degrees(angle))
        ell.set_alpha(0.25)
        ax.add_artist(ell)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    random_state = check_random_state(0)
    mvn = MultivariateNormal(random_state=random_state)
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
