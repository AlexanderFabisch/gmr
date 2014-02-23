import numpy as np
from sklearn.utils import check_random_state
from .mvn import MVN, plot_error_ellipse


class GMM(object):
    def __init__(self, n_components, priors=None, means=None, covariances=None,
                 verbose=1, random_state=None):
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    def from_samples(self, X, responsibilities_diff=1e-4, n_iter=100):
        """EM."""
        n_samples, n_features = X.shape

        if self.priors is None:
            self.priors = np.ones(self.n_components,
                                  dtype=np.float) / self.n_components

        if self.means is None:
            # TODO k-means++
            indices = np.arange(n_samples)
            self.means = X[self.random_state.choice(indices,
                                                    self.n_components)]

        if self.covariances is None:
            self.covariances = np.ndarray((n_samples, n_features, n_features))
            for k in range(self.n_components):
                self.covariances[k] = np.eye(n_features)

        r = np.zeros((self.n_components, n_samples))  # Responsibilities
        for i in range(n_iter):
            r_prev = r.copy()
            # Expectation
            for k in range(self.n_components):
                r[k] = self.priors[k] * MVN(
                    mean=self.means[k], covariance=self.covariances[k],
                    random_state=self.random_state).to_probability_density(X)
            r /= r.sum(axis=0)

            if np.linalg.norm(r - r_prev) < responsibilities_diff:
                if self.verbose:
                    print("EM converged.")
                break

            # Maximization
            w = r.sum(axis=1)
            self.priors = w / w.sum()

            for k in range(self.n_components):
                self.means[k] = r[k].dot(X) / w[k]
                Xm = X - self.means[k]
                self.covariances[k] = (r[k, :, np.newaxis] * Xm
                                       ).T.dot(Xm) / w[k]

    def sample(self, n_samples):
        mvn_indices = self.random_state.choice(
            self.n_components, size=(n_samples,), p=self.priors)
        return np.array([
            MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).sample(n_samples=1)[0]
            for k in mvn_indices])

    def to_probability_density(self, X):
        P = [MVN(
                 mean=self.means[k], covariance=self.covariances[k],
                 random_state=self.random_state).to_probability_density(X)
             for k in range(self.n_components)]
        return np.dot(self.priors, P)

    def condition(self, indices, x):
        n_features = self.means.shape[1] - len(indices)
        priors = np.ndarray(self.n_components)
        means = np.ndarray((self.n_components, n_features))
        covariances = np.ndarray((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            mvn = MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state)
            conditioned = mvn.condition(indices, x)
            priors[k] = (self.priors[k] *
                mvn.marginalize(indices).to_probability_density(x))
            means[k] = conditioned.mean
            covariances[k] = conditioned.covariance
        priors /= priors.sum()
        return GMM(n_components=self.n_components, priors=priors, means=means,
                   covariances=covariances, random_state=self.random_state)

    def predict(self, indices, X):
        """Mean prediction."""
        Y = []
        for x in X:
            conditioned = self.condition(indices, x)
            Y.append(conditioned.priors.dot(conditioned.means))
        return np.array(Y)

    def to_ellipses(self, factor=1.0):
        """Compute error ellipses."""
        res = []
        for k in range(self.n_components):
            mvn = MVN(
                mean=self.means[k],
                covariance=self.covariances[k],
                random_state=self.random_state)
            res.append((self.means[k], mvn.to_ellipse(factor)))
        return np.array(res)


def plot_error_ellipses(ax, gmm, colors=None):
    from matplotlib.patches import Ellipse
    for factor in np.linspace(0.25, 2.0, 8):
        for i, (mean, (angle, width, height)) in enumerate(gmm.to_ellipses(factor)):
            ell = Ellipse(xy=mean, width=width, height=height,
                        angle=np.degrees(angle))
            ell.set_alpha(0.25)
            if colors and i < len(colors):
                ell.set_color(colors[i])
            ax.add_artist(ell)
