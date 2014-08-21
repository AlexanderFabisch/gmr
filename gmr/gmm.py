import numpy as np
from sklearn.utils import check_random_state
from .mvn import MVN, plot_error_ellipse


class GMM(object):
    """Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int
        Number of MVNs that compose the GMM.

    priors : array, shape (n_components,), optional
        Weights of the components.

    means : array, shape (n_components, n_features), optional
        Means of the components.

    covariances : array, shape (n_components, n_features, n_features), optional
        Covariances of the components.

    verbose : int, optional
        Verbosity level.

    random_state : int or RandomState, optional
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.
    """
    def __init__(self, n_components, priors=None, means=None, covariances=None,
                 verbose=0, random_state=None):
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    def from_samples(self, X, responsibilities_diff=1e-4, n_iter=100):
        """MLE of the mean and covariance.

        Expectation-maximization is used to infer the model parameters. The
        objective function is non-convex. Hence, multiple runs can have
        different results.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true function.

        responsibilities_diff : float
            Minimum allowed difference of responsibilities between successive
            EM iterations.

        n_iter : int
            Maximum number of iterations.

        Returns
        -------
        self : MVN
            This object.
        """
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
            self.covariances = np.empty((n_samples, n_features, n_features))
            for k in range(self.n_components):
                self.covariances[k] = np.eye(n_features)

        responsibilities = np.zeros((self.n_components, n_samples))
        for i in range(n_iter):
            r_prev = responsibilities.copy()
            # Expectation
            for k in range(self.n_components):
                responsibilities[k] = self.priors[k] * MVN(
                    mean=self.means[k], covariance=self.covariances[k],
                    random_state=self.random_state).to_probability_density(X)
            responsibilities /= responsibilities.sum(axis=0)

            if np.linalg.norm(responsibilities - r_prev) < responsibilities_diff:
                if self.verbose:
                    print("EM converged.")
                break

            # Maximization
            w = responsibilities.sum(axis=1)
            self.priors = w / w.sum()

            for k in range(self.n_components):
                self.means[k] = responsibilities[k].dot(X) / w[k]
                Xm = X - self.means[k]
                self.covariances[k] = (responsibilities[k, :, np.newaxis] * Xm
                                       ).T.dot(Xm) / w[k]

        return self

    def sample(self, n_samples):
        """Sample from Gaussian mixture distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the GMM.
        """
        mvn_indices = self.random_state.choice(
            self.n_components, size=(n_samples,), p=self.priors)
        mvn_indices.sort()
        split_indices = np.hstack((
            [0], np.append(np.nonzero(np.diff(mvn_indices))[0] + 1,
                           n_samples)))
        clusters = np.unique(mvn_indices)
        lens = np.diff(split_indices)
        samples = np.empty((n_samples, self.means.shape[1]))
        for i, (k, n_samples) in enumerate(zip(clusters, lens)):
            samples[split_indices[i]:split_indices[i + 1]] = \
                MVN(mean=self.means[k], covariance=self.covariances[k],
                    random_state=self.random_state).sample(n_samples=n_samples)
        return samples

    def to_probability_density(self, X):
        """Compute probability density.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        p : array, shape (n_samples,)
            Probability densities of data.
        """
        p = [MVN(mean=self.means[k], covariance=self.covariances[k],
                 random_state=self.random_state).to_probability_density(X)
             for k in range(self.n_components)]
        return np.dot(self.priors, p)

    def condition(self, indices, x):
        """Conditional distribution over given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : GMM
            Conditional GMM distribution p(Y | X=x).
        """
        n_features = self.means.shape[1] - len(indices)
        priors = np.empty(self.n_components)
        means = np.empty((self.n_components, n_features))
        covariances = np.empty((self.n_components, n_features, n_features))
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
        """Predict means of posteriors.

        Same as condition() but for multiple samples.

        Parameters
        ----------
        indices : array, shape (n_features_1,)
            Indices of dimensions that we want to condition.

        X : array, shape (n_samples, n_features_1)
            Values of the features that we know.

        Returns
        -------
        Y : array, shape (n_samples, n_features_2)
            Predicted means of missing values.
        """
        Y = []
        for x in X:
            conditioned = self.condition(indices, x)
            Y.append(conditioned.priors.dot(conditioned.means))
        return np.array(Y)

    def to_ellipses(self, factor=1.0):
        """Compute error ellipses.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        ellipses : array, shape (n_components, 3)
            Parameters that describe the error ellipses of all components:
            angles, widths and heights.
        """
        res = []
        for k in range(self.n_components):
            mvn = MVN(
                mean=self.means[k],
                covariance=self.covariances[k],
                random_state=self.random_state)
            res.append((self.means[k], mvn.to_ellipse(factor)))
        return np.array(res)


def plot_error_ellipses(ax, gmm, colors=None):
    """Plot error ellipses of GMM components.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    gmm : GMM
        Gaussian mixture model.
    """
    from matplotlib.patches import Ellipse
    for factor in np.linspace(0.5, 4.0, 8):
        for i, (mean, (angle, width, height)) in enumerate(gmm.to_ellipses(factor)):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            ell.set_alpha(0.25)
            if colors and i < len(colors):
                ell.set_color(colors[i])
            ax.add_artist(ell)
