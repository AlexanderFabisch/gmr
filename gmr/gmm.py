import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import chi2
from .utils import check_random_state
from .mvn import MVN


def kmeansplusplus_initialization(X, n_components, random_state=None):
    """k-means++ initialization for centers of a GMM.

    Initialization of GMM centers before expectation maximization (EM).
    The first center is selected uniformly random. Subsequent centers are
    sampled from the data with probability proportional to the squared
    distance to the closest center.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Samples from the true distribution.

    n_components : int (> 0)
        Number of MVNs that compose the GMM.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.

    Returns
    -------
    initial_means : array, shape (n_components, n_features)
        Initial means
    """
    if n_components <= 0:
        raise ValueError("Only n_components > 0 allowed.")
    if n_components > len(X):
        raise ValueError(
            "More components (%d) than samples (%d) are not allowed."
            % (n_components, len(X)))

    random_state = check_random_state(random_state)

    all_indices = np.arange(len(X))
    selected_centers = [random_state.choice(all_indices, size=1).tolist()[0]]
    while len(selected_centers) < n_components:
        centers = np.atleast_2d(X[np.array(selected_centers, dtype=int)])
        i = _select_next_center(X, centers, random_state, selected_centers, all_indices)
        selected_centers.append(i)
    return X[np.array(selected_centers, dtype=int)]


def _select_next_center(X, centers, random_state, excluded_indices,
                        all_indices):
    """Sample with probability proportional to the squared distance to closest center."""
    squared_distances = cdist(X, centers, metric="sqeuclidean")
    selection_probability = squared_distances.max(axis=1)
    selection_probability[np.array(excluded_indices, dtype=int)] = 0.0
    selection_probability /= np.sum(selection_probability)
    return random_state.choice(all_indices, size=1, p=selection_probability)[0]


def covariance_initialization(X, n_components):
    """Initialize covariances.

    The standard deviation in each dimension is set to the average Euclidean
    distance of the training samples divided by the number of components.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Samples from the true distribution.

    n_components : int (> 0)
        Number of MVNs that compose the GMM.

    Returns
    -------
    initial_covariances : array, shape (n_components, n_features, n_features)
        Initial covariances
    """
    if n_components <= 0:
        raise ValueError(
            "It does not make sense to initialize 0 or fewer covariances.")
    n_features = X.shape[1]
    average_distances = np.empty(n_features)
    for i in range(n_features):
        average_distances[i] = np.mean(
            pdist(X[:, i, np.newaxis], metric="euclidean"))
    initial_covariances = np.empty((n_components, n_features, n_features))
    initial_covariances[:] = (np.eye(n_features) *
                              (average_distances / n_components) ** 2)
    return initial_covariances


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

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
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

    def _check_initialized(self):
        if self.priors is None:
            raise ValueError("Priors have not been initialized")
        if self.means is None:
            raise ValueError("Means have not been initialized")
        if self.covariances is None:
            raise ValueError("Covariances have not been initialized")

    def from_samples(self, X, R_diff=1e-4, n_iter=100, init_params="random"):
        """MLE of the mean and covariance.

        Expectation-maximization is used to infer the model parameters. The
        objective function is non-convex. Hence, multiple runs can have
        different results.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true distribution.

        R_diff : float
            Minimum allowed difference of responsibilities between successive
            EM iterations.

        n_iter : int
            Maximum number of iterations.

        init_params : str, optional (default: 'random')
            Parameter initialization strategy. If means and covariances are
            given in the constructor, this parameter will have no effect.
            'random' will sample initial means randomly from the dataset
            and set covariances to identity matrices. This is the
            computationally cheap solution.
            'kmeans++' will use k-means++ initialization for means and
            initialize covariances to diagonal matrices with variances
            set based on the average distances of samples in each dimensions.
            This is computationally more expensive but often gives much
            better results.

        Returns
        -------
        self : GMM
            This object.
        """
        n_samples, n_features = X.shape

        if self.priors is None:
            self.priors = np.ones(self.n_components,
                                  dtype=float) / self.n_components

        if init_params not in ["random", "kmeans++"]:
            raise ValueError("'init_params' must be 'random' or 'kmeans++' "
                             "but is '%s'" % init_params)

        if self.means is None:
            if init_params == "random":
                indices = self.random_state.choice(
                    np.arange(n_samples), self.n_components)
                self.means = X[indices]
            else:
                self.means = kmeansplusplus_initialization(
                    X, self.n_components, self.random_state)

        if self.covariances is None:
            if init_params == "random":
                self.covariances = np.empty(
                    (self.n_components, n_features, n_features))
                self.covariances[:] = np.eye(n_features)
            else:
                self.covariances = covariance_initialization(
                    X, self.n_components)

        R = np.zeros((n_samples, self.n_components))
        for _ in range(n_iter):
            R_prev = R

            # Expectation
            R = self.to_responsibilities(X)

            if np.linalg.norm(R - R_prev) < R_diff:
                if self.verbose:
                    print("EM converged.")
                break

            # Maximization
            w = R.sum(axis=0) + 10.0 * np.finfo(R.dtype).eps
            R_n = R / w
            self.priors = w / w.sum()
            self.means = R_n.T.dot(X)
            for k in range(self.n_components):
                Xm = X - self.means[k]
                self.covariances[k] = (R_n[:, k, np.newaxis] * Xm).T.dot(Xm)

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
        self._check_initialized()

        mvn_indices = self.random_state.choice(
            self.n_components, size=(n_samples,), p=self.priors)
        mvn_indices.sort()
        split_indices = np.hstack(
            ((0,), np.nonzero(np.diff(mvn_indices))[0] + 1, (n_samples,)))
        clusters = np.unique(mvn_indices)
        lens = np.diff(split_indices)
        samples = np.empty((n_samples, self.means.shape[1]))
        for i, (k, n_samples) in enumerate(zip(clusters, lens)):
            samples[split_indices[i]:split_indices[i + 1]] = MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).sample(n_samples=n_samples)
        return samples

    def sample_confidence_region(self, n_samples, alpha):
        """Sample from alpha confidence region.

        Each MVN is selected with its prior probability and then we
        sample from the confidence region of the selected MVN.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        alpha : float
            Value between 0 and 1 that defines the probability of the
            confidence region, e.g., 0.6827 for the 1-sigma confidence
            region or 0.9545 for the 2-sigma confidence region.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the confidence region.
        """
        self._check_initialized()

        mvn_indices = self.random_state.choice(
            self.n_components, size=(n_samples,), p=self.priors)
        mvn_indices.sort()
        split_indices = np.hstack(
            ((0,), np.nonzero(np.diff(mvn_indices))[0] + 1, (n_samples,)))
        clusters = np.unique(mvn_indices)
        lens = np.diff(split_indices)
        samples = np.empty((n_samples, self.means.shape[1]))
        for i, (k, n_samples) in enumerate(zip(clusters, lens)):
            samples[split_indices[i]:split_indices[i + 1]] = MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).sample_confidence_region(
                n_samples=n_samples, alpha=alpha)
        return samples

    def is_in_confidence_region(self, x, alpha):
        """Check if sample is in alpha confidence region.

        Check whether the sample lies in the confidence region of the closest
        MVN according to the Mahalanobis distance.

        Parameters
        ----------
        x : array, shape (n_features,)
            Sample

        alpha : float
            Value between 0 and 1 that defines the probability of the
            confidence region, e.g., 0.6827 for the 1-sigma confidence
            region or 0.9545 for the 2-sigma confidence region.

        Returns
        -------
        is_in_confidence_region : bool
            Is the sample in the alpha confidence region?
        """
        self._check_initialized()
        dists = [MVN(mean=self.means[k], covariance=self.covariances[k]
                     ).squared_mahalanobis_distance(x)
                 for k in range(self.n_components)]
        # we have one degree of freedom less than number of dimensions
        n_dof = len(x) - 1
        return min(dists) <= chi2(n_dof).ppf(alpha)

    def to_responsibilities(self, X):
        """Compute responsibilities of each MVN for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        R : array, shape (n_samples, n_components)
        """
        self._check_initialized()

        n_samples = X.shape[0]
        R = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            R[:, k] = self.priors[k] * MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).to_probability_density(X)
        R_norm = R.sum(axis=1)[:, np.newaxis]
        R_norm[np.where(R_norm == 0.0)] = 1.0
        R /= R_norm
        return R

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
        self._check_initialized()

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
        self._check_initialized()

        n_features = self.means.shape[1] - len(indices)
        priors = np.empty(self.n_components)
        means = np.empty((self.n_components, n_features))
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
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
        self._check_initialized()

        n_samples, n_features_1 = X.shape
        n_features_2 = self.means.shape[1] - n_features_1
        Y = np.empty((n_samples, n_features_2))
        for n in range(n_samples):
            conditioned = self.condition(indices, X[n])
            Y[n] = conditioned.priors.dot(conditioned.means)
        return Y

    def to_ellipses(self, factor=1.0):
        """Compute error ellipses.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        ellipses : list
            Parameters that describe the error ellipses of all components:
            mean and a tuple of angles, widths and heights. Note that widths
            and heights are semi axes, not diameters.
        """
        self._check_initialized()

        res = []
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            res.append((self.means[k], mvn.to_ellipse(factor)))
        return res

    def to_mvn(self):
        """Collapse to a single Gaussian.

        Returns
        -------
        mvn : MVN
            Multivariate normal distribution.
        """
        self._check_initialized()

        mean = np.sum(self.priors[:, np.newaxis] * self.means, 0)
        assert len(self.covariances)
        covariance = np.zeros_like(self.covariances[0])
        covariance += np.sum(self.priors[:, np.newaxis, np.newaxis] * self.covariances, axis=0)
        covariance += self.means.T.dot(np.diag(self.priors)).dot(self.means)
        covariance -= np.outer(mean, mean)
        return MVN(mean=mean, covariance=covariance,
                   verbose=self.verbose, random_state=self.random_state)

    def extract_mvn(self, component_idx):
        """Extract one of the Gaussians from the mixture.

        Parameters
        ----------
        component_idx : int
            Index of the component that should be extracted.

        Returns
        -------
        mvn : MVN
            The component_idx-th multivariate normal distribution of this GMM.
        """
        self._check_initialized()
        if component_idx < 0 or component_idx >= self.n_components:
            raise ValueError("Index of Gaussian must be in [%d, %d)"
                             % (0, self.n_components))
        return MVN(
            mean=self.means[component_idx],
            covariance=self.covariances[component_idx], verbose=self.verbose,
            random_state=self.random_state)


def plot_error_ellipses(ax, gmm, colors=None, alpha=0.25, factors=np.linspace(0.25, 2.0, 8)):
    """Plot error ellipses of GMM components.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    gmm : GMM
        Gaussian mixture model.

    colors : list of str, optional (default: None)
        Colors in which the ellipses should be plotted

    alpha : int, optional (default: 0.25)
        Alpha value for ellipses

    factors : array, optional (default: np.linspace(0.25, 2.0, 8))
        Multiples of the standard deviations that should be plotted.
    """
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    for factor in factors:
        for mean, (angle, width, height) in gmm.to_ellipses(factor):
            ell = Ellipse(xy=mean, width=2.0 * width, height=2.0 * height,
                          angle=np.degrees(angle))
            ell.set_alpha(alpha)
            if colors is not None:
                ell.set_color(next(colors))
            ax.add_artist(ell)
