import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
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


def _select_next_center(X, centers, random_state, excluded_indices=[], all_indices=None):
    squared_distances = cdist(X, centers, metric="sqeuclidean")
    selection_probability = squared_distances.max(axis=1)
    selection_probability[np.array(excluded_indices, dtype=int)] = 0.0
    selection_probability /= np.sum(selection_probability)
    if all_indices is None:
        all_indices = np.arange(len(X))
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
    n_features = X.shape[1]
    average_distances = np.empty(n_features)
    for i in range(n_features):
        average_distances[i] = np.mean(
            pdist(X[:, i, np.newaxis], metric="euclidean"))
    initial_covariances = np.empty((n_components, n_features, n_features))
    initial_covariances[:] = np.eye(n_features) * (average_distances / n_components) ** 2
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

    def from_samples(self, X, R_diff=1e-4, n_iter=100, reinit_means=False,
                     min_eff_sample=0, max_eff_sample=1.0):
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

        reinit_means : bool, optional (default: False)
            Reinitialize degenerated means. Checks distances between all means
            and initializes identical distributions.

        min_eff_sample : int, optional (default: 0)
            Minimum number of effective samples that is allowed to update one
            Gaussian before it will be reinitialized. 0 deactivates this.
            The number of features (n_features) is a good initial guess. Do
            not set too large values, otherwise small clusters might not be
            covered at all.

        max_eff_sample : float in [0, 1], optional (default: 1.0)
            Maximum fraction of effective samples from all samples that is
            allowed to update one Gaussian. If this threshold is surpassed
            it will be reinitialized. A value >= 1.0 will disable this.
            A value below 1 / n_components is not possible. A value between
            0.5 and 1 is recommended.

        Returns
        -------
        self : MVN
            This object.
        """
        if max_eff_sample <= 1.0 / self.n_components:
            raise ValueError(
                "max_eff_sample is too small. It must be set to at least "
                "1 / n_components.")

        n_samples, n_features = X.shape

        if self.priors is None:
            self.priors = np.ones(self.n_components,
                                  dtype=np.float) / self.n_components

        if self.means is None:
            indices = self.random_state.choice(
                np.arange(n_samples), self.n_components)
            self.means = X[indices]

        if self.covariances is None:
            self.covariances = np.empty((self.n_components, n_features,
                                         n_features))
            for k in range(self.n_components):
                self.covariances[k] = np.eye(n_features)

        initial_covariances = np.copy(self.covariances)

        R = np.zeros((n_samples, self.n_components))
        for it in range(n_iter):
            if self.verbose >= 2:
                print("Iteration #%d" % (it + 1))
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
                if self.verbose >= 2:
                    print("Component #%d" % k)

                # TODO
                #max_variance = np.diag(self.covariances[k]).max()
                #if max_variance < 

                Xm = X - self.means[k]
                self.covariances[k] = (R_n[:, k, np.newaxis] * Xm).T.dot(Xm)

                effective_samples = 1.0 / np.sum(R_n[:, k] ** 2)
                if effective_samples < min_eff_sample:
                    print("Not enough effective samples")
                    self._reinitialize_gaussian(k, X, initial_covariances)
                if effective_samples > int(max_eff_sample * n_samples):
                    print("Too many effective samples")
                    self._reinitialize_gaussian(k, X, initial_covariances)
                if self.verbose >= 2:
                    print("Effective samples %g" % effective_samples)

                eigvals, _ = np.linalg.eigh(self.covariances[k])
                eigvals[np.abs(eigvals) < np.finfo(R.dtype).eps] = 0.0
                nonzero_eigvals = np.count_nonzero(eigvals)
                if nonzero_eigvals < n_features:
                    print("Not enough nonzero eigenvalues")
                    #self.covariances[k] = initial_covariances[k]
                if self.verbose >= 2:
                    print("Nonzero eigenvalues %d" % nonzero_eigvals)
                    #print(eigvals)
                    #print(self.covariances[k])
                rank = np.linalg.matrix_rank(self.covariances[k])
                if self.verbose >= 2:
                    print("Too low rank")
                    #self.covariances[k] = initial_covariances[k]
                if rank < n_features:
                    print("Rank %d" % rank)

            if reinit_means:
                self._reinitialize_too_close_means(X, R, initial_covariances)

        return self

    def _reinitialize_too_close_means(self, X, R, initial_covariances):
        mean_distances = pdist(self.means)
        too_close_means = np.any(mean_distances < np.finfo(R.dtype).eps)
        if too_close_means:
            print("Too close means")
        mean_distances = squareform(mean_distances)
        #if self.verbose >= 2:
        #    print(mean_distances)
        if too_close_means:
            same_means = np.where(mean_distances + np.eye(self.n_components)
                                  < np.finfo(R.dtype).eps)
            # we only reset one mean at a time
            i = same_means[0][0]

            if self.verbose >= 2:
                print("Resetting mean #%d" % i)

            self._reinitialize_gaussian(i, X, initial_covariances)

    def _reinitialize_gaussian(self, i, X, initial_covariances):
            if i == 0:
                centers = self.means[1:]
            else:
                centers = np.vstack((self.means[:i], self.means[i + 1:]))
            n = _select_next_center(X, centers, self.random_state)
            self.means[i] = np.copy(X[n])

            self.covariances[i] = initial_covariances[i]

            self.priors[i] = 1.0 / self.n_components
            self.priors /= np.sum(self.priors)

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
        ellipses : array, shape (n_components, 3)
            Parameters that describe the error ellipses of all components:
            angles, widths and heights.
        """
        self._check_initialized()

        res = []
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            res.append((self.means[k], mvn.to_ellipse(factor)))
        return res


def plot_error_ellipses(ax, gmm, colors=None, alpha=0.25):
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
    """
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    for factor in np.linspace(0.5, 4.0, 8):
        for mean, (angle, width, height) in gmm.to_ellipses(factor):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            ell.set_alpha(alpha)
            if colors is not None:
                ell.set_color(next(colors))
            ax.add_artist(ell)
