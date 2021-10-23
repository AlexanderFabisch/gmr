import numpy as np
from .utils import check_random_state
import scipy as sp
from scipy.stats import chi2, norm
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinvh


def invert_indices(n_features, indices):
    inv = np.ones(n_features, dtype=bool)
    inv[indices] = False
    inv, = np.where(inv)
    return inv


class MVN(object):
    """Multivariate normal distribution.

    Some utility functions for MVNs. See
    http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    for more details.

    Parameters
    ----------
    mean : array-like, shape (n_features), optional
        Mean of the MVN.

    covariance : array-like, shape (n_features, n_features), optional
        Covariance of the MVN.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.
    """
    def __init__(self, mean=None, covariance=None, verbose=0,
                 random_state=None):
        self.mean = mean
        self.covariance = covariance
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.norm = None

        if self.mean is not None:
            self.mean = np.asarray(self.mean)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance)

    def _check_initialized(self):
        if self.mean is None:
            raise ValueError("Mean has not been initialized")
        if self.covariance is None:
            raise ValueError("Covariance has not been initialized")

    def from_samples(self, X, bessels_correction=True):
        """MLE of the mean and covariance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true function.

        bessels_correction : bool
            Apply Bessel's correction to the covariance estimate.

        Returns
        -------
        self : MVN
            This object.
        """
        self.mean = np.mean(X, axis=0)
        bias = 0 if bessels_correction else 1
        self.covariance = np.cov(X, rowvar=0, bias=bias)
        self.norm = None
        return self

    def sample(self, n_samples):
        """Sample from multivariate normal distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the MVN.
        """
        self._check_initialized()
        return self.random_state.multivariate_normal(
            self.mean, self.covariance, size=(n_samples,))

    def sample_confidence_region(self, n_samples, alpha):
        """Sample from alpha confidence region.

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
        return np.array([self._one_sample_confidence_region(alpha)
                         for _ in range(n_samples)])

    def _one_sample_confidence_region(self, alpha):
        x = self.sample(1)[0]
        while not self.is_in_confidence_region(x, alpha):
            x = self.sample(1)[0]
        return x

    def is_in_confidence_region(self, x, alpha):
        """Check if sample is in alpha confidence region.

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
        # we have one degree of freedom less than number of dimensions
        n_dof = len(x) - 1
        if n_dof >= 1:
            return self.squared_mahalanobis_distance(x) <= chi2(n_dof).ppf(alpha)
        else:  # 1D
            lo, hi = norm.interval(
                alpha, loc=self.mean[0], scale=self.covariance[0, 0])
            return lo <= x[0] <= hi

    def to_norm_factor_and_exponents(self, X):
        """Compute normalization factor and exponents of Gaussian.

        These values can be used to compute the probability density function
        of this Gaussian: p(x) = norm_factor * np.exp(exponents).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        norm_factor : float
            Normalization factor: constant term outside of exponential
            function in probability density function of this Gaussian.

        exponents : array, shape (n_samples,)
            Exponents to compute probability density function.
        """
        self._check_initialized()

        X = np.atleast_2d(X)
        n_features = X.shape[1]

        try:
            L = sp.linalg.cholesky(self.covariance, lower=True)
        except np.linalg.LinAlgError:
            # Degenerated covariance, try to add regularization
            L = sp.linalg.cholesky(
                self.covariance + 1e-3 * np.eye(n_features), lower=True)

        X_minus_mean = X - self.mean

        if self.norm is None:
            # Suppress a determinant of 0 to avoid numerical problems
            L_det = max(sp.linalg.det(L), np.finfo(L.dtype).eps)
            self.norm = 0.5 / np.pi ** (0.5 * n_features) / L_det

        # Solve L x = (X - mean)^T for x with triangular L
        # (LL^T = Sigma), that is, x = L^T^-1 (X - mean)^T.
        # We can avoid covariance inversion when computing
        # (X - mean) Sigma^-1 (X - mean)^T  with this trick,
        # since Sigma^-1 = L^T^-1 L^-1.
        X_normalized = sp.linalg.solve_triangular(
            L, X_minus_mean.T, lower=True).T

        exponent = -0.5 * np.sum(X_normalized ** 2, axis=1)

        return self.norm, exponent

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
        norm_factor, exponents = self.to_norm_factor_and_exponents(X)
        return norm_factor * np.exp(exponents)

    def marginalize(self, indices):
        """Marginalize over everything except the given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to keep.

        Returns
        -------
        marginal : MVN
            Marginal MVN distribution.
        """
        self._check_initialized()
        return MVN(mean=self.mean[indices],
                   covariance=self.covariance[np.ix_(indices, indices)])

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
        conditional : MVN
            Conditional MVN distribution p(Y | X=x).
        """
        self._check_initialized()
        mean, covariance = condition(
            self.mean, self.covariance,
            invert_indices(self.mean.shape[0], indices), indices, x)
        return MVN(mean=mean, covariance=covariance,
                   random_state=self.random_state)

    def predict(self, indices, X):
        """Predict means and covariance of posteriors.

        Same as condition() but for multiple samples.

        Parameters
        ----------
        indices : array-like, shape (n_features_1,)
            Indices of dimensions that we want to condition.

        X : array-like, shape (n_samples, n_features_1)
            Values of the features that we know.

        Returns
        -------
        Y : array, shape (n_samples, n_features_2)
            Predicted means of missing values.

        covariance : array, shape (n_features_2, n_features_2)
            Covariance of the predicted features.
        """
        self._check_initialized()
        indices = np.asarray(indices, dtype=int)
        X = np.asarray(X)
        return condition(
            self.mean, self.covariance,
            invert_indices(self.mean.shape[0], indices), indices, X)

    def squared_mahalanobis_distance(self, x):
        """Squared Mahalanobis distance between point and this MVN.

        Parameters
        ----------
        x : array, shape (n_features,)

        Returns
        -------
        d : float
            Squared Mahalanobis distance
        """
        self._check_initialized()
        return mahalanobis(x, self.mean, np.linalg.inv(self.covariance)) ** 2

    def to_ellipse(self, factor=1.0):
        """Compute error ellipse.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        angle : float
            Rotation angle of the ellipse.

        width : float
            Width of the ellipse (semi axis, not diameter).

        height : float
            Height of the ellipse (semi axis, not diameter).
        """
        self._check_initialized()
        vals, vecs = sp.linalg.eigh(self.covariance)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.arctan2(*vecs[:, 0][::-1])
        width, height = factor * np.sqrt(vals)
        return angle, width, height

    def _sqrt_cov(self, C):
        """Compute square root of a symmetric matrix.

        Parameters
        ----------
        C : array, shape (n_features, n_features)
            Symmetric matrix.

        Returns
        -------
        sqrt(C) : array, shape (n_features, n_features)
            Square root of covariance. The square root of a square
            matrix is defined as
            :math:`\Sigma^{\frac{1}{2}} = B \sqrt(D) B^T`, where
            :math:`\Sigma = B D B^T` is the Eigen decomposition of the
            covariance.
        """
        D, B = np.linalg.eigh(C)
        # HACK: avoid numerical problems
        D = np.maximum(D, np.finfo(float).eps)
        return B.dot(np.diag(np.sqrt(D))).dot(B.T)

    def sigma_points(self, alpha=1e-3, kappa=0.0):
        """Compute sigma points for unscented transform.

        The unscented transform allows us to estimate the resulting MVN from
        applying a nonlinear transformation :math:`f` to this MVN. In order to
        do this, you have to transform the sigma points obtained from this
        function with :math:`f` and then create the new MVN with
        :func:`MVN.estimate_from_sigma_points`. The unscented transform is most
        commonly used in the Unscented Kalman Filter (UKF).

        Parameters
        ----------
        alpha : float, optional (default: 1e-3)
            Determines the spread of the sigma points around the mean and is
            usually set to a small positive value.

        kappa : float, optional (default: 0)
            A secondary scaling parameter which is usually set to 0.

        Returns
        -------
        sigma_points : array, shape (2 * n_features + 1, n_features)
            Query points that have to be transformed to estimate the resulting
            MVN.
        """
        self._check_initialized()

        n_features = len(self.mean)
        lmbda = alpha ** 2 * (n_features + kappa) - n_features
        offset = self._sqrt_cov((n_features + lmbda) * self.covariance)

        points = np.empty(((2 * n_features + 1), n_features))
        points[0, :] = self.mean
        for i in range(n_features):
            points[1 + i, :] = self.mean + offset[i]
            points[1 + n_features + i:, :] = self.mean - offset[i]
        return points

    def estimate_from_sigma_points(self, transformed_sigma_points, alpha=1e-3, beta=2.0, kappa=0.0, random_state=None):
        """Estimate new MVN from sigma points through the unscented transform.

        See :func:`MVN.sigma_points` for more details.

        Parameters
        ----------
        transformed_sigma_points : array, shape (2 * n_features + 1, n_features)
            Query points that were transformed to estimate the resulting MVN.

        alpha : float, optional (default: 1e-3)
            Determines the spread of the sigma points around the mean and is
            usually set to a small positive value. Note that this value has
            to match the value that was used to create the sigma points.

        beta : float, optional (default: 2)
            Encodes information about the distribution. For Gaussian
            distributions, beta=2 is the optimal choice.

        kappa : float, optional (default: 0)
            A secondary scaling parameter which is usually set to 0. Note that
            this value has to match the value that was used to create the
            sigma points.

        random_state : int or RandomState, optional (default: random state of self)
            If an integer is given, it fixes the seed. Defaults to the global
            numpy random number generator.

        Returns
        -------
        mvn : MVN
            Transformed MVN: f(self).
        """
        self._check_initialized()

        n_features = len(self.mean)
        lmbda = alpha ** 2 * (n_features + kappa) - n_features

        mean_weight_0 = lmbda / (n_features + lmbda)
        cov_weight_0 = lmbda / (n_features + lmbda) + (1 - alpha ** 2 + beta)
        weights_i = 1.0 / (2.0 * (n_features + lmbda))
        mean_weights = np.empty(len(transformed_sigma_points))
        mean_weights[0] = mean_weight_0
        mean_weights[1:] = weights_i
        cov_weights = np.empty(len(transformed_sigma_points))
        cov_weights[0] = cov_weight_0
        cov_weights[1:] = weights_i

        mean = np.sum(mean_weights[:, np.newaxis] * transformed_sigma_points,
                      axis=0)
        sigma_points_minus_mean = transformed_sigma_points - mean
        covariance = sigma_points_minus_mean.T.dot(
            np.diag(cov_weights)).dot(sigma_points_minus_mean)

        if random_state is None:
            random_state = self.random_state
        return MVN(mean=mean, covariance=covariance, random_state=random_state)


def plot_error_ellipse(ax, mvn, color=None, alpha=0.25,
                       factors=np.linspace(0.25, 2.0, 8)):
    """Plot error ellipse of MVN.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    mvn : MVN
        Multivariate normal distribution.

    color : str, optional (default: None)
        Color in which the ellipse should be plotted

    alpha : int, optional (default: 0.25)
        Alpha value for ellipse

    factors : array, optional (default: np.linspace(0.25, 2.0, 8))
        Multiples of the standard deviations that should be plotted.
    """
    from matplotlib.patches import Ellipse
    for factor in factors:
        angle, width, height = mvn.to_ellipse(factor)
        ell = Ellipse(xy=mvn.mean, width=2.0 * width, height=2.0 * height,
                      angle=np.degrees(angle))
        ell.set_alpha(alpha)
        if color is not None:
            ell.set_color(color)
        ax.add_artist(ell)


def regression_coefficients(covariance, i1, i2, cov_12=None):
    """Compute regression coefficients to predict conditional distribution.

    Parameters
    ----------
    covariance : array, shape (n_features, n_features)
        Covariance of MVN

    i1 : array, shape (n_features1,)
        Input feature indices

    i2 : array, shape (n_features2,)
        Output feature indices

    cov_12 : array, shape (n_features1, n_features2), optional (default: None)
        Precomputed block of the covariance matrix between input features and
        output features

    Returns
    -------
    regression_coeffs : array, shape (n_features1, n_features2)
        Regression coefficients. These can be used to compute the mean of the
        conditional distribution as
        mean[i1] + regression_coeffs.dot((X - mean[i2]).T).T
    """
    if cov_12 is None:
        cov_12 = covariance[np.ix_(i1, i2)]
    cov_22 = covariance[np.ix_(i2, i2)]
    prec_22 = pinvh(cov_22)
    return cov_12.dot(prec_22)


def condition(mean, covariance, i1, i2, X):
    """Compute conditional mean and covariance.

    Parameters
    ----------
    mean : array, shape (n_features,)
        Mean of MVN

    covariance : array, shape (n_features, n_features)
        Covariance of MVN

    i1 : array, shape (n_features1,)
        Input feature indices

    i2 : array, shape (n_features2,)
        Output feature indices

    X : array, shape (n_samples, n_features1)
        Inputs

    Returns
    -------
    mean : array, shape (n_features2,)
        Mean of the conditional distribution

    covariance : array, shape (n_features2, n_features2)
        Covariance of the conditional distribution
    """
    cov_12 = covariance[np.ix_(i1, i2)]
    cov_11 = covariance[np.ix_(i1, i1)]
    regression_coeffs = regression_coefficients(
        covariance, i1, i2, cov_12=cov_12)

    mean = mean[i1] + regression_coeffs.dot((X - mean[i2]).T).T
    covariance = cov_11 - regression_coeffs.dot(cov_12.T)
    return mean, covariance
