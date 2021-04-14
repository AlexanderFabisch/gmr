import numpy as np

try:
    from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
    from sklearn.utils import check_X_y
    from sklearn.utils.validation import (check_is_fitted, check_array,
                                          FLOAT_DTYPES)
except ImportError:
    raise ImportError(
        "Install scikit-learn (e.g. pip install scikit-learn) to use this "
        "extension.")

from .gmm import GMM


class GaussianMixtureRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Gaussian mixture regression compatible to scikit-learn.

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

    R_diff : float, optional (default: 1e-4)
        Minimum allowed difference of responsibilities between successive
        EM iterations.

    n_iter : int, optional (default: 500)
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

    Attributes
    ----------
    gmm_ : GMM
        Underlying GMM object

    indices_ : array, shape (n_features,)
        Indices of inputs
    """

    def __init__(self, n_components, priors=None, means=None, covariances=None,
                 verbose=0, random_state=None, R_diff=1e-4, n_iter=500,
                 init_params="random"):
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.verbose = verbose
        self.random_state = random_state
        self.R_diff = R_diff
        self.n_iter = n_iter
        self.init_params = init_params

    def fit(self, X, y):
        self.gmm_ = GMM(
            self.n_components, priors=self.priors, means=self.means,
            covariances=self.covariances, verbose=self.verbose,
            random_state=self.random_state)

        X, y = check_X_y(X, y, estimator=self.gmm_, dtype=FLOAT_DTYPES,
                         multi_output=True)
        if y.ndim == 1:
            y = np.expand_dims(y, 1)

        self.indices_ = np.arange(X.shape[1])

        self.gmm_.from_samples(
            np.hstack((X, y)), R_diff=self.R_diff, n_iter=self.n_iter,
            init_params=self.init_params)
        return self

    def predict(self, X):
        check_is_fitted(self, ["gmm_", "indices_"])
        X = check_array(X, estimator=self.gmm_, dtype=FLOAT_DTYPES)

        return self.gmm_.predict(self.indices_, X)
