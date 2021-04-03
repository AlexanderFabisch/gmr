import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin

from .gmm import GMM

class GMMRegression(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Scikit-learn RegressorMixin for the GMM class.

    Parameters
    ----------

    n_components : int (> 0)
        Number of MVNs that compose the GMM.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.

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
    self : GMMRegression
        This object.
    """

    def __init__(self, n_components, priors=None, means=None, covariances=None,
                verbose=0, random_state=None, R_diff=1e-4, n_iter=500, init_params="random"):
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
        self.gmm = GMM(self.n_components, priors=self.priors, means=self.means, 
                        covariances=self.covariances, verbose=self.verbose, random_state=self.random_state)

        if y.ndim > 2:
            raise ValueError("y must have at most two dimensions.")
        elif y.ndim == 1:
            y = np.expand_dims(y, 1)
        
        if X.ndim > 2:
            raise ValueError("y must have at most two dimensions.")
        elif X.ndim == 1:
            X = np.expand_dims(X, 1)

        self._indices = np.arange(X.shape[1])

        self.gmm.from_samples(np.hstack((X, y)), 
                            R_diff=self.R_diff, n_iter=self.n_iter, init_params=self.init_params)
        return self
    
    def predict(self, X):
        if X.ndim > 2:
            raise ValueError("y must have at most two dimensions.")
        elif X.ndim == 1:
            X = np.expand_dims(X, 1)

        return self.gmm.predict(self._indices, X)