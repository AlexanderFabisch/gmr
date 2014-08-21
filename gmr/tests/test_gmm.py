import numpy as np
from sklearn.utils import check_random_state
from nose.tools import assert_less
from gmr import GMM


def test_estimate_moments():
    """Test moments estimated from samples and sampling from GMM."""
    random_state = check_random_state(0)

    actual_mean1 = np.array([0.0, 1.0])
    actual_covariance1 = np.array([[0.5, -1.0], [-1.0, 5.0]])
    X1 = random_state.multivariate_normal(actual_mean1, actual_covariance1,
                                          size=(50000,))
    actual_mean2 = np.array([2.0, -1.0])
    actual_covariance2 = np.array([[5.0, 1.0], [1.0, 0.5]])
    X2 = random_state.multivariate_normal(actual_mean2, actual_covariance2,
                                          size=(50000,))
    X = np.vstack((X1, X2))
    gmm = GMM(n_components=2, random_state=random_state)
    gmm.from_samples(X)
    assert_less(np.linalg.norm(gmm.means[0] - actual_mean1), 0.005)
    assert_less(np.linalg.norm(gmm.covariances[0] - actual_covariance1), 0.01)

    X = gmm.sample(n_samples=100000)

    gmm = GMM(n_components=2, random_state=random_state)
    gmm.from_samples(X)
    assert_less(np.linalg.norm(gmm.means[0] - actual_mean1), 0.01)
    assert_less(np.linalg.norm(gmm.covariances[0] - actual_covariance1), 0.015)