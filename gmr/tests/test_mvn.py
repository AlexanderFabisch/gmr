import numpy as np
from sklearn.utils import check_random_state
from nose.tools import assert_equal, assert_less
from gmr import MVN


def test_estimate_moments():
    random_state = check_random_state(0)

    actual_mean = np.array([0.0, 1.0])
    actual_covariance = np.array([[0.5, -1.0], [-1.0, 5.0]])
    X = random_state.multivariate_normal(actual_mean, actual_covariance,
                                         size=(100000,))
    mvn = MVN(random_state=random_state)
    mvn.from_samples(X)
    assert_less(np.linalg.norm(mvn.mean - actual_mean), 0.02)
    assert_less(np.linalg.norm(mvn.covariance - actual_covariance), 0.02)

    X2 = mvn.sample(n_samples=100000)

    mvn2 = MVN(random_state=random_state)
    mvn2.from_samples(X2)
    assert_less(np.linalg.norm(mvn2.mean - actual_mean), 0.03)
    assert_less(np.linalg.norm(mvn2.covariance - actual_covariance), 0.03)


def test_marginal_distribution():
    random_state = check_random_state(0)

    mean = np.array([0.0, 1.0])
    covariance = np.array([[0.5, -1.0], [-1.0, 5.0]])
    mvn = MVN(mean=mean, covariance=covariance, random_state=random_state)
    marginalized = mvn.marginalize(np.array([0]))
    assert_equal(marginalized.mean, np.array([0.0]))
    assert_equal(marginalized.covariance, np.array([0.5]))
