import numpy as np
from sklearn.utils import check_random_state
from nose.tools import assert_equal, assert_less
from gmr import MVN


def test_estimate_moments():
    """Test moments estimated from samples and sampling from MVN."""
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


def test_probability_density():
    """Test PDF of MVN."""
    random_state = check_random_state(0)

    mean = np.array([0.0, 1.0])
    covariance = np.array([[0.5, -1.0], [-1.0, 5.0]])
    mvn = MVN(mean, covariance, random_state=random_state)

    x = np.linspace(-100, 100, 201)
    X = np.vstack(map(np.ravel, np.meshgrid(x, x))).T
    p = mvn.to_probability_density(X)
    approx_int = np.sum(p) * ((x[-1] - x[0]) / 201) ** 2
    assert_less(np.abs(1.0 - approx_int), 0.01)


def test_marginal_distribution():
    """Test moments from marginal MVN."""
    random_state = check_random_state(0)

    mean = np.array([0.0, 1.0])
    covariance = np.array([[0.5, -1.0], [-1.0, 5.0]])
    mvn = MVN(mean=mean, covariance=covariance, random_state=random_state)

    marginalized = mvn.marginalize(np.array([0]))
    assert_equal(marginalized.mean, np.array([0.0]))
    assert_equal(marginalized.covariance, np.array([0.5]))
    marginalized = mvn.marginalize(np.array([1]))
    assert_equal(marginalized.mean, np.array([1.0]))
    assert_equal(marginalized.covariance, np.array([5.0]))


def test_conditional_distribution():
    """Test moments from conditional MVN."""
    random_state = check_random_state(0)

    mean = np.array([0.0, 1.0])
    covariance = np.array([[0.5, 0.0], [0.0, 5.0]])
    mvn = MVN(mean=mean, covariance=covariance, random_state=random_state)

    conditional = mvn.condition(np.array([1]), np.array([5.0]))
    assert_equal(conditional.mean, np.array([0.0]))
    assert_equal(conditional.covariance, np.array([0.5]))
    conditional = mvn.condition(np.array([0]), np.array([0.5]))
    assert_equal(conditional.mean, np.array([1.0]))
    assert_equal(conditional.covariance, np.array([5.0]))

def test_ellipse():
    """Test equiprobable ellipse."""
    random_state = check_random_state(0)

    mean = np.array([0.0, 1.0])
    covariance = np.array([[0.5, 0.0], [0.0, 5.0]])
    mvn = MVN(mean=mean, covariance=covariance, random_state=random_state)

    angle, width, height = mvn.to_ellipse()
    assert_equal(angle, 0.5 * np.pi)
    assert_equal(width, np.sqrt(5.0))
    assert_equal(height, np.sqrt(0.5))
