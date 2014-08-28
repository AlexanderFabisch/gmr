import numpy as np
from gmr.utils import check_random_state
from nose.tools import assert_equal, assert_less, assert_raises
from numpy.testing import assert_array_almost_equal
from gmr import MVN, plot_error_ellipse


mean = np.array([0.0, 1.0])
covariance = np.array([[0.5, -1.0], [-1.0, 5.0]])


class AxisStub:
    def __init__(self):
        self.count = 0
        self.artists = []

    def add_artist(self, artist):
        self.artists.append(artist)
        self.count += 1


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
    mvn = MVN(mean, covariance, random_state=random_state)

    x = np.linspace(-100, 100, 201)
    X = np.vstack(map(np.ravel, np.meshgrid(x, x))).T
    p = mvn.to_probability_density(X)
    approx_int = np.sum(p) * ((x[-1] - x[0]) / 201) ** 2
    assert_less(np.abs(1.0 - approx_int), 0.01)


def test_marginal_distribution():
    """Test moments from marginal MVN."""
    random_state = check_random_state(0)
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


def test_regression():
    """Test regression with MVN."""
    random_state = check_random_state(0)

    n_samples = 100
    x = np.linspace(0, 1, n_samples)[:, np.newaxis]
    y = 3 * x + 1
    noise = random_state.randn(n_samples, 1) * 0.01
    y += noise
    samples = np.hstack((x, y))

    mvn = MVN(random_state=random_state)
    mvn.from_samples(samples)
    assert_array_almost_equal(mvn.mean, np.array([0.5, 2.5]), decimal=2)

    pred, cov = mvn.predict(np.array([0]), x)
    mse = np.sum((y - pred) ** 2) / n_samples
    assert_less(mse, 1e-3)
    assert_less(cov[0, 0], 0.01)


def test_regression_without_noise():
    """Test regression without noise with MVN."""
    random_state = check_random_state(0)

    n_samples = 10
    x = np.linspace(0, 1, n_samples)[:, np.newaxis]
    y = 3 * x + 1
    samples = np.hstack((x, y))

    mvn = MVN(random_state=random_state)
    mvn.from_samples(samples)
    assert_array_almost_equal(mvn.mean, np.array([0.5, 2.5]), decimal=2)

    pred, cov = mvn.predict(np.array([0]), x)
    mse = np.sum((y - pred) ** 2) / n_samples
    assert_less(mse, 1e-10)
    assert_less(cov[0, 0], 1e-10)


def test_plot():
    """Test plot of MVN."""
    random_state = check_random_state(0)
    mvn = MVN(mean=mean, covariance=covariance, random_state=random_state)

    ax = AxisStub()
    plot_error_ellipse(ax, mvn)
    assert_equal(ax.count, 8)


def test_uninitialized():
    """Test behavior of uninitialized MVN."""
    random_state = check_random_state(0)
    mvn = MVN(random_state=random_state)
    assert_raises(ValueError, mvn.sample, 10)
    assert_raises(ValueError, mvn.to_probability_density, np.ones((1, 1)))
    assert_raises(ValueError, mvn.marginalize, np.zeros(0))
    assert_raises(ValueError, mvn.condition, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, mvn.predict, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, mvn.to_ellipse)
    mvn = MVN(mean=np.ones(2), random_state=random_state)
    assert_raises(ValueError, mvn.sample, 10)
    assert_raises(ValueError, mvn.to_probability_density, np.ones((1, 1)))
    assert_raises(ValueError, mvn.marginalize, np.zeros(0))
    assert_raises(ValueError, mvn.condition, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, mvn.predict, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, mvn.to_ellipse)
