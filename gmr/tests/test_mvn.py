import numpy as np
from gmr.utils import check_random_state
from nose.tools import assert_equal, assert_less, assert_raises, assert_true, assert_false, assert_almost_equal
from nose import SkipTest
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


def test_in_confidence_region():
    """Test check for confidence region."""
    mvn = MVN(mean=np.array([1.0, 2.0]),
              covariance=np.array([[1.0, 0.0], [0.0, 4.0]]))

    alpha_1sigma = 0.6827
    alpha_2sigma = 0.9545

    assert_true(mvn.is_in_confidence_region(mvn.mean, alpha_1sigma))
    assert_true(mvn.is_in_confidence_region(mvn.mean + np.array([1.0, 0.0]), alpha_1sigma))
    assert_false(mvn.is_in_confidence_region(mvn.mean + np.array([1.001, 0.0]), alpha_1sigma))

    assert_true(mvn.is_in_confidence_region(mvn.mean + np.array([2.0, 0.0]), alpha_2sigma))
    assert_false(mvn.is_in_confidence_region(mvn.mean + np.array([3.0, 0.0]), alpha_2sigma))

    assert_true(mvn.is_in_confidence_region(mvn.mean + np.array([0.0, 1.0]), alpha_1sigma))
    assert_true(mvn.is_in_confidence_region(mvn.mean + np.array([0.0, 2.0]), alpha_1sigma))
    assert_false(mvn.is_in_confidence_region(mvn.mean + np.array([0.0, 3.0]), alpha_1sigma))

    assert_true(mvn.is_in_confidence_region(mvn.mean + np.array([0.0, 4.0]), alpha_2sigma))
    assert_false(mvn.is_in_confidence_region(mvn.mean + np.array([0.0, 4.001]), alpha_2sigma))


def test_sample_confidence_region():
    """Test sampling of confidence region."""
    random_state = check_random_state(42)
    mvn = MVN(mean=np.array([1.0, 2.0]),
              covariance=np.array([[1.0, 0.0], [0.0, 4.0]]),
              random_state=random_state)
    samples = mvn.sample_confidence_region(100, 0.9)
    for sample in samples:
        assert_true(mvn.is_in_confidence_region(sample, 0.9))


def test_probability_density():
    """Test PDF of MVN."""
    random_state = check_random_state(0)
    mvn = MVN(mean, covariance, random_state=random_state)

    x = np.linspace(-100, 100, 201)
    X = np.vstack(list(map(np.ravel, np.meshgrid(x, x)))).T
    p = mvn.to_probability_density(X)
    approx_int = np.sum(p) * ((x[-1] - x[0]) / 201) ** 2
    assert_less(np.abs(1.0 - approx_int), 0.01)


def test_probability_density_without_noise():
    """Test probability density of MVN with not invertible covariance."""
    random_state = check_random_state(0)

    n_samples = 10
    x = np.linspace(0, 1, n_samples)[:, np.newaxis]
    y = np.ones((n_samples, 1))
    samples = np.hstack((x, y))

    mvn = MVN(random_state=random_state)
    mvn.from_samples(samples)
    assert_array_almost_equal(mvn.mean, np.array([0.5, 1.0]), decimal=2)
    assert_equal(mvn.covariance[1, 1], 0.0)
    p_training = mvn.to_probability_density(samples)
    p_test = mvn.to_probability_density(samples + 1)
    assert_true(np.all(p_training > p_test))


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


def test_regression_with_2d_input():
    """Test regression with MVN and two-dimensional input."""
    random_state = check_random_state(0)

    n_samples = 100
    x = np.linspace(0, 1, n_samples)[:, np.newaxis]
    y = 3 * x + 1
    noise = random_state.randn(n_samples, 1) * 0.01
    y += noise
    samples = np.hstack((x, x[::-1], y))

    mvn = MVN(random_state=random_state)
    mvn.from_samples(samples)
    assert_array_almost_equal(mvn.mean, np.array([0.5, 0.5, 2.5]), decimal=2)

    x_test = np.hstack((x, x[::-1]))
    pred, cov = mvn.predict(np.array([0, 1]), x_test)
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


def test_squared_mahalanobis_distance():
    """Test Mahalanobis distance."""
    mvn = MVN(mean=np.zeros(2), covariance=np.eye(2))
    assert_almost_equal(np.sqrt(mvn.squared_mahalanobis_distance(np.zeros(2))), 0.0)
    assert_almost_equal(np.sqrt(mvn.squared_mahalanobis_distance(np.array([0, 1]))), 1.0)

    mvn = MVN(mean=np.zeros(2), covariance=4.0 * np.eye(2))
    assert_almost_equal(np.sqrt(mvn.squared_mahalanobis_distance(np.array([2, 0]))), 1.0)
    assert_almost_equal(np.sqrt(mvn.squared_mahalanobis_distance(np.array([2, 2]))), np.sqrt(2))


def test_plot():
    """Test plot of MVN."""
    random_state = check_random_state(0)
    mvn = MVN(mean=mean, covariance=covariance, random_state=random_state)

    ax = AxisStub()
    try:
        plot_error_ellipse(ax, mvn)
    except ImportError:
        raise SkipTest("matplotlib is required for this test")
    assert_equal(ax.count, 8)
    plot_error_ellipse(ax, mvn, color="r")
    assert_equal(ax.count, 16)


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


def test_unscented_transform_linear_transformation():
    """Test unscented transform with a linear transformation."""
    mvn = MVN(mean=np.zeros(2), covariance=np.eye(2), random_state=42)

    points = mvn.sigma_points()
    new_points = np.copy(points)
    new_points[:, 1] *= 10
    new_points += np.array([0.5, -3.0])

    transformed_mvn = mvn.estimate_from_sigma_points(new_points)
    assert_array_almost_equal(transformed_mvn.mean, np.array([0.5, -3.0]))
    assert_array_almost_equal(
        transformed_mvn.covariance,
        np.array([[1.0, 0.0], [0.0, 100.0]])
    )

    sample1 = transformed_mvn.sample(1)
    sample2 = mvn.estimate_from_sigma_points(new_points, random_state=42).sample(1)
    assert_array_almost_equal(sample1, sample2)


def test_unscented_transform_linear_combination():
    """Test unscented transform with a linear combination."""
    mvn = MVN(mean=np.zeros(2), covariance=np.eye(2), random_state=42)

    points = mvn.sigma_points()
    new_points = np.empty_like(points)
    new_points[:, 0] = points[:, 1]
    new_points[:, 1] = points[:, 0] - 0.5 * points[:, 1]
    new_points += np.array([-0.5, 3.0])

    transformed_mvn = mvn.estimate_from_sigma_points(new_points)
    assert_array_almost_equal(transformed_mvn.mean, np.array([-0.5, 3.0]))
    assert_array_almost_equal(
        transformed_mvn.covariance,
        np.array([[1.0, -0.5], [-0.5, 1.25]])
    )


def test_unscented_transform_projection_to_more_dimensions():
    """Test unscented transform with a projection to 3D."""
    mvn = MVN(mean=np.zeros(2), covariance=np.eye(2), random_state=42)

    points = mvn.sigma_points()

    def f(points):
        new_points = np.empty((len(points), 3))
        new_points[:, 0] = points[:, 0]
        new_points[:, 1] = points[:, 1]
        new_points[:, 2] = -0.5 * points[:, 0] + 0.5 * points[:, 1]
        new_points += np.array([-0.5, 3.0, 10.0])
        return new_points

    transformed_mvn = mvn.estimate_from_sigma_points(f(points))
    assert_array_almost_equal(transformed_mvn.mean, np.array([-0.5, 3.0, 10.0]))
    assert_array_almost_equal(
        transformed_mvn.covariance,
        np.array([[1.0, 0.0, -0.5],
                  [0.0, 1.0, 0.5],
                  [-0.5, 0.5, 0.5]])
    )


def test_unscented_transform_quadratic():
    """Test unscented transform with a quadratic transformation."""
    mvn = MVN(mean=np.zeros(2), covariance=np.eye(2), random_state=42)

    points = mvn.sigma_points(alpha=0.67, kappa=5.0)

    def f(points):
        new_points = np.empty_like(points)
        new_points[:, 0] = points[:, 0] ** 2 * np.sign(points[:, 0])
        new_points[:, 1] = points[:, 1] ** 2 * np.sign(points[:, 1])
        new_points += np.array([5.0, -3.0])
        return new_points

    transformed_mvn = mvn.estimate_from_sigma_points(f(points), alpha=0.67, kappa=5.0)
    assert_array_almost_equal(transformed_mvn.mean, np.array([5.0, -3.0]))
    assert_array_almost_equal(
        transformed_mvn.covariance,
        np.array([[3.1, 0.0], [0.0, 3.1]]),
        decimal=1
    )


def test_is_in_confidence_region_1d():
    mvn = MVN(mean=[0.0], covariance=[[1.0]])
    assert_true(mvn.is_in_confidence_region([0.0], 1.0))
