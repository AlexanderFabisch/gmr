import sys
import numpy as np
from gmr.utils import check_random_state
from nose.tools import assert_equal, assert_less, assert_raises, assert_in, assert_false, assert_true
from nose.plugins.skip import SkipTest
from numpy.testing import assert_array_almost_equal
try:
    # Python 2
    from cStringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO
from gmr import GMM, plot_error_ellipses, kmeansplusplus_initialization
from test_mvn import AxisStub


random_state = check_random_state(0)

means = np.array([[0.0, 1.0],
                  [2.0, -1.0]])
covariances = np.array([[[0.5, -1.0], [-1.0, 5.0]],
                        [[5.0, 1.0], [1.0, 0.5]]])
X1 = random_state.multivariate_normal(means[0], covariances[0], size=(50000,))
X2 = random_state.multivariate_normal(means[1], covariances[1], size=(50000,))
X = np.vstack((X1, X2))


def test_kmeanspp_too_few_centers():
    X = np.array([[0.0, 1.0]])
    assert_raises(ValueError, kmeansplusplus_initialization, X, 0, 0)


def test_kmeanspp_too_many_centers():
    X = np.array([[0.0, 1.0]])
    assert_raises(ValueError, kmeansplusplus_initialization, X, 2, 0)


def test_kmeanspp_one_sample():
    X = np.array([[0.0, 1.0]])
    centers = kmeansplusplus_initialization(X, 1, 0)
    assert_array_almost_equal(X, centers)


def test_kmeanspp_two_samples():
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    centers = kmeansplusplus_initialization(X, 1, 0)
    assert_in(centers[0], X)


def test_kmeanspp_two_samples_two_centers():
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    centers = kmeansplusplus_initialization(X, 2, 0)
    assert_in(centers[0], X)
    assert_in(centers[1], X)
    assert_false(centers[0, 0] == centers[1, 0])


def test_kmeanspp_six_samples_three_centers():
    X = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [100.0, 0.0],
        [0.0, 100.0]])
    centers = kmeansplusplus_initialization(X, 3, 0)
    assert_equal(len(centers), 3)
    assert_in(np.array([100.0, 0.0]), centers)
    assert_in(np.array([0.0, 100.0]), centers)
    assert_true(
        X[0] in centers or
        X[1] in centers or
        X[2] in centers or
        X[3] in centers
    )


def test_estimate_moments():
    """Test moments estimated from samples and sampling from GMM."""
    global X
    global random_state

    gmm = GMM(n_components=2, random_state=random_state)
    gmm.from_samples(X)
    assert_less(np.linalg.norm(gmm.means[0] - means[0]), 0.005)
    assert_less(np.linalg.norm(gmm.covariances[0] - covariances[0]), 0.01)
    assert_less(np.linalg.norm(gmm.means[1] - means[1]), 0.01)
    assert_less(np.linalg.norm(gmm.covariances[1] - covariances[1]), 0.03)

    X = gmm.sample(n_samples=100000)

    gmm = GMM(n_components=2, random_state=random_state)
    gmm.from_samples(X)
    assert_less(np.linalg.norm(gmm.means[0] - means[0]), 0.01)
    assert_less(np.linalg.norm(gmm.covariances[0] - covariances[0]), 0.03)
    assert_less(np.linalg.norm(gmm.means[1] - means[1]), 0.01)
    assert_less(np.linalg.norm(gmm.covariances[1] - covariances[1]), 0.04)


def test_probability_density():
    """Test PDF of GMM."""
    global X
    global random_state

    gmm = GMM(n_components=2, random_state=random_state)
    gmm.from_samples(X)

    x = np.linspace(-100, 100, 201)
    X_grid = np.vstack(map(np.ravel, np.meshgrid(x, x))).T
    p = gmm.to_probability_density(X_grid)
    approx_int = np.sum(p) * ((x[-1] - x[0]) / 201) ** 2
    assert_less(np.abs(1.0 - approx_int), 0.01)


def test_conditional_distribution():
    """Test moments from conditional GMM."""
    random_state = check_random_state(0)

    gmm = GMM(n_components=2, priors=np.array([0.5, 0.5]), means=means,
              covariances=covariances, random_state=random_state)

    conditional = gmm.condition(np.array([1]), np.array([1.0]))
    assert_array_almost_equal(conditional.means[0], np.array([0.0]))
    assert_array_almost_equal(conditional.covariances[0], np.array([[0.3]]))
    conditional = gmm.condition(np.array([0]), np.array([2.0]))
    assert_array_almost_equal(conditional.means[1], np.array([-1.0]))
    assert_array_almost_equal(conditional.covariances[1], np.array([[0.3]]))


def test_ellipses():
    """Test equiprobable ellipses."""
    random_state = check_random_state(0)

    means = np.array([[0.0, 1.0],
                      [2.0, -1.0]])
    covariances = np.array([[[0.5, 0.0], [0.0, 5.0]],
                            [[5.0, 0.0], [0.0, 0.5]]])

    gmm = GMM(n_components=2, priors=np.array([0.5, 0.5]), means=means,
              covariances=covariances, random_state=random_state)
    ellipses = gmm.to_ellipses()

    mean, (angle, width, height) = ellipses[0]
    assert_array_almost_equal(means[0], mean)
    assert_equal(angle, 0.5 * np.pi)
    assert_equal(width, np.sqrt(5.0))
    assert_equal(height, np.sqrt(0.5))

    mean, (angle, width, height) = ellipses[1]
    assert_array_almost_equal(means[1], mean)
    assert_equal(angle, -np.pi)
    assert_equal(width, np.sqrt(5.0))
    assert_equal(height, np.sqrt(0.5))


def test_regression():
    """Test regression with GMM."""
    random_state = check_random_state(0)

    n_samples = 200
    x = np.linspace(0, 2, n_samples)[:, np.newaxis]
    y1 = 3 * x[:n_samples // 2] + 1
    y2 = -3 * x[n_samples // 2:] + 7
    noise = random_state.randn(n_samples, 1) * 0.01
    y = np.vstack((y1, y2)) + noise
    samples = np.hstack((x, y))

    gmm = GMM(n_components=2, random_state=random_state)
    gmm.from_samples(samples)
    assert_array_almost_equal(gmm.priors, 0.5 * np.ones(2), decimal=2)
    assert_array_almost_equal(gmm.means[0], np.array([0.5, 2.5]), decimal=2)
    assert_array_almost_equal(gmm.means[1], np.array([1.5, 2.5]), decimal=1)

    pred = gmm.predict(np.array([0]), x)
    mse = np.sum((y - pred) ** 2) / n_samples
    assert_less(mse, 0.01)


def test_regression_with_2d_input():
    """Test regression with GMM and two-dimensional input."""
    random_state = check_random_state(0)

    n_samples = 200
    x = np.linspace(0, 2, n_samples)[:, np.newaxis]
    y1 = 3 * x[:n_samples // 2] + 1
    y2 = -3 * x[n_samples // 2:] + 7
    noise = random_state.randn(n_samples, 1) * 0.01
    y = np.vstack((y1, y2)) + noise
    samples = np.hstack((x, x[::-1], y))

    gmm = GMM(n_components=2, random_state=random_state)
    gmm.from_samples(samples)

    pred = gmm.predict(np.array([0, 1]), np.hstack((x, x[::-1])))
    mse = np.sum((y - pred) ** 2) / n_samples


def test_regression_without_noise():
    """Test regression without noise."""
    random_state = check_random_state(0)

    n_samples = 200
    x = np.linspace(0, 2, n_samples)[:, np.newaxis]
    y1 = 3 * x[:n_samples // 2] + 1
    y2 = -3 * x[n_samples // 2:] + 7
    y = np.vstack((y1, y2))
    samples = np.hstack((x, y))

    gmm = GMM(n_components=2, random_state=random_state)
    gmm.from_samples(samples)
    assert_array_almost_equal(gmm.priors, 0.5 * np.ones(2), decimal=2)
    assert_array_almost_equal(gmm.means[0], np.array([1.5, 2.5]), decimal=2)
    assert_array_almost_equal(gmm.means[1], np.array([0.5, 2.5]), decimal=1)

    pred = gmm.predict(np.array([0]), x)
    mse = np.sum((y - pred) ** 2) / n_samples
    assert_less(mse, 0.01)


def test_plot():
    """Test plot of GMM."""
    gmm = GMM(n_components=2, priors=np.array([0.5, 0.5]), means=means,
              covariances=covariances, random_state=0)

    ax = AxisStub()
    plot_error_ellipses(ax, gmm)
    assert_equal(ax.count, 16)

    ax = AxisStub()
    plot_error_ellipses(ax, gmm, colors=["r", "g"])
    assert_equal(ax.count, 16)


def test_verbose_from_samples():
    """Test verbose output."""
    global X
    random_state = check_random_state(0)

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        gmm = GMM(n_components=2, verbose=True, random_state=random_state)
        gmm.from_samples(X)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    assert("converged" in out)


def test_uninitialized():
    """Test behavior of uninitialized GMM."""
    random_state = check_random_state(0)
    gmm = GMM(n_components=2, random_state=random_state)
    assert_raises(ValueError, gmm.sample, 10)
    assert_raises(ValueError, gmm.to_probability_density, np.ones((1, 1)))
    assert_raises(ValueError, gmm.condition, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, gmm.predict, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, gmm.to_ellipses)
    gmm = GMM(n_components=2, priors=np.ones(2), random_state=random_state)
    assert_raises(ValueError, gmm.sample, 10)
    assert_raises(ValueError, gmm.to_probability_density, np.ones((1, 1)))
    assert_raises(ValueError, gmm.condition, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, gmm.predict, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, gmm.to_ellipses)
    gmm = GMM(n_components=2, priors=np.ones(2), means=np.zeros((2, 2)),
              random_state=random_state)
    assert_raises(ValueError, gmm.sample, 10)
    assert_raises(ValueError, gmm.to_probability_density, np.ones((1, 1)))
    assert_raises(ValueError, gmm.condition, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, gmm.predict, np.zeros(0), np.zeros(0))
    assert_raises(ValueError, gmm.to_ellipses)


def test_float_precision_error():
    try:
        from sklearn.datasets import load_boston
    except ImportError:
        raise SkipTest("sklearn is not available")

    boston = load_boston()
    X, y = boston.data, boston.target
    gmm = GMM(n_components=10, random_state=2016)
    gmm.from_samples(X)
