import numpy as np
from numpy.testing import assert_array_almost_equal
from gmr.utils import check_random_state


def test_sklearn_regression():
    """Test regression with GaussianMixtureRegressor."""
    try:
        from gmr.sklearn import GaussianMixtureRegressor
    except ImportError:
        pytest.skip("sklearn is not available")

    random_state = check_random_state(0)

    n_samples = 200
    x = np.linspace(0, 2, n_samples)[:, np.newaxis]
    y1 = 3 * x[:n_samples // 2] + 1
    y2 = -3 * x[n_samples // 2:] + 7
    noise = random_state.randn(n_samples, 1) * 0.01
    y = np.vstack((y1, y2)) + noise

    gmr = GaussianMixtureRegressor(n_components=2, random_state=random_state)
    gmr.fit(x, y)
    assert_array_almost_equal(gmr.gmm_.priors, 0.5 * np.ones(2), decimal=2)
    assert_array_almost_equal(gmr.gmm_.means[0], np.array([0.5, 2.5]), decimal=2)
    assert_array_almost_equal(gmr.gmm_.means[1], np.array([1.5, 2.5]), decimal=1)

    pred = gmr.predict(x)
    mse = np.sum((y - pred) ** 2) / n_samples
    assert mse < 0.01


def test_sklearn_regression_with_2d_input():
    """Test regression with GaussianMixtureRegressor and two-dimensional input."""
    try:
        from gmr.sklearn import GaussianMixtureRegressor
    except ImportError:
        pytest.skip("sklearn is not available")

    random_state = check_random_state(0)

    n_samples = 200
    x = np.linspace(0, 2, n_samples)[:, np.newaxis]
    y1 = 3 * x[:n_samples // 2] + 1
    y2 = -3 * x[n_samples // 2:] + 7
    noise = random_state.randn(n_samples, 1) * 0.01
    y = np.vstack((y1, y2)) + noise

    gmr = GaussianMixtureRegressor(n_components=2, random_state=random_state)
    gmr.fit(x, y)

    pred = gmr.predict(x)
    mse = np.sum((y - pred) ** 2) / n_samples
    assert mse < 0.01


def test_sklearn_regression_with_1d_output():
    """Test regression with GaussianMixtureRegressor and two-dimensional input."""
    try:
        from gmr.sklearn import GaussianMixtureRegressor
    except ImportError:
        pytest.skip("sklearn is not available")

    random_state = check_random_state(0)

    n_samples = 200
    x = np.linspace(0, 2, n_samples)[:, np.newaxis]
    y = 3 * x + 1
    y = y.flatten()

    gmr = GaussianMixtureRegressor(n_components=1, random_state=random_state)
    gmr.fit(x, y)

    pred = gmr.predict(x)
    mse = np.sum((y - pred) ** 2) / n_samples
    assert mse > 0.01


def test_sklearn_regression_without_noise():
    """Test regression without noise."""
    try:
        from gmr.sklearn import GaussianMixtureRegressor
    except ImportError:
        pytest.skip("sklearn is not available")

    random_state = 0

    n_samples = 200
    x = np.linspace(0, 2, n_samples)[:, np.newaxis]
    y1 = 3 * x[:n_samples // 2] + 1
    y2 = -3 * x[n_samples // 2:] + 7
    y = np.vstack((y1, y2))

    gmr = GaussianMixtureRegressor(n_components=2, random_state=random_state)
    gmr.fit(x, y)
    assert_array_almost_equal(gmr.gmm_.priors, 0.5 * np.ones(2), decimal=2)
    assert_array_almost_equal(gmr.gmm_.means[0], np.array([1.5, 2.5]), decimal=2)
    assert_array_almost_equal(gmr.gmm_.means[1], np.array([0.5, 2.5]), decimal=1)

    pred = gmr.predict(x)
    mse = np.sum((y - pred) ** 2) / n_samples
    assert mse < 0.01
