import numpy as np


def make_demonstrations(n_demonstrations, n_steps, sigma=0.25, mu=0.5,
                        start=np.zeros(2), goal=np.ones(2), random_state=None):
    """Generates demonstration that can be used to test imitation learning.

    Parameters
    ----------
    n_demonstrations : int
        Number of noisy demonstrations

    n_steps : int
        Number of time steps

    sigma : float, optional (default: 0.25)
        Standard deviation of noisy component

    mu : float, optional (default: 0.5)
        Mean of noisy component

    start : array, shape (2,), optional (default: 0s)
        Initial position

    goal : array, shape (2,), optional (default: 1s)
        Final position

    random_state : int
        Seed for random number generator

    Returns
    -------
    X : array, shape (n_task_dims, n_steps, n_demonstrations)
        Noisy demonstrated trajectories

    ground_truth : array, shape (n_task_dims, n_steps)
        Original trajectory
    """
    random_state = np.random.RandomState(random_state)

    X = np.empty((2, n_steps, n_demonstrations))

    # Generate ground-truth for plotting
    ground_truth = np.empty((2, n_steps))
    T = np.linspace(-0, 1, n_steps)
    ground_truth[0] = T
    ground_truth[1] = (T / 20 + 1 / (sigma * np.sqrt(2 * np.pi)) *
                       np.exp(-0.5 * ((T - mu) / sigma) ** 2))

    # Generate trajectories
    for i in range(n_demonstrations):
        noisy_sigma = sigma * random_state.normal(1.0, 0.1)
        noisy_mu = mu * random_state.normal(1.0, 0.1)
        X[0, :, i] = T
        X[1, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-0.5 * ((T - noisy_mu) /
                                         noisy_sigma) ** 2))

    # Spatial alignment
    current_start = ground_truth[:, 0]
    current_goal = ground_truth[:, -1]
    current_amplitude = current_goal - current_start
    amplitude = goal - start
    ground_truth = ((ground_truth.T - current_start) * amplitude /
                    current_amplitude + start).T

    for demo_idx in range(n_demonstrations):
        current_start = X[:, 0, demo_idx]
        current_goal = X[:, -1, demo_idx]
        current_amplitude = current_goal - current_start
        X[:, :, demo_idx] = ((X[:, :, demo_idx].T - current_start) *
                             amplitude / current_amplitude + start).T

    return X, ground_truth


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from gmr import GMM, plot_error_ellipses, kmeansplusplus_initialization, covariance_initialization, plot_error_ellipses
    from gmr.utils import check_random_state

    X, _ = make_demonstrations(
        n_demonstrations=200, n_steps=100, goal=np.array([1., 2.]),
        random_state=0)
    X = X.transpose(2, 1, 0)
    n_demonstrations, n_steps, n_task_dims = X.shape
    X_train = np.empty((n_demonstrations, n_steps, n_task_dims + 1))
    X_train[:, :, 1:] = X
    t = np.linspace(0, 1, n_steps)
    X_train[:, :, 0] = t
    X_train = X_train.reshape(n_demonstrations * n_steps, n_task_dims + 1)

    random_state = check_random_state(0)
    n_components = 5
    initial_means = kmeansplusplus_initialization(X_train, n_components, random_state)
    initial_covs = covariance_initialization(X_train, n_components)
    gmm = GMM(n_components=n_components,
            priors=np.ones(n_components, dtype=np.float) / n_components,
            means=np.copy(initial_means),
            covariances=initial_covs,
            verbose=2,
            random_state=random_state)
    gmm.from_samples(X_train, n_iter=200, max_eff_sample=0.5)

    plt.figure()
    plt.subplot(111)

    for step in t:
        conditional_gmm = gmm.condition([0], np.array([step]))
        samples = conditional_gmm.sample(100)
        #plot_error_ellipses(plt.gca(), conditional_gmm, colors=["r", "g", "b"])
        #print(conditional_gmm.priors)
        #print(conditional_gmm.means)
        #print(conditional_gmm.covariances)
        plt.scatter(samples[:, 0], samples[:, 1], s=10)

    from matplotlib.patches import Ellipse
    from itertools import cycle
    colors = cycle(["r", "g", "b"])
    for factor in np.linspace(0.5, 4.0, 8):
        new_gmm = GMM(n_components=len(gmm.means), priors=gmm.priors, means=gmm.means[:, 1:], covariances=gmm.covariances[:, 1:, 1:], random_state=gmm.random_state)
        #k = 0
        for mean, (angle, width, height) in new_gmm.to_ellipses(factor):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            ell.set_alpha(0.2)
            #ell.set_alpha(new_gmm.priors[k])
            #k += 1
            ell.set_color(next(colors))
            plt.gca().add_artist(ell)

    plt.plot(X[:, :, 0].T, X[:, :, 1].T, alpha=0.2)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
