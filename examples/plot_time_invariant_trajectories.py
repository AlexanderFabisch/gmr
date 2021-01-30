"""
=============================================
Generate Time-Invariant Trajectories with GMR
=============================================

Make sure to run this example from gmr's root directory. An additional package
is required to load an SVG file: 'svgpathtools'. We will further use the
Bayesian GMM from sklearn to get a better fit of the data.

We will load an SVG that contains one path that will be used to generate our
training data. We extract a sequence of 2D points from the path. We will then
compute the differences between points and assume they are velocities between
those points. We fit a GMM on samples that contain four features: x- and
y-coordinate of the position and the corresponding velocity. We now have a
time-invariant representation of a trajectory. Starting from some position
(x, y) we can compute a conditional GMM over the velocities. We can sample
from the conditional GMM to generate a velocity, integrate the velocity
to obtain a new position, and repeat this procedure as long as we want.

Note that we use a "safe" sampling procedure here: we neglect Gaussians that
have low prior probability and we resample from a selected Gaussian until
we have a velocity that lies within the 70 % confidence interval of the
Gaussian. We do this to avoid divergence from the training data.
"""
print(__doc__)

from svgpathtools import svg2paths  # pip install svgpathtools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM


paths = svg2paths("examples/8_plain.svg")[0]  # only works if started from gmr's root directory
assert len(paths) == 1
path = paths[0]

points = []
for cb in path:
    for t in np.arange(0, 1, 0.2):
        p = cb.point(t)
        p = [p.real, p.imag]
        points.append(p)
points = np.array(points)

dt = 1.0

X = points[::50]
X_dot = (X[2:] - X[:-2]) / dt
X = X[1:-1]
X_train = np.hstack((X, X_dot))

random_state = np.random.RandomState(0)
n_components = 15

bgmm = BayesianGaussianMixture(
    n_components=n_components, max_iter=500,
    random_state=random_state).fit(X_train)
gmm = GMM(n_components=n_components, priors=bgmm.weights_, means=bgmm.means_,
          covariances=bgmm.covariances_, random_state=random_state)


sampled_path = []
x = np.array([75.0, 90.0])  # left bottom
sampling_dt = 0.2  # increases sampling frequency
for t in range(500):
    sampled_path.append(x)
    cgmm = gmm.condition([0, 1], x)
    # default alpha defines the confidence region (e.g., 0.7 -> 70 %)
    x_dot = cgmm.sample_confidence_region(1, alpha=0.7)[0]
    x = x + sampling_dt * x_dot
sampled_path = np.array(sampled_path)

plt.plot(X[:, 0], X[:, 1], alpha=0.2, label="Demonstration")
plt.plot(sampled_path[:, 0], sampled_path[:, 1], label="Reproduction")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.tight_layout()
plt.show()
