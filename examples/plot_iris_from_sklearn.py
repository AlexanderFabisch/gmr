"""
===========================
Initialize GMM from sklearn
===========================

We will cluster the Iris dataset but we will use sklearn to initialize
our GMM. sklearn allows restricted covariances such as diagonal covariances.
This is just for demonstration purposes and does not represent an example
of a particularly good fit. Take a look at `plot_iris.py` for a fit with
full covariances.
"""
print(__doc__)
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from gmr import GMM, plot_error_ellipses


X, y = load_iris(return_X_y=True)
X_pca = PCA(n_components=2, whiten=True, random_state=1).fit_transform(X)

gmm_sklearn = GaussianMixture(n_components=3, covariance_type="diag",
                              random_state=3)
gmm_sklearn.fit(X_pca)
gmm = GMM(
    n_components=3, priors=gmm_sklearn.weights_, means=gmm_sklearn.means_,
    covariances=np.array([np.diag(c) for c in gmm_sklearn.covariances_]))

plt.figure()
ax = plt.subplot(111)
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plot_error_ellipses(ax, gmm, alpha=0.1, colors=["r", "g", "b"])
plt.show()
