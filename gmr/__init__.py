"""
gmr
===

Gaussian Mixture Models (GMMs) for clustering and regression in Python.
"""

__version__ = "2.0.3"


from . import gmm, mvn, utils

__all__ = ["gmm", "mvn", "utils", "sklearn"]

from .mvn import MVN, plot_error_ellipse
from .gmm import (GMM, plot_error_ellipses, kmeansplusplus_initialization,
                  covariance_initialization)

__all__.extend(["MVN", "plot_error_ellipse", "GMM", "plot_error_ellipses",
                "kmeansplusplus_initialization", "covariance_initialization"])
