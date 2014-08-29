"""
gmr
===

Gaussian Mixture Models (GMMs) for clustering and regression in Python.
"""
from mvn import MVN, plot_error_ellipse
from gmm import GMM, plot_error_ellipses


__version__ = "1.1-git"

__all__ = ["GMM", "MVN", "plot_error_ellipse", "plot_error_ellipses"]
