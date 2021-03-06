***
gmr
***

    Gaussian Mixture Models (GMMs) for clustering and regression in Python.

.. image:: https://api.travis-ci.org/AlexanderFabisch/gmr.png?branch=master
   :target: https://travis-ci.org/AlexanderFabisch/gmr
   :alt: Travis

.. image:: https://zenodo.org/badge/17119390.svg
   :target: https://zenodo.org/badge/latestdoi/17119390
   :alt: DOI (Zenodo)

Source code repository: https://github.com/AlexanderFabisch/gmr

License: `New BSD / BSD 3-clause <https://github.com/AlexanderFabisch/gmr/blob/master/LICENSE>`_

Releases: https://github.com/AlexanderFabisch/gmr/releases

.. image:: https://raw.githubusercontent.com/AlexanderFabisch/gmr/master/gmr.png

`(Source code of example) <https://github.com/AlexanderFabisch/gmr/blob/master/examples/plot_regression.py>`_

Documentation
=============

Installation
------------

Install from `PyPI`_:

.. code-block:: bash

    pip install gmr

If you want to be able to run all examples, pip can install all necessary
examples with

.. code-block::

    pip install gmr[all]

You can also install `gmr` from source:

.. code-block:: bash

    python setup.py install
    # alternatively: pip install -e .

.. _PyPi: https://pypi.python.org/pypi

Example
-------

Estimate GMM from samples, sample from GMM, and make predictions:

.. code-block:: python

    import numpy as np
    from gmr import GMM

    # Your dataset as a NumPy array of shape (n_samples, n_features):
    X = np.random.randn(100, 2)

    gmm = GMM(n_components=3, random_state=0)
    gmm.from_samples(X)

    # Estimate GMM with expectation maximization:
    X_sampled = gmm.sample(100)

    # Make predictions with known values for the first feature:
    x1 = np.random.randn(20, 1)
    x1_index = [0]
    x2_predicted_mean = gmm.predict(x1_index, x1)


For more details, see:

.. code-block:: python

    help(gmr)


How Does It Compare to scikit-learn?
------------------------------------

There is an implementation of Gaussian Mixture Models for clustering in
`scikit-learn <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture>`_
as well. Regression could not be easily integrated in the interface of
sklearn. That is the reason why I put the code in a separate repository.
It is possible to initialize GMR from sklearn though:

.. code-block:: python

    from sklearn.mixture import GaussianMixture
    from gmr import GMM
    gmm_sklearn = GaussianMixture(n_components=3, covariance_type="diag")
    gmm_sklearn.fit(X)
    gmm = GMM(
        n_components=3, priors=gmm_sklearn.weights_, means=gmm_sklearn.means_,
        covariances=np.array([np.diag(c) for c in gmm_sklearn.covariances_]))


Gallery
-------

.. image:: https://raw.githubusercontent.com/AlexanderFabisch/gmr/master/doc/sklearn_initialization.png
    :width: 60%

`Diagonal covariances <https://github.com/AlexanderFabisch/gmr/blob/master/examples/plot_iris_from_sklearn.py>`_

.. image:: https://raw.githubusercontent.com/AlexanderFabisch/gmr/master/doc/confidence_sampling.png
    :width: 60%

`Sample from confidence interval <https://github.com/AlexanderFabisch/gmr/blob/master/examples/plot_sample_mvn_confidence_interval.py>`_

.. image:: https://raw.githubusercontent.com/AlexanderFabisch/gmr/master/doc/trajectories.png
    :width: 60%

`Generate trajectories <https://github.com/AlexanderFabisch/gmr/blob/master/examples/plot_trajectories.py>`_

.. image:: https://raw.githubusercontent.com/AlexanderFabisch/gmr/master/doc/time_invariant_trajectories.png
    :width: 60%

`Sample time-invariant trajectories <https://github.com/AlexanderFabisch/gmr/blob/master/examples/plot_time_invariant_trajectories.py>`_

You can find all examples `here <https://github.com/AlexanderFabisch/gmr/tree/master/examples>`_.


Saving a Model
--------------

This library does not directly offer a function to store fitted models. Since
the implementation is pure Python, it is possible, however, to use standard
Python tools to store Python objects. For example, you can use pickle to
temporarily store a GMM:

.. code-block:: python

    import numpy as np
    import pickle
    import gmr
    gmm = gmr.GMM(n_components=2)
    gmm.from_samples(X=np.random.randn(1000, 3))

    # Save object gmm to file 'file'
    pickle.dump(gmm, open("file", "wb"))
    # Load object from file 'file'
    gmm2 = pickle.load(open("file", "rb"))

It might be required to store models more permanently than in a pickle file,
which might break with a change of the library or with the Python version.
In this case you can choose a storage format that you like and store the
attributes `gmm.priors`, `gmm.means`, and `gmm.covariances`. These can be
used in the constructor of the GMM class to recreate the object and they can
also be used in other libraries that provide a GMM implementation. The
MVN class only needs the attributes `mean` and `covariance` to define the
model.


API Documentation
=================

Functions
---------


`covariance_initialization(X, n_components)`
:   Initialize covariances.

    The standard deviation in each dimension is set to the average Euclidean
    distance of the training samples divided by the number of components.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Samples from the true distribution.

    n_components : int (> 0)
        Number of MVNs that compose the GMM.

    Returns
    -------
    initial_covariances : array, shape (n_components, n_features, n_features)
        Initial covariances


`kmeansplusplus_initialization(X, n_components, random_state=None)`
:   k-means++ initialization for centers of a GMM.

    Initialization of GMM centers before expectation maximization (EM).
    The first center is selected uniformly random. Subsequent centers are
    sampled from the data with probability proportional to the squared
    distance to the closest center.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Samples from the true distribution.

    n_components : int (> 0)
        Number of MVNs that compose the GMM.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.

    Returns
    -------
    initial_means : array, shape (n_components, n_features)
        Initial means


`plot_error_ellipse(ax, mvn, color=None, alpha=0.25, factors=array([0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ]))`
:   Plot error ellipse of MVN.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    mvn : MVN
        Multivariate normal distribution.

    color : str, optional (default: None)
        Color in which the ellipse should be plotted

    alpha : int, optional (default: 0.25)
        Alpha value for ellipse

    factors : array, optional (default: np.linspace(0.25, 2.0, 8))
        Multiples of the standard deviations that should be plotted.


`plot_error_ellipses(ax, gmm, colors=None, alpha=0.25, factors=array([0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ]))`
:   Plot error ellipses of GMM components.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    gmm : GMM
        Gaussian mixture model.

    colors : list of str, optional (default: None)
        Colors in which the ellipses should be plotted

    alpha : int, optional (default: 0.25)
        Alpha value for ellipses

    factors : array, optional (default: np.linspace(0.25, 2.0, 8))
        Multiples of the standard deviations that should be plotted.

Classes
-------

`GMM(n_components, priors=None, means=None, covariances=None, verbose=0, random_state=None)`
:   Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int
        Number of MVNs that compose the GMM.

    priors : array, shape (n_components,), optional
        Weights of the components.

    means : array, shape (n_components, n_features), optional
        Means of the components.

    covariances : array, shape (n_components, n_features, n_features), optional
        Covariances of the components.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.

    ### Methods

    `condition(self, indices, x)`
    :   Conditional distribution over given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : GMM
            Conditional GMM distribution p(Y | X=x).

    `extract_mvn(self, component_idx)`
    :   Extract one of the Gaussians from the mixture.

        Parameters
        ----------
        component_idx : int
            Index of the component that should be extracted.

        Returns
        -------
        mvn : MVN
            The component_idx-th multivariate normal distribution of this GMM.

    `from_samples(self, X, R_diff=0.0001, n_iter=100, init_params='random')`
    :   MLE of the mean and covariance.

        Expectation-maximization is used to infer the model parameters. The
        objective function is non-convex. Hence, multiple runs can have
        different results.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true distribution.

        R_diff : float
            Minimum allowed difference of responsibilities between successive
            EM iterations.

        n_iter : int
            Maximum number of iterations.

        init_params : str, optional (default: 'random')
            Parameter initialization strategy. If means and covariances are
            given in the constructor, this parameter will have no effect.
            'random' will sample initial means randomly from the dataset
            and set covariances to identity matrices. This is the
            computationally cheap solution.
            'kmeans++' will use k-means++ initialization for means and
            initialize covariances to diagonal matrices with variances
            set based on the average distances of samples in each dimensions.
            This is computationally more expensive but often gives much
            better results.

        Returns
        -------
        self : GMM
            This object.

    `is_in_confidence_region(self, x, alpha)`
    :   Check if sample is in alpha confidence region.

        Check whether the sample lies in the confidence region of the closest
        MVN according to the Mahalanobis distance.

        Parameters
        ----------
        x : array, shape (n_features,)
            Sample

        alpha : float
            Value between 0 and 1 that defines the probability of the
            confidence region, e.g., 0.6827 for the 1-sigma confidence
            region or 0.9545 for the 2-sigma confidence region.

        Returns
        -------
        is_in_confidence_region : bool
            Is the sample in the alpha confidence region?

    `predict(self, indices, X)`
    :   Predict means of posteriors.

        Same as condition() but for multiple samples.

        Parameters
        ----------
        indices : array, shape (n_features_1,)
            Indices of dimensions that we want to condition.

        X : array, shape (n_samples, n_features_1)
            Values of the features that we know.

        Returns
        -------
        Y : array, shape (n_samples, n_features_2)
            Predicted means of missing values.

    `sample(self, n_samples)`
    :   Sample from Gaussian mixture distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the GMM.

    `sample_confidence_region(self, n_samples, alpha)`
    :   Sample from alpha confidence region.

        Each MVN is selected with its prior probability and then we
        sample from the confidence region of the selected MVN.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        alpha : float
            Value between 0 and 1 that defines the probability of the
            confidence region, e.g., 0.6827 for the 1-sigma confidence
            region or 0.9545 for the 2-sigma confidence region.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the confidence region.

    `to_ellipses(self, factor=1.0)`
    :   Compute error ellipses.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        ellipses : list
            Parameters that describe the error ellipses of all components:
            mean and a tuple of angles, widths and heights. Note that widths
            and heights are semi axes, not diameters.

    `to_mvn(self)`
    :   Collapse to a single Gaussian.

        Returns
        -------
        mvn : MVN
            Multivariate normal distribution.

    `to_probability_density(self, X)`
    :   Compute probability density.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        p : array, shape (n_samples,)
            Probability densities of data.

    `to_responsibilities(self, X)`
    :   Compute responsibilities of each MVN for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        R : array, shape (n_samples, n_components)

`MVN(mean=None, covariance=None, verbose=0, random_state=None)`
:   Multivariate normal distribution.

    Some utility functions for MVNs. See
    http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    for more details.

    Parameters
    ----------
    mean : array, shape (n_features), optional
        Mean of the MVN.

    covariance : array, shape (n_features, n_features), optional
        Covariance of the MVN.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.

    ### Methods

    `condition(self, indices, x)`
    :   Conditional distribution over given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : MVN
            Conditional MVN distribution p(Y | X=x).

    `estimate_from_sigma_points(self, transformed_sigma_points, alpha=0.001, beta=2.0, kappa=0.0, random_state=None)`
    :   Estimate new MVN from sigma points through the unscented transform.

        See :func:`MVN.sigma_points` for more details.

        Parameters
        ----------
        transformed_sigma_points : array, shape (2 * n_features + 1, n_features)
            Query points that were transformed to estimate the resulting MVN.

        alpha : float, optional (default: 1e-3)
            Determines the spread of the sigma points around the mean and is
            usually set to a small positive value. Note that this value has
            to match the value that was used to create the sigma points.

        beta : float, optional (default: 2)
            Encodes information about the distribution. For Gaussian
            distributions, beta=2 is the optimal choice.

        kappa : float, optional (default: 0)
            A secondary scaling parameter which is usually set to 0. Note that
            this value has to match the value that was used to create the
            sigma points.

        random_state : int or RandomState, optional (default: random state of self)
            If an integer is given, it fixes the seed. Defaults to the global
            numpy random number generator.

        Returns
        -------
        mvn : MVN
            Transformed MVN: f(self).

    `from_samples(self, X, bessels_correction=True)`
    :   MLE of the mean and covariance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true function.

        Returns
        -------
        self : MVN
            This object.

    `is_in_confidence_region(self, x, alpha)`
    :   Check if sample is in alpha confidence region.

        Parameters
        ----------
        x : array, shape (n_features,)
            Sample

        alpha : float
            Value between 0 and 1 that defines the probability of the
            confidence region, e.g., 0.6827 for the 1-sigma confidence
            region or 0.9545 for the 2-sigma confidence region.

        Returns
        -------
        is_in_confidence_region : bool
            Is the sample in the alpha confidence region?

    `marginalize(self, indices)`
    :   Marginalize over everything except the given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to keep.

        Returns
        -------
        marginal : MVN
            Marginal MVN distribution.

    `predict(self, indices, X)`
    :   Predict means and covariance of posteriors.

        Same as condition() but for multiple samples.

        Parameters
        ----------
        indices : array, shape (n_features_1,)
            Indices of dimensions that we want to condition.

        X : array, shape (n_samples, n_features_1)
            Values of the features that we know.

        Returns
        -------
        Y : array, shape (n_samples, n_features_2)
            Predicted means of missing values.

        covariance : array, shape (n_features_2, n_features_2)
            Covariance of the predicted features.

    `sample(self, n_samples)`
    :   Sample from multivariate normal distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the MVN.

    `sample_confidence_region(self, n_samples, alpha)`
    :   Sample from alpha confidence region.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        alpha : float
            Value between 0 and 1 that defines the probability of the
            confidence region, e.g., 0.6827 for the 1-sigma confidence
            region or 0.9545 for the 2-sigma confidence region.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the confidence region.

    `sigma_points(self, alpha=0.001, kappa=0.0)`
    :   Compute sigma points for unscented transform.

        The unscented transform allows us to estimate the resulting MVN from
        applying a nonlinear transformation :math:`f` to this MVN. In order to
        do this, you have to transform the sigma points obtained from this
        function with :math:`f` and then create the new MVN with
        :func:`MVN.estimate_from_sigma_points`. The unscented transform is most
        commonly used in the Unscented Kalman Filter (UKF).

        Parameters
        ----------
        alpha : float, optional (default: 1e-3)
            Determines the spread of the sigma points around the mean and is
            usually set to a small positive value.

        kappa : float, optional (default: 0)
            A secondary scaling parameter which is usually set to 0.

        Returns
        -------
        sigma_points : array, shape (2 * n_features + 1, n_features)
            Query points that have to be transformed to estimate the resulting
            MVN.

    `squared_mahalanobis_distance(self, x)`
    :   Squared Mahalanobis distance between point and this MVN.

        Parameters
        ----------
        x : array, shape (n_features,)

        Returns
        -------
        d : float
            Squared Mahalanobis distance

    `to_ellipse(self, factor=1.0)`
    :   Compute error ellipse.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        angle : float
            Rotation angle of the ellipse.

        width : float
            Width of the ellipse (semi axis, not diameter).

        height : float
            Height of the ellipse (semi axis, not diameter).

    `to_probability_density(self, X)`
    :   Compute probability density.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        p : array, shape (n_samples,)
            Probability densities of data.


Contributing
============

How can I contribute?
---------------------

If you discover bugs, have feature requests, or want to improve the
documentation, you can open an issue at the
`issue tracker <https://github.com/AlexanderFabisch/gmr/issues>`_
of the project.

If you want to contribute code, please open a pull request via
GitHub by forking the project, committing changes to your fork,
and then opening a
`pull request <https://github.com/AlexanderFabisch/gmr/pulls>`_
from your forked branch to the main branch of `gmr`.


Development Environment
-----------------------

I would recommend to install `gmr` from source in editable mode with `pip` and
install all dependencies:

.. code-block::

    pip install -e .[all,test,doc]

You can now run tests with

    nosetests --with-coverage

The option `--with-coverage` will print a coverage report and output an
HTML overview to the folder `cover/`.

Generate Documentation
----------------------

The API documentation is generated with
`pdoc3 <https://pdoc3.github.io/pdoc/>`_. If you want to regenerate it,
you can run

.. code-block::

    pdoc gmr --skip-errors > apidoc.rst


Related Publications
====================

The first publication that presents the GMR algorithm is

    [1] Z. Ghahramani, M. I. Jordan, "Supervised learning from incomplete data via an EM approach," Advances in Neural Information Processing Systems 6, 1994, pp. 120-127, http://papers.nips.cc/paper/767-supervised-learning-from-incomplete-data-via-an-em-approach

but it does not use the term Gaussian Mixture Regression, which to my knowledge occurs first in

    [2] S. Calinon, F. Guenter and A. Billard, "On Learning, Representing, and Generalizing a Task in a Humanoid Robot," in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 37, no. 2, 2007, pp. 286-298, doi: `10.1109/TSMCB.2006.886952 <https://doi.org/10.1109/TSMCB.2006.886952>`_.

A recent survey on various regression models including GMR is the following:

    [3] F. Stulp, O. Sigaud, "Many regression algorithms, one unified model: A review," in Neural Networks, vol. 69, 2015, pp. 60-79, doi: `10.1016/j.neunet.2015.05.005 <https://doi.org/10.1016/j.neunet.2015.05.005>`_.

Sylvain Calinon has a good introduction in his `slides on nonlinear regression <http://calinon.ch/misc/EE613/EE613-slides-9.pdf>`_ for his `machine learning course <http://calinon.ch/teaching_EPFL.htm>`_.
