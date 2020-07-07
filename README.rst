===
gmr
===

.. image:: https://api.travis-ci.org/AlexanderFabisch/gmr.png?branch=master
   :target: https://travis-ci.org/AlexanderFabisch/gmr
   :alt: Travis
.. image:: https://landscape.io/github/AlexanderFabisch/gmr/master/landscape.svg?style=flat
   :target: https://landscape.io/github/AlexanderFabisch/gmr/master
   :alt: Code Health

Gaussian Mixture Models (GMMs) for clustering and regression in Python.

Source code repository: https://github.com/AlexanderFabisch/gmr

.. image:: https://raw.githubusercontent.com/AlexanderFabisch/gmr/master/gmr.png


Example
-------

Estimate GMM from samples and sample from GMM::

    from gmr import GMM

    gmm = GMM(n_components=3, random_state=random_state)
    gmm.from_samples(X)
    X_sampled = gmm.sample(100)


For more details, see::

    help(gmr)


Installation
------------

Install from `PyPI`_::

    sudo pip install gmr

or from source::

    sudo python setup.py install

.. _PyPi: https://pypi.python.org/pypi


How Does It Compare to scikit-learn?
------------------------------------

There is an implementation of Gaussian Mixture Models for clustering in
`scikit-learn <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html>`_
as well. Regression could not be easily integrated in the interface of
sklearn. That is the reason why I put the code in a separate repository.
It is possible to initialize GMR from sklearn though::

    from sklearn.mixture import GaussianMixture
    from gmr import GMM
    gmm_sklearn = GaussianMixture(n_components=3, covariance_type="diag")
    gmm_sklearn.fit(X)
    gmm = GMM(
        n_components=3, priors=gmm_sklearn.weights_, means=gmm_sklearn.means_,
        covariances=np.array([np.diag(c) for c in gmm_sklearn.covariances_]))


Gallery
-------

.. raw:: html

    <img src="doc/sklearn_initialization.png" width="45%"><img src="doc/confidence_sampling.png" width="45%">
    <img src="doc/trajectories.png" width="45%"><img src="doc/time_invariant_trajectories.png" width="45%">


Original Publication(s)
-----------------------

The first publication that presents the GMR algorithm is

    [1] Z. Ghahramani, M. I. Jordan, "Supervised learning from incomplete data via an EM approach," Advances in Neural Information Processing Systems 6, 1994, pp. 120-127, http://papers.nips.cc/paper/767-supervised-learning-from-incomplete-data-via-an-em-approach

but it does not use the term Gaussian Mixture Regression, which to my knowledge occurs first in

    [2] S. Calinon, F. Guenter and A. Billard, "On Learning, Representing, and Generalizing a Task in a Humanoid Robot," in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 37, no. 2, 2007, pp. 286-298, doi: `10.1109/TSMCB.2006.886952 <https://doi.org/10.1109/TSMCB.2006.886952>`_.

A recent survey on various regression models including GMR is the following:

    [3] F. Stulp, O. Sigaud, "Many regression algorithms, one unified model: A review," in Neural Networks, vol. 69, 2015, pp. 60-79, doi: `10.1016/j.neunet.2015.05.005 <https://doi.org/10.1016/j.neunet.2015.05.005>`_.
