gmr
===

Gaussian Mixture Models (GMMs) for clustering and regression in Python.

Source code repository: https://github.com/AlexanderFabisch/gmr

Example
-------

```
    from gmr import GMM

    gmm = GMM(n_components=3, random_state=random_state)
    gmm.from_samples(X)
    X_sampled = gmm.sample(100)
```

See `help(gmr)` for more details.

Installation
------------

```
    sudo pip install gmr
```

or

```
    sudo python setup.py install
```
