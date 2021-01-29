---
title: 'gmr: Gaussian Mixture Regression'
tags:
  - regression
authors:
 - name: Alexander Fabisch
   orcid: 0000-0003-2824-7956
   affiliation: 1
affiliations:
 - name: Robotics Innovation Center, DFKI GmbH
   index: 1
date: 29 January 2021
bibliography: paper.bib
---

# Summary

gmr is a Python library for Gausian mixture regression (GMR). GMR is a
regression approach that models probability distributions rather than
functions. Hence, it is possible to model multimodal mappings.

In GMR we first learn a joint probability distribution
$p(\boldsymbol{x}, \boldsymbol{y})$ of input ($\boldsymbol{x}$) and output
($\boldsymbol{y}$) variables through expectation maximization [@Dempster1977]
and then compute the conditional distribution $p(y|x)$ to make predictions.
Thus, training is the same procedure as in a standard Gaussian mixture model
(GMM).

The library gmr is fully compatible with scikit-learn [@Pedregosa2011]. It
has its own implementation of expectation maximization (EM), but it can also
be initialized with a GMM from sklearn, which means that we can also initialize
it from a Bayesian GMM of sklearn. The prediction process for regression is
not available in sklearn and will be provided by gmr.

# Background

During the training phase we learn a Gaussian mixture model

$$p(\boldsymbol{x}, \boldsymbol{y}) = \sum_{k=1}^K \pi_k \mathcal{N}_k(\boldsymbol{x}, \boldsymbol{y}|\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

through EM, where $K$ is the number of Gaussians, $0 \leq \pi_k \leq 1$ are
the priors and $\sum_{k=1}^K = 1$,
$\mathcal{N}_k(\boldsymbol{x}, \boldsymbol{y}|\boldsymbol{\mu}, \boldsymbol{\Sigma})$
are Gaussian distributions with mean $\boldsymbol{\mu}$ and covariance
$\boldsymbol{\Sigma}$.

TODO

- connection to sklearn
- examples
- references: GMR, sklearn
- math: conditioning

# References
