---
title: 'gmr: Gaussian Mixture Regression'
tags:
  - regression
authors:
 - name: Alexander Fabisch
   orcid: 0000-0003-2824-7956
   affiliation: 1,2
affiliations:
 - name: Robotics Innovation Center, DFKI GmbH
   index: 1
 - name: Robotics Research Group, University of Bremen
   index: 2
date: 16 June 2020
bibliography: paper.bib
---

# Summary

gmr is a Python library for Gausian mixture regression (GMR). GMR is a
regression approach that models probability distributions rather than
functions. Hence, it is possible to model multimodal mappings.

In GMR we first learn a joint probability distribution $p(x, y)$ of input
and output variables through expectation maximization (EM) and then compute
the conditional distribution $p(y|x)$ to make predictions. Thus, training is
the same procedure as in a standard Gaussian mixture model (GMM).

The library gmr is fully compatible with sklearn. It has its own
implementation of expectation maximization but it can also be initialized
with a GMM from sklearn, which means that we can also initialize it from a
Bayesian GMM of sklearn. The prediction process for regression is not
available in sklearn and will be provided by gmr.

TODO

- connection to sklearn
- examples
- references: GMR, sklearn
- math?

# References
